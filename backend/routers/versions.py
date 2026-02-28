from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any
from ..database import get_db
from ..models import GraphVersion as GraphVersionModel, ExperimentRun as ExperimentRunModel, ExperimentStatus
from ..schemas import (
    GraphVersion as GraphVersionSchema, GraphVersionCreate, StructuralDiff, ImpactAnalysis, 
    MetricDelta, ImpactLevel, NodeSchema, EdgeSchema
)

router = APIRouter(prefix="/versions", tags=["versions"])

@router.post("", response_model=GraphVersionSchema)
async def create_version(version: GraphVersionCreate, db: AsyncSession = Depends(get_db)):
    # Auto-increment version_number for this graph
    result = await db.execute(
        select(func.max(GraphVersionModel.version_number))
        .where(GraphVersionModel.graph_id == version.graph_id)
    )
    max_version = result.scalar() or 0
    new_version_number = max_version + 1
    
    # Create new version
    db_version = GraphVersionModel(
        graph_id=version.graph_id,
        version_number=new_version_number,
        parent_version_id=version.parent_version_id,
        nodes=[node.model_dump() for node in version.nodes],
        edges=[edge.model_dump() for edge in version.edges]
    )
    db.add(db_version)
    await db.commit()
    await db.refresh(db_version)
    
    return db_version

@router.get("/{version_id}", response_model=GraphVersionSchema)
async def get_version(version_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GraphVersionModel).where(GraphVersionModel.id == version_id))
    version = result.scalar_one_or_none()
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return version

@router.get("/{version_id}/lineage", response_model=List[GraphVersionSchema])
async def get_version_lineage(version_id: int, db: AsyncSession = Depends(get_db)):
    # Get the version and walk up the parent chain
    lineage = []
    current_id = version_id
    
    while current_id:
        result = await db.execute(
            select(GraphVersionModel)
            .options(selectinload(GraphVersionModel.parent_version))
            .where(GraphVersionModel.id == current_id)
        )
        version = result.scalar_one_or_none()
        
        if not version:
            break
            
        lineage.append(version)
        current_id = version.parent_version_id
    
    # Reverse to get from root to current
    lineage.reverse()
    return lineage

@router.get("/graphs/{graph_id}/lineage-tree")
async def get_graph_lineage_tree(graph_id: int, db: AsyncSession = Depends(get_db)):
    # Get all versions for this graph
    result = await db.execute(
        select(GraphVersionModel)
        .where(GraphVersionModel.graph_id == graph_id)
        .order_by(GraphVersionModel.version_number)
    )
    versions = result.scalars().all()
    
    if not versions:
        return {"id": None, "version_number": None, "children": []}
    
    # Build tree structure
    version_map = {v.id: {"id": v.id, "version_number": v.version_number, "children": []} for v in versions}
    roots = []
    
    for version in versions:
        version_dict = version_map[version.id]
        if version.parent_version_id is None:
            roots.append(version_dict)
        else:
            parent = version_map.get(version.parent_version_id)
            if parent:
                parent["children"].append(version_dict)
    
    # Return first root (there should be only one for a well-formed tree)
    return roots[0] if roots else {"id": None, "version_number": None, "children": []}

@router.post("/diff", response_model=StructuralDiff)
async def compute_diff(
    version_a_id: int, 
    version_b_id: int, 
    db: AsyncSession = Depends(get_db)
):
    # Get both versions
    result = await db.execute(
        select(GraphVersionModel).where(GraphVersionModel.id.in_([version_a_id, version_b_id]))
    )
    versions = {v.id: v for v in result.scalars().all()}
    
    if version_a_id not in versions or version_b_id not in versions:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    
    version_a = versions[version_a_id]
    version_b = versions[version_b_id]
    
    # Convert to dictionaries for easier comparison
    nodes_a = {node["id"]: node for node in version_a.nodes}
    nodes_b = {node["id"]: node for node in version_b.nodes}
    edges_a = {edge["id"]: edge for edge in version_a.edges}
    edges_b = {edge["id"]: edge for edge in version_b.edges}
    
    # Find added nodes (in B but not in A)
    added_nodes = [
        NodeSchema(**nodes_b[node_id]) 
        for node_id in nodes_b if node_id not in nodes_a
    ]
    
    # Find removed nodes (in A but not in B)
    removed_nodes = [
        NodeSchema(**nodes_a[node_id]) 
        for node_id in nodes_a if node_id not in nodes_b
    ]
    
    # Find modified nodes (in both but different)
    modified_nodes = []
    for node_id in nodes_a:
        if node_id in nodes_b:
            node_a = nodes_a[node_id]
            node_b = nodes_b[node_id]
            if node_a != node_b:  # Simple comparison
                modified_nodes.append({
                    "node_id": node_id,
                    "old_node": node_a,
                    "new_node": node_b
                })
    
    # Find added edges (in B but not in A)
    added_edges = [
        EdgeSchema(**edges_b[edge_id]) 
        for edge_id in edges_b if edge_id not in edges_a
    ]
    
    # Find removed edges (in A but not in B)
    removed_edges = [
        EdgeSchema(**edges_a[edge_id]) 
        for edge_id in edges_a if edge_id not in edges_b
    ]
    
    return StructuralDiff(
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        modified_nodes=modified_nodes,
        added_edges=added_edges,
        removed_edges=removed_edges
    )

@router.post("/impact", response_model=ImpactAnalysis)
async def compute_impact(
    version_a_id: int, 
    version_b_id: int, 
    db: AsyncSession = Depends(get_db)
):
    # Compute structural diff first
    structural_diff = await compute_diff(version_a_id, version_b_id, db)
    
    # Find best experiment for each version (lowest val_loss among completed runs)
    def get_best_experiment_query(version_id):
        return (
            select(ExperimentRunModel)
            .where(
                ExperimentRunModel.version_id == version_id,
                ExperimentRunModel.status == ExperimentStatus.COMPLETED,
                ExperimentRunModel.metrics.isnot(None)
            )
            .order_by(ExperimentRunModel.metrics["val_loss"].asc())
            .limit(1)
        )
    
    result_a = await db.execute(get_best_experiment_query(version_a_id))
    best_a = result_a.scalar_one_or_none()
    
    result_b = await db.execute(get_best_experiment_query(version_b_id))
    best_b = result_b.scalar_one_or_none()
    
    # Default values if no experiments found
    metric_delta = None
    impact_score = None
    impact_level = ImpactLevel.NEUTRAL
    
    if best_a and best_b and best_a.metrics and best_b.metrics:
        val_loss_a = best_a.metrics.get("val_loss", 1.0)
        val_loss_b = best_b.metrics.get("val_loss", 1.0)
        
        # Compute metric delta
        delta = val_loss_b - val_loss_a
        
        # Compute impact score
        if val_loss_a > 0:
            impact_score = abs(delta / val_loss_a) * 100
        else:
            impact_score = 0
        
        # Determine impact level
        if impact_score > 15:
            if delta < 0:
                impact_level = ImpactLevel.HIGH_POSITIVE
            else:
                impact_level = ImpactLevel.HIGH_NEGATIVE
        elif impact_score >= 5:
            if delta < 0:
                impact_level = ImpactLevel.LOW_POSITIVE
            else:
                impact_level = ImpactLevel.LOW_NEGATIVE
        else:
            impact_level = ImpactLevel.NEUTRAL
        
        metric_delta = MetricDelta(
            train_loss_delta=(
                best_b.metrics.get("train_loss", 0) - best_a.metrics.get("train_loss", 0)
            ),
            val_loss_delta=delta,
            accuracy_delta=(
                best_b.metrics.get("accuracy", 0) - best_a.metrics.get("accuracy", 0)
            )
        )
    
    return ImpactAnalysis(
        version_a_id=version_a_id,
        version_b_id=version_b_id,
        structural_diff=structural_diff,
        metric_delta=metric_delta,
        impact_score=impact_score,
        impact_level=impact_level
    )
