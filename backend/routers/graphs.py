from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List
from ..database import get_db
from ..models import Graph as GraphModel, GraphVersion as GraphVersionModel, ExperimentRun as ExperimentRunModel
from ..schemas import Graph, GraphCreate, GraphUpdate, GraphWithVersions

router = APIRouter(prefix="/graphs", tags=["graphs"])

@router.post("", response_model=Graph)
async def create_graph(graph: GraphCreate, db: AsyncSession = Depends(get_db)):
    # Create new graph (no auto-version — user must explicitly save)
    db_graph = GraphModel(
        name=graph.name,
        description=graph.description
    )
    db.add(db_graph)
    await db.commit()
    await db.refresh(db_graph)
    
    return db_graph

@router.get("", response_model=List[GraphWithVersions])
async def get_graphs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GraphModel).options(selectinload(GraphModel.versions))
    )
    graphs = result.scalars().unique().all()
    return graphs

@router.get("/{graph_id}", response_model=GraphWithVersions)
async def get_graph(graph_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GraphModel)
        .options(selectinload(GraphModel.versions))
        .where(GraphModel.id == graph_id)
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    return graph

@router.put("/{graph_id}", response_model=Graph)
async def update_graph(graph_id: int, graph: GraphUpdate, db: AsyncSession = Depends(get_db)):
    # Get existing graph
    result = await db.execute(select(GraphModel).where(GraphModel.id == graph_id))
    db_graph = result.scalar_one_or_none()
    
    if not db_graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Update fields if provided
    if graph.name is not None:
        db_graph.name = graph.name
    if graph.description is not None:
        db_graph.description = graph.description
    
    await db.commit()
    await db.refresh(db_graph)
    return db_graph

@router.delete("/{graph_id}")
async def delete_graph(graph_id: int, db: AsyncSession = Depends(get_db)):
    # Eagerly load graph with versions and their experiments for proper cascade
    result = await db.execute(
        select(GraphModel)
        .options(
            selectinload(GraphModel.versions)
            .selectinload(GraphVersionModel.experiments)
        )
        .where(GraphModel.id == graph_id)
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Null out self-referential parent_version_id to avoid FK conflicts
    for version in graph.versions:
        version.parent_version_id = None
    await db.flush()
    
    # Now delete — ORM cascade removes versions & experiments
    await db.delete(graph)
    await db.commit()
    
    return {"message": "Graph deleted successfully"}
