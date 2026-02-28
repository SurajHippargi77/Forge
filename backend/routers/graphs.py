from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from typing import List
from ..database import get_db
from ..models import Graph as GraphModel, GraphVersion as GraphVersionModel
from ..schemas import Graph, GraphCreate, GraphUpdate, GraphWithVersions, GraphVersionCreate

router = APIRouter(prefix="/graphs", tags=["graphs"])

@router.post("", response_model=Graph)
async def create_graph(graph: GraphCreate, db: AsyncSession = Depends(get_db)):
    # Create new graph
    db_graph = GraphModel(
        name=graph.name,
        description=graph.description
    )
    db.add(db_graph)
    await db.flush()  # Get the ID without committing
    
    # Auto-create Version 1 with empty nodes/edges
    version_1 = GraphVersionModel(
        graph_id=db_graph.id,
        version_number=1,
        parent_version_id=None,
        nodes=[],
        edges=[]
    )
    db.add(version_1)
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
    # Check if graph exists
    result = await db.execute(select(GraphModel).where(GraphModel.id == graph_id))
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Delete graph (cascade will delete versions and experiments)
    await db.execute(delete(GraphModel).where(GraphModel.id == graph_id))
    await db.commit()
    
    return {"message": "Graph deleted successfully"}
