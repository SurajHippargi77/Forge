from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List
from ..database import get_db
from ..models import CustomAlgorithm as CustomAlgorithmModel
from ..schemas import CustomAlgorithm, CustomAlgorithmCreate, CustomAlgorithmUpdate

router = APIRouter(prefix="/algorithms", tags=["algorithms"])

def validate_algorithm_code(code: str) -> bool:
    """
    Validate that the code string contains required elements for a custom algorithm.
    """
    code_lower = code.lower()
    
    # Check for required keywords
    has_class = "class" in code_lower
    has_forward = "def forward" in code_lower
    
    return has_class and has_forward

@router.post("", response_model=CustomAlgorithm)
async def create_algorithm(algorithm: CustomAlgorithmCreate, db: AsyncSession = Depends(get_db)):
    # Validate algorithm code
    if not validate_algorithm_code(algorithm.code):
        raise HTTPException(
            status_code=400, 
            detail="Algorithm code must contain 'class' and 'def forward' methods"
        )
    
    # Create new algorithm
    db_algorithm = CustomAlgorithmModel(
        name=algorithm.name,
        description=algorithm.description,
        code=algorithm.code
    )
    db.add(db_algorithm)
    await db.commit()
    await db.refresh(db_algorithm)
    
    return db_algorithm

@router.get("", response_model=List[CustomAlgorithm])
async def get_algorithms(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CustomAlgorithmModel))
    algorithms = result.scalars().all()
    return algorithms

@router.get("/{algorithm_id}", response_model=CustomAlgorithm)
async def get_algorithm(algorithm_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CustomAlgorithmModel).where(CustomAlgorithmModel.id == algorithm_id))
    algorithm = result.scalar_one_or_none()
    
    if not algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    return algorithm

@router.put("/{algorithm_id}", response_model=CustomAlgorithm)
async def update_algorithm(
    algorithm_id: int, 
    algorithm: CustomAlgorithmUpdate, 
    db: AsyncSession = Depends(get_db)
):
    # Get existing algorithm
    result = await db.execute(select(CustomAlgorithmModel).where(CustomAlgorithmModel.id == algorithm_id))
    db_algorithm = result.scalar_one_or_none()
    
    if not db_algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # Update fields if provided
    if algorithm.name is not None:
        db_algorithm.name = algorithm.name
    if algorithm.description is not None:
        db_algorithm.description = algorithm.description
    if algorithm.code is not None:
        # Validate new code
        if not validate_algorithm_code(algorithm.code):
            raise HTTPException(
                status_code=400, 
                detail="Algorithm code must contain 'class' and 'def forward' methods"
            )
        db_algorithm.code = algorithm.code
    
    await db.commit()
    await db.refresh(db_algorithm)
    return db_algorithm

@router.delete("/{algorithm_id}")
async def delete_algorithm(algorithm_id: int, db: AsyncSession = Depends(get_db)):
    # Check if algorithm exists
    result = await db.execute(select(CustomAlgorithmModel).where(CustomAlgorithmModel.id == algorithm_id))
    algorithm = result.scalar_one_or_none()
    
    if not algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # Delete algorithm
    await db.execute(delete(CustomAlgorithmModel).where(CustomAlgorithmModel.id == algorithm_id))
    await db.commit()
    
    return {"message": "Algorithm deleted successfully"}
