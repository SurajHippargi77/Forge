from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
import itertools
from ..database import get_db
from ..models import ExperimentRun as ExperimentRunModel, ExperimentStatus
from ..schemas import ExperimentRun as ExperimentRunSchema, ExperimentRunCreate, ExperimentRunUpdate, HyperparametersSchema
from ..executor import execute_experiment

router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("", response_model=ExperimentRunSchema)
async def create_experiment(
    experiment: ExperimentRunCreate, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    # Create experiment with pending status
    db_experiment = ExperimentRunModel(
        version_id=experiment.version_id,
        status=ExperimentStatus.PENDING,
        hyperparameters=experiment.hyperparameters.model_dump()
    )
    db.add(db_experiment)
    await db.commit()
    await db.refresh(db_experiment)
    
    # Trigger background execution
    background_tasks.add_task(execute_experiment, db_experiment.id)
    
    return db_experiment

@router.get("/{experiment_id}", response_model=ExperimentRunSchema)
async def get_experiment(experiment_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExperimentRunModel).where(ExperimentRunModel.id == experiment_id))
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment

@router.get("/versions/{version_id}/experiments", response_model=List[ExperimentRunSchema])
async def get_version_experiments(version_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ExperimentRunModel)
        .where(ExperimentRunModel.version_id == version_id)
        .order_by(
            # Sort by val_loss ascending for completed experiments, then by created_at
            ExperimentRunModel.metrics["val_loss"].asc().nulls_last(),
            ExperimentRunModel.created_at.desc()
        )
    )
    experiments = result.scalars().all()
    return experiments

@router.post("/sweep", response_model=List[int])
async def create_parameter_sweep(
    version_id: int,
    param_grid: Dict[str, List[Any]],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    # Extract parameter lists from grid
    learning_rates = param_grid.get("learning_rate", [0.01])
    batch_sizes = param_grid.get("batch_size", [32])
    epochs_list = param_grid.get("epochs", [10])
    optimizers = param_grid.get("optimizer", ["adam"])
    
    # Generate all combinations using itertools.product
    combinations = list(itertools.product(
        learning_rates, batch_sizes, epochs_list, optimizers
    ))
    
    if not combinations:
        raise HTTPException(status_code=400, detail="No valid parameter combinations found")
    
    created_experiment_ids = []
    
    # Create experiment for each combination
    for lr, batch_size, epochs, optimizer in combinations:
        hyperparams = HyperparametersSchema(
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer
        )
        
        experiment = ExperimentRunModel(
            version_id=version_id,
            status=ExperimentStatus.PENDING,
            hyperparameters=hyperparams.model_dump()
        )
        db.add(experiment)
        await db.flush()  # Get the ID without committing
        created_experiment_ids.append(experiment.id)
        
        # Add background task for this experiment
        background_tasks.add_task(execute_experiment, experiment.id)
    
    await db.commit()
    
    return created_experiment_ids

@router.put("/{experiment_id}", response_model=ExperimentRunSchema)
async def update_experiment(
    experiment_id: int, 
    experiment: ExperimentRunUpdate, 
    db: AsyncSession = Depends(get_db)
):
    # Get existing experiment
    result = await db.execute(select(ExperimentRunModel).where(ExperimentRunModel.id == experiment_id))
    db_experiment = result.scalar_one_or_none()
    
    if not db_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Update fields if provided
    if experiment.status is not None:
        db_experiment.status = experiment.status
    if experiment.hyperparameters is not None:
        db_experiment.hyperparameters = experiment.hyperparameters.model_dump()
    if experiment.metrics is not None:
        db_experiment.metrics = experiment.metrics.model_dump()
    if experiment.completed_at is not None:
        db_experiment.completed_at = experiment.completed_at
    
    await db.commit()
    await db.refresh(db_experiment)
    return db_experiment

@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: int, db: AsyncSession = Depends(get_db)):
    # Check if experiment exists
    result = await db.execute(select(ExperimentRunModel).where(ExperimentRunModel.id == experiment_id))
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Delete experiment
    await db.delete(experiment)
    await db.commit()
    
    return {"message": "Experiment deleted successfully"}
