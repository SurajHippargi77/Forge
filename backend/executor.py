import asyncio
import random
import time
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from .database import AsyncSessionLocal
from .models import ExperimentRun, ExperimentStatus
from .schemas import MetricsSchema

async def execute_experiment(experiment_id: int):
    """
    Simulate training execution for an experiment.
    This runs in the background and updates the experiment with metrics.
    """
    async with AsyncSessionLocal() as db:
        try:
            # Mark as running
            await db.execute(
                update(ExperimentRun)
                .where(ExperimentRun.id == experiment_id)
                .values(status=ExperimentStatus.RUNNING)
            )
            await db.commit()
            
            # Get experiment hyperparameters
            result = await db.execute(
                select(ExperimentRun).where(ExperimentRun.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return
            
            hyperparams = experiment.hyperparameters
            learning_rate = hyperparams.get("learning_rate", 0.01)
            batch_size = hyperparams.get("batch_size", 32)
            epochs = hyperparams.get("epochs", 10)
            optimizer = hyperparams.get("optimizer", "adam")
            
            # Simulate training time (3-8 seconds)
            training_time = random.uniform(3.0, 8.0)
            await asyncio.sleep(training_time)
            
            # Generate realistic fake metrics
            # Base loss influenced by hyperparameters
            base_loss = 0.8
            
            # Learning rate impact (lower LR = higher loss, but more stable)
            lr_impact = (0.05 - learning_rate) * 2
            
            # Batch size impact (larger batch = slightly better loss)
            batch_impact = (batch_size - 32) / 1000
            
            # Epochs impact (more epochs = lower loss, but diminishing returns)
            epoch_impact = -min(epochs / 100, 0.3)
            
            # Optimizer impact
            optimizer_impact = 0
            if optimizer == "adam":
                optimizer_impact = -0.05
            elif optimizer == "sgd":
                optimizer_impact = 0.02
            elif optimizer == "rmsprop":
                optimizer_impact = -0.03
            
            # Add random noise
            noise = random.uniform(-0.1, 0.1)
            
            # Calculate train_loss
            train_loss = base_loss + lr_impact + batch_impact + epoch_impact + optimizer_impact + noise
            
            # Clamp to realistic range
            train_loss = max(0.1, min(2.0, train_loss))
            
            # Val loss is train loss + small offset
            val_offset = random.uniform(0.05, 0.2)
            val_loss = train_loss + val_offset
            
            # Accuracy inversely related to loss
            accuracy_base = 0.5
            accuracy_factor = max(0, (1.5 - val_loss)) * 0.4
            accuracy_noise = random.uniform(-0.05, 0.05)
            accuracy = accuracy_base + accuracy_factor + accuracy_noise
            
            # Clamp accuracy
            accuracy = max(0.1, min(0.99, accuracy))
            
            # Create metrics object
            metrics = MetricsSchema(
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
                accuracy=round(accuracy, 4)
            )
            
            # Update experiment with results
            await db.execute(
                update(ExperimentRun)
                .where(ExperimentRun.id == experiment_id)
                .values(
                    status=ExperimentStatus.COMPLETED,
                    metrics=metrics.model_dump(),
                    completed_at=datetime.utcnow()
                )
            )
            await db.commit()
            
        except Exception as e:
            # Mark as failed on any error
            await db.execute(
                update(ExperimentRun)
                .where(ExperimentRun.id == experiment_id)
                .values(status=ExperimentStatus.FAILED)
            )
            await db.commit()
            print(f"Experiment {experiment_id} failed: {str(e)}")
