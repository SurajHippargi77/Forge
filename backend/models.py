from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import enum

class ExperimentStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Graph(Base):
    __tablename__ = "graphs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to versions
    versions = relationship("GraphVersion", back_populates="graph", cascade="all, delete-orphan")

class GraphVersion(Base):
    __tablename__ = "graph_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(Integer, ForeignKey("graphs.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    parent_version_id = Column(Integer, ForeignKey("graph_versions.id"), nullable=True, index=True)
    
    # JSON fields for graph structure
    nodes = Column(JSON, nullable=False)  # List of {id, type, label, params: {}}
    edges = Column(JSON, nullable=False)  # List of {id, source, target}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    graph = relationship("Graph", back_populates="versions")
    parent_version = relationship("GraphVersion", remote_side=[id])
    child_versions = relationship("GraphVersion", remote_side=[parent_version_id])
    experiments = relationship("ExperimentRun", back_populates="version", cascade="all, delete-orphan")

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("graph_versions.id"), nullable=False, index=True)
    status = Column(Enum(ExperimentStatus), nullable=False, default=ExperimentStatus.PENDING)
    
    # JSON fields for experiment data
    hyperparameters = Column(JSON, nullable=False)  # {learning_rate, batch_size, epochs, optimizer}
    metrics = Column(JSON, nullable=True)  # {train_loss, val_loss, accuracy} (filled after run)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    version = relationship("GraphVersion", back_populates="experiments")

class CustomAlgorithm(Base):
    __tablename__ = "custom_algorithms"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    code = Column(Text, nullable=False)  # Raw Python class code as string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
