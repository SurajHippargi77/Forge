from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Graph Schemas
class GraphBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)

class GraphCreate(GraphBase):
    pass

class GraphUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)

class Graph(GraphBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# GraphVersion Schemas
class NodeSchema(BaseModel):
    id: str
    type: str
    label: str
    params: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[Dict[str, Any]] = None

class EdgeSchema(BaseModel):
    id: str
    source: str
    target: str

class GraphVersionBase(BaseModel):
    graph_id: int
    version_number: int
    parent_version_id: Optional[int] = None
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class GraphVersionCreate(BaseModel):
    graph_id: int
    parent_version_id: Optional[int] = None
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class GraphVersionUpdate(BaseModel):
    version_number: Optional[int] = None
    parent_version_id: Optional[int] = None
    nodes: Optional[List[NodeSchema]] = None
    edges: Optional[List[EdgeSchema]] = None

class GraphVersion(GraphVersionBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# ExperimentRun Schemas
class HyperparametersSchema(BaseModel):
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None

class MetricsSchema(BaseModel):
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None

class ExperimentRunBase(BaseModel):
    version_id: int
    status: ExperimentStatus = ExperimentStatus.PENDING
    hyperparameters: HyperparametersSchema

class ExperimentRunCreate(ExperimentRunBase):
    pass

class ExperimentRunUpdate(BaseModel):
    status: Optional[ExperimentStatus] = None
    hyperparameters: Optional[HyperparametersSchema] = None
    metrics: Optional[MetricsSchema] = None
    completed_at: Optional[datetime] = None

class ExperimentRun(ExperimentRunBase):
    id: int
    metrics: Optional[MetricsSchema] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# CustomAlgorithm Schemas
class CustomAlgorithmBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    code: str = Field(..., min_length=1)

class CustomAlgorithmCreate(CustomAlgorithmBase):
    pass

class CustomAlgorithmUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    code: Optional[str] = Field(None, min_length=1)

class CustomAlgorithm(CustomAlgorithmBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Analysis Schemas
class StructuralDiff(BaseModel):
    added_nodes: List[NodeSchema] = Field(default_factory=list)
    removed_nodes: List[NodeSchema] = Field(default_factory=list)
    modified_nodes: List[Dict[str, Any]] = Field(default_factory=list)  # {node_id, old_node, new_node}
    added_edges: List[EdgeSchema] = Field(default_factory=list)
    removed_edges: List[EdgeSchema] = Field(default_factory=list)

class ImpactLevel(str, Enum):
    HIGH_POSITIVE = "high_positive"
    LOW_POSITIVE = "low_positive"
    NEUTRAL = "neutral"
    LOW_NEGATIVE = "low_negative"
    HIGH_NEGATIVE = "high_negative"

class MetricDelta(BaseModel):
    train_loss_delta: Optional[float] = None
    val_loss_delta: Optional[float] = None
    accuracy_delta: Optional[float] = None

class ImpactAnalysis(BaseModel):
    version_a_id: int
    version_b_id: int
    structural_diff: StructuralDiff
    metric_delta: Optional[MetricDelta] = None
    impact_score: Optional[float] = None  # percentage scale (0-100+)
    impact_level: ImpactLevel = ImpactLevel.NEUTRAL

# Response schemas with relationships
class GraphWithVersions(Graph):
    versions: List[GraphVersion] = Field(default_factory=list)

class GraphVersionWithExperiments(GraphVersion):
    experiments: List[ExperimentRun] = Field(default_factory=list)

class GraphVersionWithLineage(GraphVersion):
    parent_version: Optional[GraphVersion] = None
    child_versions: List[GraphVersion] = Field(default_factory=list)
    experiments: List[ExperimentRun] = Field(default_factory=list)
