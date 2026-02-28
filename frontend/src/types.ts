// Core entities
export interface Graph {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  versions?: GraphVersion[];
}

export interface Node {
  id: string;
  type: string;
  label: string;
  params: Record<string, any>;
  position: { x: number; y: number };
}

export interface Edge {
  id: string;
  source: string;
  target: string;
}

export interface GraphVersion {
  id: number;
  graph_id: number;
  version_number: number;
  parent_version_id: number | null;
  nodes: Node[];
  edges: Edge[];
  created_at: string;
  experiments?: ExperimentRun[];
}

export interface Hyperparameters {
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  optimizer?: string;
}

export interface Metrics {
  train_loss?: number;
  val_loss?: number;
  accuracy?: number;
}

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface ExperimentRun {
  id: number;
  version_id: number;
  status: ExperimentStatus;
  hyperparameters: Hyperparameters;
  metrics: Metrics | null;
  created_at: string;
  completed_at: string | null;
}

export interface CustomAlgorithm {
  id: number;
  name: string;
  description: string | null;
  code: string;
  created_at: string;
}

// Analysis types
export interface StructuralDiff {
  added_nodes: Node[];
  removed_nodes: Node[];
  modified_nodes: Array<{
    node_id: string;
    old_node: Node;
    new_node: Node;
  }>;
  added_edges: Edge[];
  removed_edges: Edge[];
}

export interface MetricDelta {
  train_loss_delta?: number;
  val_loss_delta?: number;
  accuracy_delta?: number;
}

export type ImpactLevel = 'high_positive' | 'low_positive' | 'neutral' | 'low_negative' | 'high_negative';

export interface ImpactAnalysis {
  version_a_id: number;
  version_b_id: number;
  structural_diff: StructuralDiff;
  metric_delta: MetricDelta | null;
  impact_score: number | null;
  impact_level: ImpactLevel;
}

// API Request/Response types
export interface CreateGraphRequest {
  name: string;
  description?: string;
}

export interface CreateVersionRequest {
  graph_id: number;
  parent_version_id?: number;
  nodes: Node[];
  edges: Edge[];
}

export interface CreateExperimentRequest {
  version_id: number;
  hyperparameters: Hyperparameters;
}

export interface RunSweepRequest {
  version_id: number;
  param_grid: {
    learning_rate?: number[];
    batch_size?: number[];
    epochs?: number[];
    optimizer?: string[];
  };
}

export interface SaveAlgorithmRequest {
  name: string;
  description?: string;
  code: string;
}

export interface DiffRequest {
  version_a_id: number;
  version_b_id: number;
}

export interface ImpactRequest {
  version_a_id: number;
  version_b_id: number;
}

// Lineage tree structure
export interface LineageTreeNode {
  id: number | null;
  version_number: number | null;
  children: LineageTreeNode[];
}

// Store state interface
export interface StoreState {
  // Data
  graphs: Graph[];
  activeGraph: Graph | null;
  activeVersion: GraphVersion | null;
  versions: GraphVersion[];
  experiments: ExperimentRun[];
  lineageTree: LineageTreeNode | null;
  diffResult: StructuralDiff | null;
  impactResult: ImpactAnalysis | null;
  algorithms: CustomAlgorithm[];
  
  // UI state
  isLoading: boolean;
  error: string | null;
}

// Error types
export interface ApiError {
  detail: string;
  status?: number;
}

// Response wrapper types
export interface ApiResponse<T> {
  data: T;
  message?: string;
}

// Polling types
export interface PollingOptions {
  interval?: number;
  maxAttempts?: number;
  onComplete?: () => void;
  onError?: (error: string) => void;
}
