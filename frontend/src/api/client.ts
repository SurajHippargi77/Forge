import axios from 'axios';
import type { AxiosInstance, AxiosResponse } from 'axios';
import type {
  Graph,
  GraphVersion,
  ExperimentRun,
  CustomAlgorithm,
  StructuralDiff,
  ImpactAnalysis,
  Node,
  Edge,
  CreateGraphRequest,
  CreateVersionRequest,
  CreateExperimentRequest,
  SaveAlgorithmRequest,
  LineageTreeNode,
  ApiError
} from '../types';

const BASE_URL = 'http://localhost:8000/api/v1';
const ROOT_URL = 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        return response;
      },
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: any): ApiError {
    if (error.response) {
      return {
        detail: error.response.data?.detail || 'Server error',
        status: error.response.status,
      };
    } else if (error.request) {
      return {
        detail: 'Network error - unable to reach server',
      };
    } else {
      return {
        detail: error.message || 'Unknown error occurred',
      };
    }
  }

  // Generic request methods
  private async get<T>(url: string, params?: Record<string, any>): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url, { params });
    return response.data;
  }

  private async post<T>(url: string, data?: any, params?: Record<string, any>): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data, { params });
    return response.data;
  }

  private async put<T>(url: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.put(url, data);
    return response.data;
  }

  private async delete<T>(url: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.delete(url);
    return response.data;
  }

  // Graph endpoints
  async createGraph(name: string, description?: string): Promise<Graph> {
    const request: CreateGraphRequest = { name, description };
    return this.post<Graph>('/graphs', request);
  }

  async listGraphs(): Promise<Graph[]> {
    return this.get<Graph[]>('/graphs');
  }

  async getGraph(graphId: number): Promise<Graph> {
    return this.get<Graph>(`/graphs/${graphId}`);
  }

  async deleteGraph(graphId: number): Promise<{ message: string }> {
    return this.delete<{ message: string }>(`/graphs/${graphId}`);
  }

  // Version endpoints
  async createVersion(
    graphId: number,
    parentVersionId: number | null,
    nodes: Node[],
    edges: Edge[]
  ): Promise<GraphVersion> {
    const request: CreateVersionRequest = {
      graph_id: graphId,
      parent_version_id: parentVersionId || undefined,
      nodes,
      edges,
    };
    return this.post<GraphVersion>('/versions', request);
  }

  async getVersion(versionId: number): Promise<GraphVersion> {
    return this.get<GraphVersion>(`/versions/${versionId}`);
  }

  async getLineageTree(graphId: number): Promise<LineageTreeNode> {
    return this.get<LineageTreeNode>(`/versions/graphs/${graphId}/lineage-tree`);
  }

  // diff and impact: version_a_id and version_b_id are FastAPI query params (simple types)
  async diffVersions(versionAId: number, versionBId: number): Promise<StructuralDiff> {
    return this.post<StructuralDiff>('/versions/diff', undefined, {
      version_a_id: versionAId,
      version_b_id: versionBId,
    });
  }

  async getImpactAnalysis(versionAId: number, versionBId: number): Promise<ImpactAnalysis> {
    return this.post<ImpactAnalysis>('/versions/impact', undefined, {
      version_a_id: versionAId,
      version_b_id: versionBId,
    });
  }

  // Experiment endpoints
  async createExperiment(versionId: number, hyperparameters: object): Promise<ExperimentRun> {
    const request: CreateExperimentRequest = {
      version_id: versionId,
      hyperparameters: hyperparameters as any,
    };
    return this.post<ExperimentRun>('/experiments', request);
  }

  async getExperiment(experimentId: number): Promise<ExperimentRun> {
    return this.get<ExperimentRun>(`/experiments/${experimentId}`);
  }

  async listExperiments(versionId: number): Promise<ExperimentRun[]> {
    return this.get<ExperimentRun[]>(`/experiments/versions/${versionId}/experiments`);
  }

  // runSweep: version_id is a query param, param_grid is the JSON body
  async runSweep(versionId: number, paramGrid: object): Promise<number[]> {
    return this.post<number[]>('/experiments/sweep', paramGrid, {
      version_id: versionId,
    });
  }

  // Algorithm endpoints
  async saveAlgorithm(name: string, description: string, code: string): Promise<CustomAlgorithm> {
    const request: SaveAlgorithmRequest = { name, description, code };
    return this.post<CustomAlgorithm>('/algorithms', request);
  }

  async listAlgorithms(): Promise<CustomAlgorithm[]> {
    return this.get<CustomAlgorithm[]>('/algorithms');
  }

  async getAlgorithm(algorithmId: number): Promise<CustomAlgorithm> {
    return this.get<CustomAlgorithm>(`/algorithms/${algorithmId}`);
  }

  async deleteAlgorithm(algorithmId: number): Promise<{ message: string }> {
    return this.delete<{ message: string }>(`/algorithms/${algorithmId}`);
  }

  async updateAlgorithm(algorithmId: number, data: Partial<SaveAlgorithmRequest>): Promise<CustomAlgorithm> {
    return this.put<CustomAlgorithm>(`/algorithms/${algorithmId}`, data);
  }

  // Health check — hits the root URL, not /api/v1
  async healthCheck(): Promise<{ status: string }> {
    const response = await axios.get<{ status: string }>(`${ROOT_URL}/health`, { timeout: 5000 });
    return response.data;
  }

  // Quick connection check — returns true if backend is reachable
  async checkConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }
}

// Create singleton instance
export const apiClient = new ApiClient();
