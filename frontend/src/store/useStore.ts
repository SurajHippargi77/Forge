import { create } from 'zustand';
import { apiClient } from '../api/client';
import {
  StoreState,
  Graph,
  GraphVersion,
  ExperimentRun,
  CustomAlgorithm,
  StructuralDiff,
  ImpactAnalysis,
  Node,
  Edge,
  PollingOptions,
  ExperimentStatus,
} from '../types';

interface StoreActions {
  // Graph actions
  fetchGraphs: () => Promise<void>;
  createGraph: (name: string, description?: string) => Promise<Graph | null>;
  deleteGraph: (graphId: number) => Promise<void>;
  setActiveGraph: (graph: Graph | null) => Promise<void>;
  
  // Version actions
  setActiveVersion: (version: GraphVersion | null) => Promise<void>;
  fetchAndSetActiveVersion: (versionId: number) => Promise<void>;
  saveVersion: (nodes: Node[], edges: Edge[]) => Promise<void>;
  fetchVersions: (graphId: number) => Promise<void>;
  
  // Experiment actions
  runSweep: (paramGrid: object) => Promise<number[]>;
  pollExperiments: (options?: PollingOptions) => Promise<void>;
  fetchExperiments: (versionId: number) => Promise<void>;
  
  // Analysis actions
  compareTwoVersions: (versionAId: number, versionBId: number) => Promise<void>;
  
  // Algorithm actions
  saveAlgorithm: (name: string, description: string, code: string) => Promise<void>;
  fetchAlgorithms: () => Promise<void>;
  
  // UI actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

type Store = StoreState & StoreActions;

export const useStore = create<Store>((set, get) => ({
  // Initial state
  graphs: [],
  activeGraph: null,
  activeVersion: null,
  versions: [],
  experiments: [],
  lineageTree: null,
  diffResult: null,
  impactResult: null,
  algorithms: [],
  isLoading: false,
  error: null,

  // Graph actions
  fetchGraphs: async () => {
    try {
      set({ isLoading: true, error: null });
      const graphs = await apiClient.listGraphs();
      set({ graphs, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch graphs',
        isLoading: false 
      });
    }
  },

  createGraph: async (name: string, description?: string) => {
    try {
      set({ isLoading: true, error: null });
      const newGraph = await apiClient.createGraph(name, description);
      // Refresh the graphs list
      const graphs = await apiClient.listGraphs();
      set({ graphs, isLoading: false });
      return newGraph;
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to create graph',
        isLoading: false 
      });
      return null;
    }
  },

  deleteGraph: async (graphId: number) => {
    try {
      set({ isLoading: true, error: null });
      await apiClient.deleteGraph(graphId);
      const currentActive = get().activeGraph;
      const graphs = await apiClient.listGraphs();
      const updates: Partial<Store> = { graphs, isLoading: false };
      if (currentActive?.id === graphId) {
        updates.activeGraph = null;
        updates.activeVersion = null;
        updates.versions = [];
        updates.experiments = [];
      }
      set(updates);
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to delete graph',
        isLoading: false,
      });
    }
  },

  setActiveGraph: async (graph: Graph | null) => {
    try {
      set({ isLoading: true, error: null, activeGraph: graph });
      
      if (!graph) {
        set({ 
          activeVersion: null, 
          versions: [], 
          experiments: [], 
          lineageTree: null,
          isLoading: false 
        });
        return;
      }

      // Fetch lineage tree and versions in parallel
      const [lineageTree, versions] = await Promise.all([
        apiClient.getLineageTree(graph.id),
        // Get versions from the graph object if available, otherwise fetch
        graph.versions ? Promise.resolve(graph.versions) : apiClient.getGraph(graph.id).then(g => g.versions || [])
      ]);

      // Auto-select the latest version so saves are properly parented
      const latestVersion = versions.length > 0 
        ? versions.reduce((latest, v) => v.version_number > latest.version_number ? v : latest, versions[0])
        : null;

      set({ 
        lineageTree, 
        versions, 
        activeVersion: latestVersion, 
        experiments: [],
        isLoading: false 
      });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to set active graph',
        isLoading: false 
      });
    }
  },

  // Version actions
  setActiveVersion: async (version: GraphVersion | null) => {
    try {
      set({ isLoading: true, error: null, activeVersion: version });
      
      if (!version) {
        set({ experiments: [], isLoading: false });
        return;
      }

      // Fetch experiments for this version
      const experiments = await apiClient.listExperiments(version.id);
      set({ experiments, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to set active version',
        isLoading: false 
      });
    }
  },

  fetchAndSetActiveVersion: async (versionId: number) => {
    try {
      set({ isLoading: true, error: null });
      const version = await apiClient.getVersion(versionId);
      set({ activeVersion: version });
      const experiments = await apiClient.listExperiments(version.id);
      set({ experiments, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch version',
        isLoading: false,
      });
    }
  },

  saveVersion: async (nodes: Node[], edges: Edge[]) => {
    try {
      const { activeVersion, activeGraph } = get();
      
      if (!activeGraph) {
        throw new Error('No active graph selected');
      }

      set({ isLoading: true, error: null });
      
      const parentVersionId = activeVersion?.id || null;
      const newVersion = await apiClient.createVersion(
        activeGraph.id,
        parentVersionId,
        nodes,
        edges
      );

      // Update versions list and set as active
      const { versions } = get();
      const updatedVersions = [...versions, newVersion];
      set({ 
        versions: updatedVersions, 
        activeVersion: newVersion,
        isLoading: false 
      });

      // Refresh graphs list (sidebar version count) and lineage tree
      get().fetchGraphs();
      if (activeGraph) {
        apiClient.getLineageTree(activeGraph.id).then(lineageTree => {
          set({ lineageTree });
        });
      }
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to save version',
        isLoading: false 
      });
    }
  },

  fetchVersions: async (graphId: number) => {
    try {
      set({ isLoading: true, error: null });
      const graph = await apiClient.getGraph(graphId);
      set({ versions: graph.versions || [], isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch versions',
        isLoading: false 
      });
    }
  },

  // Experiment actions
  runSweep: async (paramGrid: object) => {
    try {
      const { activeVersion } = get();
      
      if (!activeVersion) {
        throw new Error('No active version selected');
      }

      set({ isLoading: true, error: null });
      
      const experimentIds = await apiClient.runSweep(activeVersion.id, paramGrid);
      
      // Start polling for results
      get().pollExperiments({
        interval: 3000,
        maxAttempts: 100,
      });

      set({ isLoading: false });
      return experimentIds;
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to run sweep',
        isLoading: false 
      });
      throw error;
    }
  },

  pollExperiments: async (options: PollingOptions = {}) => {
    const {
      interval = 3000,
      maxAttempts = 100,
      onComplete,
      onError,
    } = options;

    const { activeVersion } = get();
    
    if (!activeVersion) {
      return;
    }

    let attempts = 0;
    
    const poll = async () => {
      try {
        attempts++;
        
        // Fetch latest experiments
        const experiments = await apiClient.listExperiments(activeVersion.id);
        set({ experiments });

        // Check if all experiments are completed or failed
        const allFinished = experiments.every(
          exp => exp.status === 'completed' || exp.status === 'failed'
        );

        if (allFinished || attempts >= maxAttempts) {
          if (onComplete) onComplete();
          return;
        }

        // Continue polling
        setTimeout(poll, interval);
      } catch (error) {
        console.error('Polling error:', error);
        if (onError) {
          onError(error instanceof Error ? error.message : 'Polling failed');
        }
      }
    };

    // Start polling
    setTimeout(poll, interval);
  },

  fetchExperiments: async (versionId: number) => {
    try {
      set({ isLoading: true, error: null });
      const experiments = await apiClient.listExperiments(versionId);
      set({ experiments, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch experiments',
        isLoading: false 
      });
    }
  },

  // Analysis actions
  compareTwoVersions: async (versionAId: number, versionBId: number) => {
    try {
      set({ isLoading: true, error: null });
      
      const [diffResult, impactResult] = await Promise.all([
        apiClient.diffVersions(versionAId, versionBId),
        apiClient.getImpactAnalysis(versionAId, versionBId)
      ]);

      set({ diffResult, impactResult, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to compare versions',
        isLoading: false 
      });
    }
  },

  // Algorithm actions
  saveAlgorithm: async (name: string, description: string, code: string) => {
    try {
      set({ isLoading: true, error: null });
      await apiClient.saveAlgorithm(name, description, code);
      
      // Refresh algorithms list
      const algorithms = await apiClient.listAlgorithms();
      set({ algorithms, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to save algorithm',
        isLoading: false 
      });
    }
  },

  fetchAlgorithms: async () => {
    try {
      set({ isLoading: true, error: null });
      const algorithms = await apiClient.listAlgorithms();
      set({ algorithms, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch algorithms',
        isLoading: false 
      });
    }
  },

  // UI actions
  setLoading: (loading: boolean) => set({ isLoading: loading }),
  setError: (error: string | null) => set({ error }),
  clearError: () => set({ error: null }),
}));

// Selectors for commonly used computed values
export const useGraphs = () => useStore(state => state.graphs);
export const useActiveGraph = () => useStore(state => state.activeGraph);
export const useActiveVersion = () => useStore(state => state.activeVersion);
export const useVersions = () => useStore(state => state.versions);
export const useExperiments = () => useStore(state => state.experiments);
export const useLineageTree = () => useStore(state => state.lineageTree);
export const useDiffResult = () => useStore(state => state.diffResult);
export const useImpactResult = () => useStore(state => state.impactResult);
export const useAlgorithms = () => useStore(state => state.algorithms);
export const useLoading = () => useStore(state => state.isLoading);
export const useError = () => useStore(state => state.error);

// Computed selectors
export const useCompletedExperiments = () => {
  const experiments = useExperiments();
  return experiments.filter(exp => exp.status === 'completed');
};

export const useRunningExperiments = () => {
  const experiments = useExperiments();
  return experiments.filter(exp => exp.status === 'running');
};

export const useBestExperiment = () => {
  const completedExperiments = useCompletedExperiments();
  return completedExperiments.reduce((best, current) => {
    if (!best || !current.metrics?.val_loss) return current;
    if (!best.metrics?.val_loss) return current;
    return current.metrics.val_loss < best.metrics.val_loss ? current : best;
  }, null as ExperimentRun | null);
};
