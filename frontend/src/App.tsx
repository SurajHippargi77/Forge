import React, { useEffect, useState } from 'react';
import { Zap, Plus, Cpu, GitBranch, Beaker, WifiOff } from 'lucide-react';

import { useStore, useGraphs, useActiveGraph, useActiveVersion, useExperiments, useVersions } from './store/useStore';
import { apiClient } from './api/client';
import GraphEditor from './components/GraphEditor';
import LineageTree from './components/LineageTree';
import ExperimentDashboard from './components/ExperimentDashboard';
import AlgorithmManager from './components/AlgorithmManager';

type TabType = 'editor' | 'lineage' | 'experiments';
type ViewType = 'main' | 'algorithms';

const App: React.FC = () => {
  const graphs = useGraphs();
  const activeGraph = useActiveGraph();
  const activeVersion = useActiveVersion();
  const experiments = useExperiments();
  const versions = useVersions();
  const { fetchGraphs, fetchAlgorithms, createGraph, setActiveGraph, setLoading } = useStore();
  
  const [activeTab, setActiveTab] = useState<TabType>('editor');
  const [currentView, setCurrentView] = useState<ViewType>('main');
  const [showNewGraphForm, setShowNewGraphForm] = useState(false);
  const [backendOffline, setBackendOffline] = useState(false);
  const [sweepToast, setSweepToast] = useState<string | null>(null);
  const [newGraphForm, setNewGraphForm] = useState({
    name: '',
    description: '',
  });

  // Check backend connection on load and periodically
  useEffect(() => {
    const checkBackend = async () => {
      const connected = await apiClient.checkConnection();
      setBackendOffline(!connected);
    };
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchGraphs();
    fetchAlgorithms();
  }, [fetchGraphs, fetchAlgorithms]);

  // Auto-select first graph on initial load so judges see content immediately
  useEffect(() => {
    if (graphs.length > 0 && !activeGraph) {
      setActiveGraph(graphs[0]);
    }
  }, [graphs, activeGraph, setActiveGraph]);

  // Auto-dismiss sweep toast
  useEffect(() => {
    if (sweepToast) {
      const timer = setTimeout(() => setSweepToast(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [sweepToast]);

  const handleCreateGraph = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newGraphForm.name.trim()) return;

    try {
      setLoading(true);
      const newGraph = await createGraph(newGraphForm.name, newGraphForm.description);
      setNewGraphForm({ name: '', description: '' });
      setShowNewGraphForm(false);
      // Auto-select the newly created graph and switch to editor
      if (newGraph) {
        await setActiveGraph(newGraph);
        setActiveTab('editor');
      }
    } catch (error) {
      console.error('Failed to create graph:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSweepLaunched = (experimentCount: number) => {
    setSweepToast(`\u{1F680} Sweep launched! ${experimentCount} experiments running...`);
    setActiveTab('experiments');
  };

  const getVersionCount = (graphId: number) => {
    const graph = graphs.find(g => g.id === graphId);
    return graph?.versions?.length || 0;
  };

  const getExperimentCount = () => {
    return experiments.length;
  };

  const tabs = [
    { id: 'editor' as TabType, label: 'Graph Editor', icon: Zap },
    { id: 'lineage' as TabType, label: `Lineage Tree (${versions.length})`, icon: GitBranch },
    { id: 'experiments' as TabType, label: 'Experiments', icon: Beaker },
  ];

  if (currentView === 'algorithms') {
    return <AlgorithmManager onBack={() => setCurrentView('main')} />;
  }

  return (
    <div className="flex flex-col h-screen bg-[#0f0f1a]">
      {/* Backend Offline Banner */}
      {backendOffline && (
        <div className="bg-red-600/90 text-white px-4 py-2 flex items-center justify-center space-x-2 text-sm backdrop-blur-sm">
          <WifiOff size={16} />
          <span>Backend offline — cannot reach http://localhost:8000</span>
        </div>
      )}

      {/* Sweep Toast */}
      {sweepToast && (
        <div className="bg-green-600/90 text-white px-4 py-2 flex items-center justify-center text-sm font-medium backdrop-blur-sm">
          {sweepToast}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
      {/* Left Sidebar */}
      <div className="w-64 bg-gray-800/60 border-r border-gray-700/50 flex flex-col backdrop-blur-sm">
        {/* Logo */}
        <div className="p-4 border-b border-gray-700/50">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center">
              <Zap className="text-white" size={18} />
            </div>
            <div>
              <h1 className="text-white font-bold text-lg tracking-tight">Froge</h1>
              <p className="text-gray-500 text-[10px] uppercase tracking-widest">ML Workflow OS</p>
            </div>
          </div>
        </div>

        {/* Graphs Section */}
        <div className="flex-1 overflow-y-auto relative">
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-gray-400 font-semibold text-[11px] uppercase tracking-wider">Graphs</h2>
              <button
                onClick={() => setShowNewGraphForm(!showNewGraphForm)}
                className="text-gray-500 hover:text-white transition-colors duration-150 p-0.5 rounded hover:bg-gray-700"
                title="New Graph"
              >
                <Plus size={14} />
              </button>
            </div>

            {/* New Graph Form */}
            {showNewGraphForm && (
              <form onSubmit={handleCreateGraph} className="mb-4 p-3 bg-gray-700/50 rounded-lg border border-gray-600/50">
                <input
                  type="text"
                  value={newGraphForm.name}
                  onChange={(e) => setNewGraphForm({ ...newGraphForm, name: e.target.value })}
                  placeholder="Graph name"
                  className="w-full px-2.5 py-1.5 bg-gray-800 border border-gray-600 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 mb-2 transition-colors"
                  autoFocus
                />
                <input
                  type="text"
                  value={newGraphForm.description}
                  onChange={(e) => setNewGraphForm({ ...newGraphForm, description: e.target.value })}
                  placeholder="Description (optional)"
                  className="w-full px-2.5 py-1.5 bg-gray-800 border border-gray-600 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 mb-2 transition-colors"
                />
                <div className="flex space-x-2">
                  <button
                    type="submit"
                    className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-500 transition-colors duration-150 font-medium"
                  >
                    Create
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowNewGraphForm(false)}
                    className="px-3 py-1.5 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-500 transition-colors duration-150"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            )}

            {/* Graphs List */}
            <div className="space-y-1">
              {graphs.length === 0 ? (
                <div className="text-center py-6">
                  <p className="text-gray-500 text-sm">No graphs yet.</p>
                  <p className="text-gray-600 text-xs mt-1">Create your first graph to begin.</p>
                </div>
              ) : (
                graphs.map((graph) => {
                  const vCount = getVersionCount(graph.id);
                  const hasExperiments = (graph as any).versions?.some((v: any) => v.experiments && v.experiments.length > 0);
                  return (
                    <button
                      key={graph.id}
                      onClick={() => setActiveGraph(graph)}
                      className={`
                        w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-150
                        ${activeGraph?.id === graph.id 
                          ? 'bg-blue-600/15 text-white border-l-[3px] border-l-blue-500' 
                          : 'text-gray-300 hover:bg-gray-700/50 border-l-[3px] border-l-transparent'
                        }
                      `}
                    >
                      <div className="flex items-center space-x-2">
                        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${hasExperiments ? 'bg-green-400' : 'bg-gray-600'}`} />
                        <span className="font-medium truncate">{graph.name}</span>
                      </div>
                      <div className="flex items-center mt-1 ml-4">
                        <span className="text-[11px] bg-gray-700/60 text-gray-400 px-2 py-0.5 rounded-full">
                          {vCount} version{vCount !== 1 ? 's' : ''}
                        </span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
          {/* Gradient fade at bottom */}
          <div className="sticky bottom-0 h-8 bg-gradient-to-t from-gray-800/60 to-transparent pointer-events-none" />
        </div>

        {/* Algorithms Button */}
        <div className="p-4 border-t border-gray-700/50">
          <button
            onClick={() => setCurrentView('algorithms')}
            className="w-full flex items-center space-x-2.5 px-3 py-2.5 bg-gray-700/40 text-gray-300 rounded-lg hover:bg-gray-700/70 transition-all duration-150 border border-gray-700/50"
          >
            <Cpu size={16} />
            <span className="text-sm font-medium">Algorithms</span>
          </button>
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col">
        {!activeGraph ? (
          // Welcome Screen
          <div className="flex-1 flex items-center justify-center bg-[#0f0f1a]">
            <div className="text-center">
              <div className="flex items-center justify-center space-x-3 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center">
                  <Zap className="text-white" size={32} />
                </div>
              </div>
              <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">Froge</h1>
              <p className="text-lg text-gray-400 mb-1">Graph-Native ML Workflow OS</p>
              <p className="text-gray-600 text-sm">Create or select a graph to get started</p>
            </div>
          </div>
        ) : (
          <>
            {/* Top Bar */}
            <div className="bg-gray-800/80 border-b border-gray-700/50 px-6 py-3 backdrop-blur-sm relative">
              {/* Animated gradient top accent */}
              <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500 animate-gradient-x" />
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <h2 className="text-lg font-semibold text-white tracking-tight">
                    {activeGraph.name}
                  </h2>
                  {activeVersion && (
                    <span className="bg-gray-700/60 text-gray-300 text-xs px-3 py-1 rounded-full font-medium">
                      v{activeVersion.version_number}
                    </span>
                  )}
                  {activeVersion && experiments.length > 0 && (
                    <span className="bg-blue-500/15 text-blue-400 text-xs px-3 py-1 rounded-full font-medium border border-blue-500/20">
                      {getExperimentCount()} experiment{getExperimentCount() !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Tab Bar */}
            <div className="bg-gray-800/60 border-b border-gray-700/50 backdrop-blur-sm">
              <div className="flex space-x-1 px-6">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`
                        flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-all duration-150
                        ${activeTab === tab.id
                          ? 'text-white border-blue-500'
                          : 'text-gray-500 border-transparent hover:text-gray-300'
                        }
                      `}
                    >
                      <Icon size={15} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden">
              {activeTab === 'editor' && <GraphEditor onSweepLaunched={handleSweepLaunched} />}
              {activeTab === 'lineage' && <LineageTree onVersionSelect={() => setActiveTab('experiments')} />}
              {activeTab === 'experiments' && <ExperimentDashboard />}
            </div>
          </>
        )}
      </div>
      </div>
    </div>
  );
};

export default App;
