import React, { useCallback, useEffect, useState, useRef, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Play, Save, X, Layers } from 'lucide-react';

import CustomNode from './CustomNode';
import { useStore, useActiveGraph, useActiveVersion, useAlgorithms } from '../store/useStore';
import { Node as NodeType, Edge as EdgeType } from '../types';

// Register custom node type
const nodeTypes = {
  custom: CustomNode,
};

// Node type configurations
const nodeTypeConfigs: Record<string, { type: string; label: string; color: string; defaultParams?: Record<string, any> }> = {
  input: { type: 'input', label: 'Input', color: 'bg-blue-500', defaultParams: {} },
  dense: { type: 'dense', label: 'Dense', color: 'bg-purple-500', defaultParams: { units: 128 } },
  conv2d: { type: 'conv2d', label: 'Conv2D', color: 'bg-orange-500', defaultParams: { filters: 32, kernel_size: 3 } },
  relu: { type: 'relu', label: 'ReLU', color: 'bg-green-500', defaultParams: {} },
  batchnorm: { type: 'batchnorm', label: 'BatchNorm', color: 'bg-yellow-500', defaultParams: {} },
  dropout: { type: 'dropout', label: 'Dropout', color: 'bg-red-500', defaultParams: { rate: 0.5 } },
  output: { type: 'output', label: 'Output', color: 'bg-blue-500', defaultParams: {} },
};

interface GraphEditorProps {
  onSweepLaunched?: (experimentCount: number) => void;
}

const GraphEditor: React.FC<GraphEditorProps> = ({ onSweepLaunched }) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  const [showSweepPanel, setShowSweepPanel] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [sweepConfig, setSweepConfig] = useState({
    learning_rate: '0.01, 0.001',
    batch_size: '32, 64',
    epochs: '10, 20',
    optimizer: 'adam, sgd',
  });

  const activeGraph = useActiveGraph();
  const activeVersion = useActiveVersion();
  const algorithms = useAlgorithms();
  const { saveVersion, runSweep, setLoading, error } = useStore();

  // Initialize nodes and edges from active version
  const initialNodes: Node[] = activeVersion ? activeVersion.nodes.map(node => ({
    id: node.id,
    type: 'custom',
    position: node.position,
    data: {
      type: node.type,
      label: node.label,
      params: node.params,
      onDelete: handleDeleteNode,
      onParamsChange: handleParamsChange,
    },
  })) : [];

  const initialEdges: Edge[] = activeVersion ? activeVersion.edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: 'smoothstep',
  })) : [];

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when active version changes
  useEffect(() => {
    if (activeVersion) {
      const newNodes = activeVersion.nodes.map(node => ({
        id: node.id,
        type: 'custom',
        position: node.position,
        data: {
          type: node.type,
          label: node.label,
          params: node.params,
          onDelete: handleDeleteNode,
          onParamsChange: handleParamsChange,
        },
      }));
      setNodes(newNodes);

      const newEdges = activeVersion.edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
      }));
      setEdges(newEdges);
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [activeVersion, setNodes, setEdges]);

  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) => addEdge({ ...params, type: 'smoothstep' }, eds));
    setHasUnsavedChanges(true);
  }, [setEdges]);

  const onInit = useCallback((rfi: any) => {
    setReactFlowInstance(rfi);
  }, []);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const type = event.dataTransfer.getData('application/reactflow');
      const algorithmName = event.dataTransfer.getData('algorithm/name');

      if (!type && !algorithmName) return;

      // Calculate position using screenToFlowPosition (replaces deprecated project())
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      position.x -= 75;
      position.y -= 40;

      let nodeType: string;
      let label: string;
      let params: Record<string, any>;

      if (algorithmName) {
        // Custom algorithm node
        nodeType = 'custom';
        label = algorithmName;
        params = {};
      } else {
        // Standard node type
        const config = nodeTypeConfigs[type];
        if (!config) return;
        nodeType = config.type;
        label = config.label;
        params = config.defaultParams || {};
      }

      const newNode: Node = {
        id: `${nodeType}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          type: nodeType,
          label,
          params,
          onDelete: handleDeleteNode,
          onParamsChange: handleParamsChange,
        },
      };

      setNodes((nds) => nds.concat(newNode));
      setHasUnsavedChanges(true);
    },
    [reactFlowInstance, setNodes]
  );

  function handleDeleteNode(nodeId: string) {
    setNodes((nds) => nds.filter((node) => node.id !== nodeId));
    setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
    setHasUnsavedChanges(true);
  }

  function handleParamsChange(nodeId: string, params: Record<string, any>) {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, params } }
          : node
      )
    );
    setHasUnsavedChanges(true);
  }

  const handleSaveVersion = async () => {
    if (!activeGraph) return;

    try {
      setLoading(true);
      
      // Convert ReactFlow nodes to our Node type
      const graphNodes: NodeType[] = nodes.map(node => ({
        id: node.id,
        type: node.data.type,
        label: node.data.label,
        params: node.data.params,
        position: node.position,
      }));

      // Convert ReactFlow edges to our Edge type
      const graphEdges: EdgeType[] = edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      }));

      await saveVersion(graphNodes, graphEdges);
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error('Failed to save version:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLaunchSweep = async () => {
    try {
      setLoading(true);
      
      // Parse sweep configuration
      const paramGrid = {
        learning_rate: sweepConfig.learning_rate.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n)),
        batch_size: sweepConfig.batch_size.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
        epochs: sweepConfig.epochs.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)),
        optimizer: sweepConfig.optimizer.split(',').map(s => s.trim()).filter(s => s),
      };

      const experimentIds = await runSweep(paramGrid);
      setShowSweepPanel(false);
      onSweepLaunched?.(experimentIds.length);
    } catch (error) {
      console.error('Failed to launch sweep:', error);
    } finally {
      setLoading(false);
    }
  };

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const onAlgorithmDragStart = (event: React.DragEvent, algorithmName: string) => {
    event.dataTransfer.setData('algorithm/name', algorithmName);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="flex flex-col h-full bg-[#0f0f1a]">
      {/* Top Toolbar */}
      <div className="bg-gray-800/80 border-b border-gray-700/50 px-4 py-3 flex items-center justify-between backdrop-blur-sm">
        <div className="flex items-center space-x-4">
          <h1 className="text-white font-semibold">
            {activeGraph ? activeGraph.name : 'No Graph Selected'}
          </h1>
          {activeVersion && (
            <span className="text-gray-500 text-sm">
              Version {activeVersion.version_number}
            </span>
          )}
          {hasUnsavedChanges && (
            <span className="text-orange-400 text-xs bg-orange-400/10 border border-orange-400/30 px-2.5 py-0.5 rounded-full font-medium">
              Unsaved changes
            </span>
          )}
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={handleSaveVersion}
            disabled={!activeGraph}
            className={`flex items-center space-x-2 px-4 h-9 bg-blue-600 text-white rounded-md hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition-all duration-150 text-sm font-medium ${hasUnsavedChanges ? 'animate-pulse shadow-lg shadow-blue-500/25' : ''}`}
          >
            <Save size={14} />
            <span>Save Version</span>
          </button>
          
          <button
            onClick={() => setShowSweepPanel(true)}
            disabled={!activeVersion}
            className="flex items-center space-x-2 px-4 h-9 bg-green-600 text-white rounded-md hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition-all duration-150 text-sm font-medium"
          >
            <Play size={14} />
            <span>Run Sweep</span>
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-48 bg-gray-800/60 border-r border-gray-700/50 overflow-y-auto backdrop-blur-sm">
          <div className="p-4">
            <h3 className="text-gray-400 font-semibold mb-3 text-[11px] uppercase tracking-wider">Node Palette</h3>
            <div className="space-y-1.5">
              {Object.entries(nodeTypeConfigs).map(([key, config]) => (
                <div
                  key={key}
                  draggable
                  onDragStart={(event) => onDragStart(event, key)}
                  className="flex items-center space-x-2.5 p-2 bg-gray-700/50 rounded-md cursor-move hover:bg-gray-600/70 hover:brightness-110 transition-all duration-150 border border-transparent hover:border-gray-600"
                >
                  <div className={`w-2.5 h-2.5 rounded-full ${config.color}`}></div>
                  <span className="text-gray-200 text-sm">{config.label}</span>
                </div>
              ))}
            </div>
          </div>

          {algorithms.length > 0 && (
            <div className="p-4 border-t border-gray-700/50">
              <h3 className="text-gray-400 font-semibold mb-3 text-[11px] uppercase tracking-wider">Custom Algorithms</h3>
              <div className="space-y-1.5">
                {algorithms.map((algorithm) => (
                  <div
                    key={algorithm.id}
                    draggable
                    onDragStart={(event) => onAlgorithmDragStart(event, algorithm.name)}
                    className="flex items-center space-x-2.5 p-2 bg-gray-700/50 rounded-md cursor-move hover:bg-gray-600/70 hover:brightness-110 transition-all duration-150 border border-transparent hover:border-gray-600"
                  >
                    <div className="w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
                    <span className="text-gray-200 text-sm truncate">{algorithm.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Main Canvas */}
        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={onInit}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
            className="bg-[#0f0f1a]"
          >
            <Background color="#1e293b" gap={20} size={1} />
            <Controls className="bg-gray-800 border border-gray-700" />
            <MiniMap
              className="bg-gray-800"
              nodeColor={() => '#4b5563'}
            />
          </ReactFlow>

          {/* Empty canvas instructions */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className="text-center">
                <Layers className="mx-auto text-gray-600 mb-3" size={48} />
                <p className="text-gray-500 text-lg font-medium mb-1">Empty Canvas</p>
                <p className="text-gray-600 text-sm">Drag nodes from the palette to start building your model</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sweep Configuration Panel */}
      {showSweepPanel && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="rounded-xl p-6 w-96 max-w-full mx-4 border border-[#334155]" style={{ backgroundColor: '#1e2a3a' }}>
            <div className="flex justify-between items-center mb-5">
              <h2 className="text-lg font-bold text-white">Configure Sweep</h2>
              <button
                onClick={() => setShowSweepPanel(false)}
                className="text-gray-400 hover:text-white transition-colors duration-150 p-1 rounded-md hover:bg-gray-700"
                aria-label="Close sweep configuration panel"
                title="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1 uppercase tracking-wider">Learning Rate</label>
                <input type="text" value={sweepConfig.learning_rate} onChange={(e) => setSweepConfig({ ...sweepConfig, learning_rate: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors" placeholder="0.01, 0.001" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1 uppercase tracking-wider">Batch Size</label>
                <input type="text" value={sweepConfig.batch_size} onChange={(e) => setSweepConfig({ ...sweepConfig, batch_size: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors" placeholder="32, 64" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1 uppercase tracking-wider">Epochs</label>
                <input type="text" value={sweepConfig.epochs} onChange={(e) => setSweepConfig({ ...sweepConfig, epochs: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors" placeholder="10, 20" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1 uppercase tracking-wider">Optimizer</label>
                <input type="text" value={sweepConfig.optimizer} onChange={(e) => setSweepConfig({ ...sweepConfig, optimizer: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors" placeholder="adam, sgd" />
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button onClick={() => setShowSweepPanel(false)}
                className="px-4 h-9 text-gray-300 bg-gray-700 rounded-md hover:bg-gray-600 transition-colors duration-150 text-sm font-medium">
                Cancel
              </button>
              <button onClick={handleLaunchSweep}
                className="px-4 h-9 bg-green-600 text-white rounded-md hover:bg-green-500 transition-colors duration-150 text-sm font-medium">
                Launch Sweep
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-600 text-white px-4 py-3 rounded-lg shadow-lg max-w-md">
          <div className="flex items-center justify-between">
            <span className="text-sm">{error}</span>
            <button
              onClick={() => useStore.setState({ error: null })}
              className="ml-3 text-white/80 hover:text-white"
              aria-label="Dismiss error message"
              title="Dismiss"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Wrap with ReactFlowProvider
const GraphEditorWithProvider: React.FC<GraphEditorProps> = ({ onSweepLaunched }) => (
  <ReactFlowProvider>
    <GraphEditor onSweepLaunched={onSweepLaunched} />
  </ReactFlowProvider>
);

export default GraphEditorWithProvider;
