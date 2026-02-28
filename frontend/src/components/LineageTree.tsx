import React, { useCallback, useEffect, useState, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  ReactFlowProvider,
  useReactFlow,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { GitCompare, Maximize2, X } from 'lucide-react';

import { useStore, useLineageTree, useActiveVersion, useVersions } from '../store/useStore';
import { LineageTreeNode } from '../types';
import DiffViewer from './DiffViewer';

interface VersionNodeData {
  versionId: number;
  versionNumber: number;
  nodeCount: number;
  createdAt: string;
  isActive: boolean;
  onVersionClick: (versionId: number) => void;
  onCompareClick: (versionId: number) => void;
  compareMode: boolean;
  isSelected: boolean;
}

const VersionNode: React.FC<{ data: VersionNodeData }> = ({ data }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div
      className={`
        rounded-xl shadow-lg transition-all duration-200 cursor-pointer
        hover:shadow-xl hover:scale-[1.03] min-w-[140px] text-center relative
        ${data.isSelected ? 'ring-2 ring-purple-400 ring-offset-2 ring-offset-gray-900' : ''}
      `}
      style={{
        backgroundColor: '#1e2a3a',
        border: data.isActive ? '2px solid #3b82f6' : '1px solid #334155',
        boxShadow: data.isActive ? '0 0 16px rgba(59,130,246,0.25)' : undefined,
      }}
      onClick={() => data.onVersionClick(data.versionId)}
    >
      <div className="p-3">
        <div className="font-bold text-lg text-white">
          V{data.versionNumber}
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {formatDate(data.createdAt)}
        </div>
        {data.nodeCount > 0 && (
          <div className="mt-1.5 text-[10px] text-gray-500 font-medium">
            {data.nodeCount} node{data.nodeCount !== 1 ? 's' : ''}
          </div>
        )}
        {data.isActive && (
          <div className="mt-2 inline-block px-2.5 py-0.5 bg-blue-500/90 text-white text-[10px] font-semibold rounded-full tracking-wide uppercase">
            Active
          </div>
        )}
      </div>
      
      {/* Compare Button */}
      <div className="absolute -top-2 -right-2 opacity-0 hover:opacity-100 transition-all duration-150">
        <button
          onClick={(e) => {
            e.stopPropagation();
            data.onCompareClick(data.versionId);
          }}
          className="bg-purple-500 text-white p-1 rounded-full hover:bg-purple-600 shadow-md"
          title="Compare this version"
        >
          <GitCompare size={12} />
        </button>
      </div>
    </div>
  );
};

// Register custom node type
const nodeTypes = {
  version: VersionNode,
};

// Layout algorithm for tree
const calculateTreeLayout = (tree: LineageTreeNode | null): { nodes: Node[], edges: Edge[] } => {
  if (!tree || !tree.id) {
    return { nodes: [], edges: [] };
  }

  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const levelWidths: Record<number, number> = {};

  // First pass: calculate width of each level
  const calculateLevelWidth = (node: LineageTreeNode, level: number = 0): void => {
    levelWidths[level] = (levelWidths[level] || 0) + 1;
    node.children.forEach(child => calculateLevelWidth(child, level + 1));
  };

  // Second pass: position nodes
  const positionNodes = (
    node: LineageTreeNode,
    level: number = 0,
    indexInLevel: number = 0,
    parentX: number = 0
  ): { x: number, y: number } => {
    const levelWidth = levelWidths[level] || 1;
    const horizontalSpacing = 200;
    const verticalSpacing = 120;
    
    // Center the level and position this node
    const totalLevelWidth = (levelWidth - 1) * horizontalSpacing;
    const startX = parentX - totalLevelWidth / 2;
    const x = startX + indexInLevel * horizontalSpacing;
    const y = level * verticalSpacing;

    // Create node
    nodes.push({
      id: node.id.toString(),
      type: 'version',
      position: { x, y },
      data: {
        versionId: node.id!,
        versionNumber: node.version_number!,
        nodeCount: 0, // Will be updated from store
        createdAt: new Date().toISOString(),
        isActive: false,
        onVersionClick: () => {},
        onCompareClick: () => {},
        compareMode: false,
        isSelected: false,
      },
    });

    // Create edges to children
    node.children.forEach((child, childIndex) => {
      if (child.id) {
        edges.push({
          id: `${node.id}-${child.id}`,
          source: node.id.toString(),
          target: child.id.toString(),
          type: 'smoothstep',
          animated: true,
          style: { strokeDasharray: '6,4', stroke: '#3b82f6', strokeWidth: 2 },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#3b82f6',
          },
        });

        // Recursively position children
        positionNodes(child, level + 1, childIndex, x);
      }
    });

    return { x, y };
  };

  calculateLevelWidth(tree);
  positionNodes(tree);

  return { nodes, edges };
};

interface LineageTreeProps {
  onVersionSelect?: () => void;
}

const LineageTreeContent: React.FC<LineageTreeProps> = ({ onVersionSelect }) => {
  const { fitView } = useReactFlow();
  const lineageTree = useLineageTree();
  const activeVersion = useActiveVersion();
  const versions = useVersions();
  const { fetchAndSetActiveVersion, compareTwoVersions } = useStore();
  
  const [compareMode, setCompareMode] = useState(false);
  const [selectedVersions, setSelectedVersions] = useState<number[]>([]);
  const [showDiffViewer, setShowDiffViewer] = useState(false);
  const [diffVersions, setDiffVersions] = useState<{ a: number, b: number } | null>(null);

  // Calculate layout
  const { nodes: layoutNodes, edges: layoutEdges } = useMemo(() => {
    return calculateTreeLayout(lineageTree);
  }, [lineageTree]);

  // Update nodes with store data and handlers
  // Build a map of versionId -> nodeCount from the versions array
  const versionNodeCounts = useMemo(() => {
    const map: Record<number, number> = {};
    versions.forEach(v => { map[v.id] = v.nodes?.length || 0; });
    return map;
  }, [versions]);

  const nodes = useMemo(() => {
    return layoutNodes.map(node => ({
      ...node,
      data: {
        ...node.data,
        nodeCount: versionNodeCounts[node.data.versionId] || 0,
        isActive: activeVersion?.id === node.data.versionId,
        isSelected: selectedVersions.includes(node.data.versionId),
        compareMode,
        onVersionClick: (versionId: number) => {
          if (compareMode) {
            handleVersionSelect(versionId);
          } else {
            // Fetch the full version from the API and set as active
            fetchAndSetActiveVersion(versionId).then(() => {
              onVersionSelect?.();
            });
          }
        },
        onCompareClick: (versionId: number) => {
          handleVersionSelect(versionId);
        },
      },
    }));
  }, [layoutNodes, activeVersion, selectedVersions, compareMode, fetchAndSetActiveVersion, onVersionSelect]);

  const [nodesState, setNodes, onNodesChange] = useNodesState(nodes);
  const [edgesState, setEdges, onEdgesChange] = useEdgesState(layoutEdges);

  // Update nodes when dependencies change
  useEffect(() => {
    setNodes(nodes);
  }, [nodes, setNodes]);

  useEffect(() => {
    setEdges(layoutEdges);
  }, [layoutEdges, setEdges]);

  const handleVersionSelect = (versionId: number) => {
    if (!compareMode) return;

    const newSelection = [...selectedVersions];
    const existingIndex = newSelection.indexOf(versionId);

    if (existingIndex >= 0) {
      // Remove if already selected
      newSelection.splice(existingIndex, 1);
    } else {
      // Add to selection
      newSelection.push(versionId);
    }

    setSelectedVersions(newSelection);

    // If we have 2 versions selected, open diff viewer
    if (newSelection.length === 2) {
      setDiffVersions({ a: newSelection[0], b: newSelection[1] });
      setShowDiffViewer(true);
      setCompareMode(false);
      setSelectedVersions([]);
    }
  };

  const handleFitView = () => {
    fitView({ padding: 0.2, duration: 800 });
  };

  const getVersionCount = (tree: LineageTreeNode | null): number => {
    if (!tree) return 0;
    let count = 1;
    tree.children.forEach(child => {
      count += getVersionCount(child);
    });
    return count;
  };

  const versionCount = getVersionCount(lineageTree);

  if (!lineageTree || !lineageTree.id) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">No versions yet</div>
          <div className="text-gray-500 text-sm">Build a graph and save a version</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="font-semibold text-white">
            {versionCount} Version{versionCount !== 1 ? 's' : ''}
          </h2>
          {compareMode && (
            <div className="text-sm text-purple-300 bg-purple-900/50 px-3 py-1 rounded-full">
              Click two versions to compare
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          {!compareMode ? (
            <button
              onClick={() => setCompareMode(true)}
              className="flex items-center space-x-2 px-3 py-1.5 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 transition-colors"
            >
              <GitCompare size={14} />
              <span>Compare Versions</span>
            </button>
          ) : (
            <button
              onClick={() => {
                setCompareMode(false);
                setSelectedVersions([]);
              }}
              className="flex items-center space-x-2 px-3 py-1.5 bg-gray-600 text-white text-sm rounded hover:bg-gray-700 transition-colors"
            >
              <X size={14} />
              <span>Cancel Compare</span>
            </button>
          )}
          
          <button
            onClick={handleFitView}
            className="p-1.5 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
            title="Fit View"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>

      {/* Tree Visualization */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodesState}
          edges={edgesState}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={true}
          className="bg-gray-900"
        >
          <Background color="#374151" gap={16} />
          <Controls className="bg-gray-800 border border-gray-700" />
        </ReactFlow>
      </div>

      {/* DiffViewer Modal */}
      {showDiffViewer && diffVersions && (
        <DiffViewer
          versionAId={diffVersions.a}
          versionBId={diffVersions.b}
          onClose={() => {
            setShowDiffViewer(false);
            setDiffVersions(null);
            setSelectedVersions([]);
          }}
        />
      )}
    </div>
  );
};

// Wrap with ReactFlowProvider
const LineageTree: React.FC<LineageTreeProps> = ({ onVersionSelect }) => (
  <ReactFlowProvider>
    <LineageTreeContent onVersionSelect={onVersionSelect} />
  </ReactFlowProvider>
);

export default LineageTree;
