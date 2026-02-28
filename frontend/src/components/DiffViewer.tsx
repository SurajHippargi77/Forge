import React, { useEffect } from 'react';
import { X, TrendingUp, TrendingDown, Minus, AlertTriangle, Rocket } from 'lucide-react';
import { useStore, useDiffResult, useImpactResult } from '../store/useStore';

interface DiffViewerProps {
  versionAId: number;
  versionBId: number;
  onClose: () => void;
}

const DiffViewer: React.FC<DiffViewerProps> = ({ versionAId, versionBId, onClose }) => {
  const { compareTwoVersions, isLoading, versions } = useStore();
  const diffResult = useDiffResult();
  const impactResult = useImpactResult();

  const versionANumber = versions.find(v => v.id === versionAId)?.version_number ?? versionAId;
  const versionBNumber = versions.find(v => v.id === versionBId)?.version_number ?? versionBId;

  useEffect(() => {
    compareTwoVersions(versionAId, versionBId);
  }, [versionAId, versionBId, compareTwoVersions]);

  const getImpactBanner = () => {
    if (!impactResult) return null;
    const { impact_level, impact_score, metric_delta } = impactResult;
    const configs: Record<string, { bg: string; text: string; icon: any; label: string }> = {
      high_positive: { bg: 'bg-green-900/60 border border-green-700/50', text: 'text-green-300', icon: Rocket, label: '🚀 High Positive Impact' },
      low_positive:  { bg: 'bg-green-900/40 border border-green-800/50', text: 'text-green-400', icon: TrendingUp, label: '✅ Low Positive Impact' },
      neutral:       { bg: 'bg-gray-800/60 border border-gray-600/50',   text: 'text-gray-300',  icon: Minus, label: '➖ Neutral' },
      low_negative:  { bg: 'bg-orange-900/40 border border-orange-700/50', text: 'text-orange-300', icon: AlertTriangle, label: '⚠️ Low Negative Impact' },
      high_negative: { bg: 'bg-red-900/40 border border-red-700/50',     text: 'text-red-300',   icon: TrendingDown, label: '🔴 High Negative Impact' },
    };
    const config = configs[impact_level];
    const Icon = config.icon;
    return (
      <div className={`${config.bg} ${config.text} p-4 rounded-lg`}>
        <div className="flex items-center space-x-3">
          <Icon size={24} />
          <div>
            <div className="font-semibold text-lg">{config.label}</div>
            {metric_delta && (
              <div className="text-sm mt-1 text-gray-400">
                Val Loss: {metric_delta.val_loss_delta?.toFixed(4) || 'N/A'} |{' '}
                Delta: {metric_delta.val_loss_delta ? (
                  <span className={metric_delta.val_loss_delta < 0 ? 'text-green-400' : 'text-red-400'}>
                    {metric_delta.val_loss_delta > 0 ? '+' : ''}{metric_delta.val_loss_delta.toFixed(4)}
                  </span>
                ) : 'N/A'} |{' '}
                Impact Score: {impact_score?.toFixed(1) || 'N/A'}%
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderChangesList = (title: string, items: any[], type: 'added' | 'removed' | 'modified', icon: string, emptyText: string = 'No changes') => {
    const chipStyles = {
      added:    'bg-[#14532d] text-green-300 border border-green-800/40',
      removed:  'bg-[#7f1d1d] text-red-300 border border-red-800/40',
      modified: 'bg-yellow-950 text-yellow-300 border border-yellow-800/40',
    };
    const countBg = { added: 'bg-green-600', removed: 'bg-red-600', modified: 'bg-yellow-600' };

    if (items.length === 0) {
      return (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-[#334155]">
          <h4 className="font-semibold text-white mb-2">{title}</h4>
          <div className="text-[#6b7280] text-sm italic">{emptyText}</div>
        </div>
      );
    }
    return (
      <div className="bg-gray-800/50 rounded-lg p-4 border border-[#334155]">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-white">{title}</h4>
          <span className={`${countBg[type]} text-white text-xs px-2.5 py-0.5 rounded-full font-medium`}>{items.length}</span>
        </div>
        <div className="space-y-2">
          {items.map((item, index) => (
            <div key={index} className={`${chipStyles[type]} px-3 py-2 rounded-md text-sm flex items-center space-x-2`}>
              <span className="font-bold opacity-60">{icon}</span>
              <span className="font-medium">{item.label || item.node_id} {item.type && <span className="opacity-60">({item.type})</span>}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (isLoading || !diffResult) {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="rounded-xl p-8 w-96 max-w-full mx-4 text-center border border-[#334155]" style={{ backgroundColor: '#1e2a3a' }}>
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <div className="text-gray-300">Analyzing structural changes...</div>
        </div>
      </div>
    );
  }

  const { added_nodes, removed_nodes, modified_nodes, added_edges, removed_edges } = diffResult;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden border border-[#334155]" style={{ backgroundColor: '#1e2a3a' }}>
        {/* Header */}
        <div className="border-b border-[#334155] px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">
            Structural Diff: V{versionANumber} → V{versionBNumber}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors duration-150 p-1.5 rounded-md hover:bg-gray-700" title="Close">
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="p-6 space-y-5">
            {getImpactBanner()}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {renderChangesList('Added Nodes', added_nodes, 'added', '+')}
              {renderChangesList('Removed Nodes', removed_nodes, 'removed', '−')}
            </div>
            {renderChangesList('Modified Nodes', modified_nodes, 'modified', '~', 'No modified nodes')}

            {/* Edge Changes */}
            <div className="bg-gray-800/50 rounded-lg p-4 border border-[#334155]">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-white">Edge Changes</h4>
                <span className="bg-blue-600 text-white text-xs px-2.5 py-0.5 rounded-full font-medium">
                  {added_edges.length + removed_edges.length}
                </span>
              </div>
              {added_edges.length === 0 && removed_edges.length === 0 ? (
                <div className="text-[#6b7280] text-sm italic">No edge changes</div>
              ) : (
                <div className="space-y-3">
                  {added_edges.length > 0 && (
                    <div>
                      <h5 className="text-sm font-medium text-green-400 mb-2">Added Edges:</h5>
                      <div className="space-y-1">
                        {added_edges.map((edge, index) => {
                          const srcName = edge.source.split('-')[0].charAt(0).toUpperCase() + edge.source.split('-')[0].slice(1);
                          const tgtName = edge.target.split('-')[0].charAt(0).toUpperCase() + edge.target.split('-')[0].slice(1);
                          return (
                            <div key={index} className="bg-[#14532d] text-green-300 border border-green-800/40 px-3 py-2 rounded-md text-sm">
                              + {srcName} → {tgtName}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  {removed_edges.length > 0 && (
                    <div>
                      <h5 className="text-sm font-medium text-red-400 mb-2">Removed Edges:</h5>
                      <div className="space-y-1">
                        {removed_edges.map((edge, index) => {
                          const srcName = edge.source.split('-')[0].charAt(0).toUpperCase() + edge.source.split('-')[0].slice(1);
                          const tgtName = edge.target.split('-')[0].charAt(0).toUpperCase() + edge.target.split('-')[0].slice(1);
                          return (
                            <div key={index} className="bg-[#7f1d1d] text-red-300 border border-red-800/40 px-3 py-2 rounded-md text-sm">
                              − {srcName} → {tgtName}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-[#334155] px-6 py-4 flex justify-end">
          <button onClick={onClose} className="px-6 h-9 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition-colors duration-150 text-sm font-medium">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DiffViewer;
