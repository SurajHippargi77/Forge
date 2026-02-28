import React, { useEffect, useState } from 'react';
import { RefreshCw, TrendingUp, Activity, CheckCircle, XCircle, Clock, Zap, BarChart3 } from 'lucide-react';
import { useStore, useActiveVersion, useExperiments, useVersions } from '../store/useStore';
import { ExperimentRun, ExperimentStatus } from '../types';

const ExperimentDashboard: React.FC = () => {
  const activeVersion = useActiveVersion();
  const experiments = useExperiments();
  const versions = useVersions();
  const { fetchExperiments, setLoading, setActiveVersion } = useStore();
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (!activeVersion && versions.length > 0) {
      setActiveVersion(versions[versions.length - 1]);
    }
  }, [activeVersion, versions, setActiveVersion]);

  useEffect(() => {
    if (activeVersion) {
      fetchExperiments(activeVersion.id);
    }
  }, [activeVersion, fetchExperiments]);

  useEffect(() => {
    const hasActiveExperiments = experiments.some(
      exp => exp.status === 'running' || exp.status === 'pending'
    );
    if (!hasActiveExperiments || !activeVersion) return;
    const interval = setInterval(() => {
      fetchExperiments(activeVersion.id);
    }, 3000);
    return () => clearInterval(interval);
  }, [experiments, activeVersion, fetchExperiments]);

  const handleRefresh = async () => {
    if (!activeVersion) return;
    setIsRefreshing(true);
    try { await fetchExperiments(activeVersion.id); }
    catch (error) { console.error('Failed to refresh experiments:', error); }
    finally { setIsRefreshing(false); }
  };

  const getStatusBadge = (status: ExperimentStatus) => {
    const configs = {
      pending:   { color: 'bg-gray-600/80', text: 'Pending',   icon: Clock,       animate: false },
      running:   { color: 'bg-blue-600/80',  text: 'Running',   icon: Activity,    animate: true },
      completed: { color: 'bg-green-600/80', text: 'Completed', icon: CheckCircle, animate: false },
      failed:    { color: 'bg-red-600/80',   text: 'Failed',    icon: XCircle,     animate: false },
    };
    const config = configs[status];
    const Icon = config.icon;
    return (
      <span className={`${config.color} text-white text-xs px-3 py-1 rounded-full flex items-center space-x-1.5 w-fit ${config.animate ? 'animate-pulse' : ''}`}>
        <Icon size={11} />
        <span className="font-medium">{config.text}</span>
      </span>
    );
  };

  const getBestExperiment = () => {
    const completed = experiments.filter(exp => exp.status === 'completed' && exp.metrics?.val_loss);
    return completed.reduce((best, current) => {
      if (!best || !current.metrics?.val_loss) return current;
      if (!best.metrics?.val_loss) return current;
      return current.metrics.val_loss < best.metrics.val_loss ? current : best;
    }, null as ExperimentRun | null);
  };

  const getBestAccuracy = () => {
    const completed = experiments.filter(exp => exp.status === 'completed' && exp.metrics?.accuracy);
    return completed.reduce((best, current) => {
      if (!best || !current.metrics?.accuracy) return current;
      if (!best.metrics?.accuracy) return current;
      return current.metrics.accuracy > best.metrics.accuracy ? current : best;
    }, null as ExperimentRun | null);
  };

  const bestExperiment = getBestExperiment();
  const bestAccuracyExperiment = getBestAccuracy();

  const completedCount = experiments.filter(e => e.status === 'completed').length;
  const runningCount = experiments.filter(e => e.status === 'running').length;
  const pendingCount = experiments.filter(e => e.status === 'pending').length;

  if (!activeVersion) {
    return (
      <div className="flex items-center justify-center h-full bg-[#0f0f1a]">
        <div className="text-center">
          <Activity className="mx-auto text-gray-500 mb-4" size={48} />
          <h3 className="text-xl font-semibold text-white mb-2">No Version Selected</h3>
          <p className="text-gray-500">Select a version to view experiments</p>
        </div>
      </div>
    );
  }

  const sortedExperiments = [...experiments].sort((a, b) => {
    if (a.status === 'completed' && b.status !== 'completed') return -1;
    if (a.status !== 'completed' && b.status === 'completed') return 1;
    if (a.status === 'completed' && b.status === 'completed') {
      const aLoss = a.metrics?.val_loss || Infinity;
      const bLoss = b.metrics?.val_loss || Infinity;
      if (aLoss !== bLoss) return aLoss - bLoss;
    }
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  return (
    <div className="p-6 bg-[#0f0f1a] min-h-full pb-16">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Experiments</h1>
            <p className="text-gray-500 text-sm">
              Version {activeVersion.version_number} · {experiments.length} experiment{experiments.length !== 1 ? 's' : ''}
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="flex items-center space-x-2 px-4 h-9 bg-gray-800 text-gray-300 rounded-md hover:bg-gray-700 disabled:opacity-50 transition-colors duration-150 border border-gray-700"
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
            <span className="text-sm font-medium">Refresh</span>
          </button>
        </div>

        {/* Summary Cards */}
        {experiments.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
            <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
              <p className="text-gray-500 text-[11px] uppercase tracking-wider font-medium">Total</p>
              <p className="text-2xl font-bold text-white mt-0.5">{experiments.length}</p>
            </div>
            <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
              <p className="text-green-500 text-[11px] uppercase tracking-wider font-medium">Completed</p>
              <p className="text-2xl font-bold text-green-400 mt-0.5">{completedCount}</p>
            </div>
            <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
              <p className="text-blue-500 text-[11px] uppercase tracking-wider font-medium">Running</p>
              <p className="text-2xl font-bold text-blue-400 mt-0.5">{runningCount}</p>
            </div>
            {bestExperiment && (
              <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
                <p className="text-gray-500 text-[11px] uppercase tracking-wider font-medium flex items-center gap-1"><TrendingUp size={10} /> Best Loss</p>
                <p className="text-2xl font-bold text-green-400 mt-0.5">{bestExperiment.metrics?.val_loss?.toFixed(4)}</p>
              </div>
            )}
            {bestAccuracyExperiment && (
              <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
                <p className="text-gray-500 text-[11px] uppercase tracking-wider font-medium flex items-center gap-1"><BarChart3 size={10} /> Best Acc</p>
                <p className="text-2xl font-bold text-blue-400 mt-0.5">{((bestAccuracyExperiment.metrics?.accuracy || 0) * 100).toFixed(1)}%</p>
              </div>
            )}
          </div>
        )}

        {/* Experiments Table */}
        {experiments.length === 0 ? (
          <div className="bg-gray-800/40 rounded-xl p-8 text-center border border-gray-700/50">
            <Activity className="mx-auto text-gray-600 mb-4" size={48} />
            <h3 className="text-lg font-semibold text-white mb-2">No Experiments Yet</h3>
            <p className="text-gray-500 text-sm">Run a sweep from the Graph Editor to get started</p>
          </div>
        ) : (
          <div className="bg-gray-800/40 rounded-xl overflow-hidden border border-gray-700/50">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0f172a]">
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">ID</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Status</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Learning Rate</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Batch Size</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Epochs</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Optimizer</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Train Loss</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Val Loss</th>
                    <th className="px-4 py-3 text-left text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700/50">
                  {sortedExperiments.map((experiment) => {
                    const isBest = bestExperiment?.id === experiment.id;
                    const accuracy = experiment.metrics?.accuracy;
                    return (
                      <tr
                        key={experiment.id}
                        className={`hover:bg-gray-700/30 transition-colors duration-150 ${isBest ? 'border-l-4 border-l-green-500 bg-green-900/10' : 'border-l-4 border-l-transparent'}`}
                      >
                        <td className="px-4 py-3 text-sm text-white font-medium">
                          <span className="flex items-center gap-2">
                            #{experiment.id}
                            {isBest && (
                              <span className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full font-semibold border border-green-500/30">
                                Best
                              </span>
                            )}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {getStatusBadge(experiment.status)}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300 font-mono">
                          {experiment.hyperparameters.learning_rate || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">
                          {experiment.hyperparameters.batch_size || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">
                          {experiment.hyperparameters.epochs || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">
                          {experiment.hyperparameters.optimizer || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300 font-mono">
                          {experiment.metrics?.train_loss?.toFixed(4) || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300 font-mono">
                          {experiment.metrics?.val_loss?.toFixed(4) || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {accuracy != null ? (
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{
                                    width: `${Math.min(accuracy * 100, 100)}%`,
                                    backgroundColor: accuracy >= 0.9 ? '#22c55e' : accuracy >= 0.8 ? '#3b82f6' : '#f59e0b',
                                  }}
                                />
                              </div>
                              <span className="text-gray-300 font-mono text-xs">{(accuracy * 100).toFixed(1)}%</span>
                            </div>
                          ) : (
                            <span className="text-gray-600">-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExperimentDashboard;
