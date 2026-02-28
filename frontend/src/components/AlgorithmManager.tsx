import React, { useState } from 'react';
import { Plus, Trash2, Code, CheckCircle, ArrowLeft } from 'lucide-react';
import { useStore, useAlgorithms } from '../store/useStore';
import { CustomAlgorithm } from '../types';
import { apiClient } from '../api/client';

interface AlgorithmManagerProps {
  onBack?: () => void;
}

const AlgorithmManager: React.FC<AlgorithmManagerProps> = ({ onBack }) => {
  const algorithms = useAlgorithms();
  const { saveAlgorithm, fetchAlgorithms, setLoading, error, clearError } = useStore();
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    code: '',
  });
  const [showSuccess, setShowSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim() || !formData.code.trim()) {
      return;
    }

    try {
      setLoading(true);
      await saveAlgorithm(formData.name, formData.description, formData.code);
      
      // Reset form
      setFormData({ name: '', description: '', code: '' });
      setShowSuccess(true);
      
      // Hide success message after 3 seconds
      setTimeout(() => setShowSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to save algorithm:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (algorithmId: number) => {
    try {
      await apiClient.deleteAlgorithm(algorithmId);
      await fetchAlgorithms();
    } catch (error) {
      console.error('Failed to delete algorithm:', error);
    }
  };

  const getCodePreview = (code: string) => {
    const lines = code.split('\n').slice(0, 3);
    return lines.join('\n');
  };

  return (
    <div className="p-6 bg-gray-900 min-h-full">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center space-x-4">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
              title="Back to main view"
            >
              <ArrowLeft size={20} />
            </button>
          )}
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Custom Algorithms</h1>
            <p className="text-gray-400">Create and manage your custom ML algorithms</p>
          </div>
        </div>

        {/* Success Message */}
        {showSuccess && (
          <div className="mb-6 bg-green-600 text-white px-4 py-3 rounded-lg flex items-center space-x-2">
            <CheckCircle size={20} />
            <span>Algorithm saved successfully!</span>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-600 text-white px-4 py-3 rounded-lg flex items-center justify-between">
            <span>{error}</span>
            <button onClick={clearError} className="text-white/80 hover:text-white">
              ×
            </button>
          </div>
        )}

        {/* Create Form */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <Plus size={20} />
            <span>Create New Algorithm</span>
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Algorithm name"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <input
                  type="text"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Brief description"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Python Code *
              </label>
              <textarea
                value={formData.code}
                onChange={(e) => setFormData({ ...formData, code: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 h-64 resize-y"
                placeholder="class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x"
                required
              />
            </div>
            
            <button
              type="submit"
              disabled={!formData.name.trim() || !formData.code.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              <Code size={16} />
              <span>Save Algorithm</span>
            </button>
          </form>
        </div>

        {/* Algorithms List */}
        <div>
          <h2 className="text-xl font-semibold text-white mb-4">Saved Algorithms</h2>
          
          {algorithms.length === 0 ? (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <Code className="mx-auto text-gray-400 mb-3" size={48} />
              <p className="text-gray-400">No algorithms saved yet</p>
              <p className="text-gray-500 text-sm mt-1">Create your first custom algorithm above</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {algorithms.map((algorithm) => (
                <div key={algorithm.id} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold text-white truncate flex-1">
                      {algorithm.name}
                    </h3>
                    <button
                      onClick={() => handleDelete(algorithm.id)}
                      className="text-gray-400 hover:text-red-400 transition-colors ml-2"
                      title="Delete algorithm"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  
                  {algorithm.description && (
                    <p className="text-gray-400 text-sm mb-3 line-clamp-2">
                      {algorithm.description}
                    </p>
                  )}
                  
                  <div className="bg-gray-900 rounded p-3 mb-3">
                    <pre className="text-xs text-gray-300 font-mono overflow-x-auto">
                      {getCodePreview(algorithm.code)}
                      {algorithm.code.split('\n').length > 3 && '\n...'}
                    </pre>
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    Created: {new Date(algorithm.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AlgorithmManager;
