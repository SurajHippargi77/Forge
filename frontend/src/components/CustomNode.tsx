import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { X } from 'lucide-react';

interface CustomNodeData {
  type: string;
  label: string;
  params: Record<string, any>;
  onDelete: (nodeId: string) => void;
  onParamsChange: (nodeId: string, params: Record<string, any>) => void;
}

// Node type configurations
const nodeTypeConfig = {
  input: { color: 'bg-blue-500', label: 'Input' },
  dense: { color: 'bg-purple-500', label: 'Dense', params: ['units'] },
  conv2d: { color: 'bg-orange-500', label: 'Conv2D', params: ['filters', 'kernel_size'] },
  relu: { color: 'bg-green-500', label: 'ReLU' },
  batchnorm: { color: 'bg-yellow-500', label: 'BatchNorm' },
  dropout: { color: 'bg-red-500', label: 'Dropout', params: ['rate'] },
  output: { color: 'bg-blue-500', label: 'Output' },
  custom: { color: 'bg-gray-500', label: 'Custom' },
};

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ id, data, selected }) => {
  const config = nodeTypeConfig[data.type as keyof typeof nodeTypeConfig] || nodeTypeConfig.custom;
  const hasParams = config.params && config.params.length > 0;

  const handleParamChange = (paramName: string, value: string) => {
    const newParams = { ...data.params };
    
    // Convert to number if it's a numeric parameter
    if (paramName === 'units' || paramName === 'filters' || paramName === 'kernel_size' || paramName === 'rate') {
      const numValue = parseFloat(value);
      newParams[paramName] = isNaN(numValue) ? 0 : numValue;
    } else {
      newParams[paramName] = value;
    }
    
    data.onParamsChange(id, newParams);
  };

  const getParamValue = (paramName: string) => {
    return data.params[paramName] || '';
  };

  return (
    <div
      className={`
        bg-white rounded-lg shadow-lg border-2 transition-all duration-200 min-w-[150px]
        ${selected ? 'border-blue-400 shadow-xl' : 'border-gray-200'}
      `}
    >
      {/* Header */}
      <div className={`${config.color} text-white px-3 py-2 rounded-t-lg flex justify-between items-center`}>
        <span className="font-semibold text-sm">{data.label}</span>
        <button
          onClick={() => data.onDelete(id)}
          className="hover:bg-white/20 rounded p-0.5 transition-colors"
          title="Delete node"
        >
          <X size={14} />
        </button>
      </div>

      {/* Parameters */}
      {hasParams && (
        <div className="p-3 space-y-2">
          {config.params!.map((paramName) => (
            <div key={paramName} className="flex flex-col space-y-1">
              <label className="text-xs font-medium text-gray-600 capitalize">
                {paramName.replace('_', ' ')}
              </label>
              <input
                type={paramName === 'rate' ? 'number' : 'text'}
                step={paramName === 'rate' ? '0.1' : '1'}
                min={paramName === 'rate' ? '0' : '1'}
                value={getParamValue(paramName)}
                onChange={(e) => handleParamChange(paramName, e.target.value)}
                className="w-full px-2 py-1 text-sm text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
                placeholder={`Enter ${paramName}`}
              />
            </div>
          ))}
        </div>
      )}

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-gray-400 border-2 border-white"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-gray-400 border-2 border-white"
      />
    </div>
  );
};

export default memo(CustomNode);
