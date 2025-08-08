import React from 'react';
import { usePredictSession } from '../../contexts/PredictSessionContext';

const ActionInfoPanel = () => {
  const { 
    meshData,
    logs,
    status
  } = usePredictSession();

  // Extract the latest action info from logs or mesh data
  const getLatestActionInfo = () => {
    // Look for the most recent action-related log
    const actionLogs = logs
      .filter(log => log.data && (log.data.action_type || log.data.meshData))
      .slice(-1);
    
    if (actionLogs.length > 0) {
      const latestLog = actionLogs[0];
      return latestLog.data;
    }
    
    // Fallback to meshData if available
    if (meshData && meshData.lastAction) {
      return meshData.lastAction;
    }
    
    return null;
  };

  const actionInfo = getLatestActionInfo();

  const getValidityColor = (valid) => {
    if (valid === true) return 'text-green-600 bg-green-50';
    if (valid === false) return 'text-red-600 bg-red-50';
    return 'text-gray-500 bg-gray-50';
  };

  const getValidityIcon = (valid) => {
    if (valid === true) return '✓';
    if (valid === false) return '✗';
    return '?';
  };

  const formatCoordinates = (coords) => {
    if (!coords) return 'N/A';
    
    if (Array.isArray(coords)) {
      if (coords.length === 2 && typeof coords[0] === 'number') {
        // Single coordinate pair
        return `(${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`;
      } else if (coords.length > 0 && Array.isArray(coords[0])) {
        // Multiple coordinate pairs
        return coords
          .map(coord => `(${coord[0].toFixed(2)}, ${coord[1].toFixed(2)})`)
          .join(', ');
      }
    }
    
    return String(coords);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Action Information</h3>
        <div className="w-2 h-2 rounded-full bg-blue-500" />
      </div>
      
      <div className="space-y-3">
        {/* Action Type */}
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-600">Action Type</span>
          <span className="text-xs font-medium text-gray-800 bg-gray-50 px-2 py-1 rounded">
            {actionInfo?.action_type || actionInfo?.action_name || 'None'}
          </span>
        </div>

        {/* Vertex Information */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Reference Vertex</span>
          <span className="text-xs font-mono text-gray-800">
            {actionInfo?.reference_vertex_idx ?? 
             actionInfo?.vertex_index ?? 
             actionInfo?.refPointIndex ?? 'N/A'}
          </span>
        </div>

        {/* Validity Status */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Validity</span>
          <div className={`text-xs font-medium px-2 py-1 rounded-full ${getValidityColor(actionInfo?.valid)}`}>
            <span className="mr-1">{getValidityIcon(actionInfo?.valid)}</span>
            {actionInfo?.valid === true ? 'Valid' : 
             actionInfo?.valid === false ? 'Invalid' : 
             'Unknown'}
          </div>
        </div>

        {/* Coordinates */}
        <div className="py-2 border-t border-gray-100">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs text-gray-600">Coordinates</span>
          </div>
          
          {/* Click Point / Generated Element */}
          {actionInfo?.clicked_point && (
            <div className="mb-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Click Point:</span>
                <span className="text-xs font-mono text-gray-700">
                  {formatCoordinates(actionInfo.clicked_point)}
                </span>
              </div>
            </div>
          )}
          
          {actionInfo?.decoded_coords && (
            <div className="mb-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Decoded:</span>
                <span className="text-xs font-mono text-gray-700">
                  {formatCoordinates(actionInfo.decoded_coords)}
                </span>
              </div>
            </div>
          )}
          
          {actionInfo?.generated_element && (
            <div>
              <div className="text-xs text-gray-500 mb-1">Generated Element:</div>
              <div className="text-xs font-mono text-gray-700 bg-gray-50 p-2 rounded max-h-16 overflow-y-auto">
                {actionInfo.generated_element.map((coord, idx) => (
                  <div key={idx}>
                    {idx + 1}: {formatCoordinates(coord)}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {!actionInfo?.clicked_point && !actionInfo?.decoded_coords && !actionInfo?.generated_element && (
            <span className="text-xs text-gray-400 italic">No coordinate data available</span>
          )}
        </div>

        {/* Polar Coordinates (if available) */}
        {actionInfo?.polar_coordinates && (
          <div className="flex justify-between items-center py-2 border-t border-gray-100">
            <span className="text-xs text-gray-600">Polar</span>
            <span className="text-xs font-mono text-gray-800">
              r: {actionInfo.polar_coordinates.r?.toFixed(3)}, 
              θ: {actionInfo.polar_coordinates.theta?.toFixed(3)}
            </span>
          </div>
        )}

        {/* Status Indicator */}
        <div className="pt-2 border-t border-gray-100">
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-600">Status</span>
            <span className={`text-xs px-2 py-1 rounded ${
              status === 'running' ? 'bg-green-100 text-green-700' :
              status === 'paused' ? 'bg-yellow-100 text-yellow-700' :
              status === 'error' ? 'bg-red-100 text-red-700' :
              'bg-gray-100 text-gray-700'
            }`}>
              {status === 'running' ? 'Active' :
               status === 'paused' ? 'Paused' :
               status === 'error' ? 'Error' :
               'Idle'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ActionInfoPanel;
