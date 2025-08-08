import React from 'react';
import { usePredictSession } from '../../contexts/PredictSessionContext';

const ReferencePointPanel = () => {
  const { 
    refPoint,
    logs,
    status
  } = usePredictSession();

  // Extract reference point info from logs if refPoint is not available
  const getReferencePointInfo = () => {
    if (refPoint) {
      return refPoint;
    }

    // Look for reference point information in logs
    const refPointLogs = logs
      .filter(log => log.data && (log.data.reference_point || log.data.refPoint || log.data.index !== undefined))
      .slice(-1);
    
    if (refPointLogs.length > 0) {
      const latestLog = refPointLogs[0];
      return latestLog.data.reference_point || latestLog.data.refPoint || latestLog.data;
    }
    
    return null;
  };

  const refPointInfo = getReferencePointInfo();

  const formatCoordinates = (coords) => {
    if (!coords) return 'N/A';
    if (Array.isArray(coords) && coords.length === 2) {
      return `(${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`;
    }
    return String(coords);
  };

  const formatAngle = (angle) => {
    if (angle === null || angle === undefined) return 'N/A';
    return `${angle.toFixed(2)}°`;
  };

  const getAngleColor = (angle) => {
    if (!angle) return 'text-gray-500';
    
    if (angle < 30 || angle > 150) return 'text-red-500'; // Sharp or very obtuse
    if (angle >= 80 && angle <= 100) return 'text-green-500'; // Near right angle
    return 'text-yellow-500'; // Moderate angle
  };

  const getAngleQuality = (angle) => {
    if (!angle) return 'Unknown';
    
    if (angle < 30 || angle > 150) return 'Poor';
    if (angle >= 80 && angle <= 100) return 'Excellent';
    if (angle >= 60 && angle <= 120) return 'Good';
    return 'Fair';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Reference Point</h3>
        <div className={`w-2 h-2 rounded-full ${
          refPointInfo ? 'bg-green-500' : 'bg-gray-300'
        }`} />
      </div>
      
      <div className="space-y-3">
        {/* Selector Information */}
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-600">Selector Index</span>
          <span className="text-xs font-mono text-gray-800 bg-gray-50 px-2 py-1 rounded">
            {refPointInfo?.index ?? refPointInfo?.selector ?? 'None'}
          </span>
        </div>

        {/* Coordinates */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Coordinates</span>
          <span className="text-xs font-mono text-gray-800">
            {formatCoordinates(refPointInfo?.coordinates)}
          </span>
        </div>

        {/* Interior Angle */}
        <div className="py-2 border-t border-gray-100">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs text-gray-600">Interior Angle</span>
            <span className={`text-xs font-mono ${getAngleColor(refPointInfo?.interior_angle)}`}>
              {formatAngle(refPointInfo?.interior_angle)}
            </span>
          </div>
          
          {/* Angle Quality Indicator */}
          {refPointInfo?.interior_angle && (
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Quality</span>
              <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                getAngleQuality(refPointInfo.interior_angle) === 'Excellent' ? 'bg-green-100 text-green-700' :
                getAngleQuality(refPointInfo.interior_angle) === 'Good' ? 'bg-blue-100 text-blue-700' :
                getAngleQuality(refPointInfo.interior_angle) === 'Fair' ? 'bg-yellow-100 text-yellow-700' :
                'bg-red-100 text-red-700'
              }`}>
                {getAngleQuality(refPointInfo.interior_angle)}
              </span>
            </div>
          )}
        </div>

        {/* Neighbor Vertices */}
        {refPointInfo?.neighbor_vertices && (
          <div className="py-2 border-t border-gray-100">
            <div className="text-xs text-gray-600 mb-2">Neighbor Vertices</div>
            <div className="max-h-20 overflow-y-auto bg-gray-50 rounded p-2">
              {refPointInfo.neighbor_vertices.map((vertex, idx) => (
                <div key={idx} className="text-xs font-mono text-gray-700">
                  {idx + 1}: {formatCoordinates(vertex)}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Additional Properties */}
        {refPointInfo?.element_index !== undefined && (
          <div className="flex justify-between items-center py-2 border-t border-gray-100">
            <span className="text-xs text-gray-600">Element Index</span>
            <span className="text-xs font-mono text-gray-800">
              {refPointInfo.element_index}
            </span>
          </div>
        )}

        {refPointInfo?.boundary_position && (
          <div className="flex justify-between items-center py-2 border-t border-gray-100">
            <span className="text-xs text-gray-600">Boundary Position</span>
            <span className="text-xs font-medium text-gray-800">
              {refPointInfo.boundary_position}
            </span>
          </div>
        )}

        {/* Status */}
        <div className="pt-2 border-t border-gray-100">
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-600">Point Status</span>
            <span className={`text-xs px-2 py-1 rounded ${
              refPointInfo ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
            }`}>
              {refPointInfo ? 'Selected' : 'Not Set'}
            </span>
          </div>
        </div>

        {/* Visual Angle Indicator */}
        {refPointInfo?.interior_angle && (
          <div className="pt-2 border-t border-gray-100">
            <div className="text-xs text-gray-600 mb-2">Angle Visualization</div>
            <div className="relative h-6 bg-gray-100 rounded overflow-hidden">
              <div 
                className={`h-full transition-all duration-300 ${
                  getAngleQuality(refPointInfo.interior_angle) === 'Excellent' ? 'bg-green-500' :
                  getAngleQuality(refPointInfo.interior_angle) === 'Good' ? 'bg-blue-500' :
                  getAngleQuality(refPointInfo.interior_angle) === 'Fair' ? 'bg-yellow-500' :
                  'bg-red-500'
                }`}
                style={{ 
                  width: `${Math.min((refPointInfo.interior_angle / 180) * 100, 100)}%`
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xs font-medium text-white mix-blend-difference">
                  {refPointInfo.interior_angle.toFixed(1)}°
                </span>
              </div>
            </div>
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0°</span>
              <span>90°</span>
              <span>180°</span>
            </div>
          </div>
        )}

        {/* No Data State */}
        {!refPointInfo && (
          <div className="text-center py-4">
            <div className="text-xs text-gray-400 italic">
              No reference point selected
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Start a prediction to see reference point data
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReferencePointPanel;
