import React, { useRef, useEffect } from 'react';
import { useOperationLog, LogType } from '../hooks/useOperationLog';

const OperationLog = ({ height = 300, className = '' }) => {
  const { logs, clearLog, LogType: Types } = useOperationLog();
  const logEndRef = useRef(null);

  // Auto scroll to latest log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Get log type CSS class name
  const getLogTypeClass = (type, level) => {
    if (level === 'error' || type === Types.API_ERROR || type === Types.ERROR) {
      return 'text-red-600 bg-red-50';
    }
    if (type === Types.API_SUCCESS) {
      return 'text-green-600 bg-green-50';
    }
    if (type === Types.USER_ACTION) {
      return 'text-blue-600 bg-blue-50';
    }
    if (type === Types.SYSTEM) {
      return 'text-gray-600 bg-gray-50';
    }
    if (type === Types.PREDICTION) {
      return 'text-purple-600 bg-purple-50';
    }
    if (type === Types.MESH) {
      return 'text-orange-600 bg-orange-50';
    }
    return 'text-gray-600 bg-gray-50';
  };

  // Get log type display name
  const getLogTypeName = (type) => {
    switch (type) {
      case Types.SYSTEM: return 'SYSTEM';
      case Types.USER_ACTION: return 'USER';
      case Types.API_SUCCESS: return 'API';
      case Types.API_ERROR: return 'ERROR';
      case Types.PREDICTION: return 'PREDICT';
      case Types.MESH: return 'MESH';
      case Types.ERROR: return 'ERROR';
      default: return 'OTHER';
    }
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header bar */}
      <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-400 rounded-full"></div>
          <h3 className="text-sm font-medium text-gray-700">Operation Log</h3>
          <span className="text-xs text-gray-500">({logs.length}/200)</span>
        </div>
        <button
          onClick={clearLog}
          disabled={logs.length === 0}
          className="text-xs text-gray-500 hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Clear
        </button>
      </div>

      {/* Log content area */}
      <div
        className="overflow-y-auto"
        style={{ height: `${height}px` }}
      >
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            No log records
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {logs.map((log) => (
              <div
                key={log.id}
                className={`flex items-start space-x-2 text-xs p-2 rounded ${getLogTypeClass(log.type, log.level)}`}
              >
                {/* Timestamp */}
                <span className="text-gray-500 shrink-0 font-mono">
                  {formatTimestamp(log.timestamp)}
                </span>
                
                {/* Type label */}
                <span className="shrink-0 px-1.5 py-0.5 rounded text-xs font-medium bg-white">
                  {getLogTypeName(log.type)}
                </span>
                
                {/* Message content */}
                <span className="flex-1 break-words">
                  {log.message}
                </span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export default OperationLog;
