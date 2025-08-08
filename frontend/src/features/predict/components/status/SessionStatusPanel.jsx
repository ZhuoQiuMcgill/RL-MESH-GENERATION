import React from 'react';
import { usePredictSession } from '../../contexts/PredictSessionContext';

const SessionStatusPanel = () => {
  const { 
    currentStep, 
    totalSteps, 
    progress, 
    status, 
    startTime, 
    endTime 
  } = usePredictSession();

  const getStatusColor = (status) => {
    switch (status) {
      case 'idle': return 'text-gray-500';
      case 'configuring': return 'text-blue-500';
      case 'initializing': return 'text-yellow-500';
      case 'running': return 'text-green-500';
      case 'paused': return 'text-orange-500';
      case 'completed': return 'text-emerald-600';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const formatDuration = (startTime, endTime) => {
    if (!startTime) return '00:00:00';
    
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const duration = Math.floor((end - start) / 1000);
    
    const hours = Math.floor(duration / 3600);
    const minutes = Math.floor((duration % 3600) / 60);
    const seconds = duration % 60;
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Session Status</h3>
        <span className={`text-xs font-semibold px-2 py-1 rounded-full ${getStatusColor(status)} bg-opacity-10`}>
          {status.toUpperCase()}
        </span>
      </div>
      
      <div className="space-y-3">
        {/* Step Progress */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-600">Steps</span>
            <span className="text-xs font-mono text-gray-800">
              {currentStep} / {totalSteps || '∞'}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {progress.toFixed(1)}% Complete
          </div>
        </div>

        {/* Boundary Status */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Boundary Status</span>
          <span className="text-xs font-medium text-gray-800">
            {status === 'running' || status === 'completed' ? 'Active' : 'Inactive'}
          </span>
        </div>

        {/* Duration */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Duration</span>
          <span className="text-xs font-mono text-gray-800">
            {formatDuration(startTime, endTime)}
          </span>
        </div>

        {/* Completion Status */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Completion</span>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              status === 'completed' ? 'bg-green-500' : 
              status === 'running' ? 'bg-blue-500 animate-pulse' : 
              'bg-gray-300'
            }`} />
            <span className="text-xs font-medium text-gray-800">
              {status === 'completed' ? 'Finished' : 
               status === 'running' ? 'In Progress' : 
               'Pending'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionStatusPanel;
