import React from 'react';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';
import { OperationLog } from '../features/predict/components';
import { useOperationLog, LogType } from '../features/predict/hooks/useOperationLog';

// Demo control buttons component
const DemoControls = () => {
  const { addLog, clearLog } = useOperationLog();

  const handleDemoActions = () => {
    addLog(LogType.SYSTEM, 'System initialization completed');
    setTimeout(() => addLog(LogType.USER_ACTION, 'User modified configuration parameters'), 500);
    setTimeout(() => addLog(LogType.API_SUCCESS, 'Server status retrieved successfully'), 1000);
    setTimeout(() => addLog(LogType.PREDICTION, 'Starting mesh generation prediction...'), 1500);
    setTimeout(() => addLog(LogType.MESH, 'Mesh quality check passed'), 2000);
    setTimeout(() => addLog(LogType.API_ERROR, 'Network connection timeout, attempting to reconnect...'), 2500);
  };

  return (
    <div className="space-y-2 sm:space-y-3">
      <button
        onClick={handleDemoActions}
        className="
          w-full bg-primary-600 hover:bg-primary-700 
          text-white px-3 py-2 sm:px-4 
          rounded-md text-xs sm:text-sm 
          transition-colors duration-200
          font-medium
        "
      >
        Demo Logs
      </button>
      <button
        onClick={() => addLog(LogType.USER_ACTION, 'User clicked button')}
        className="
          w-full bg-success hover:bg-success/90 
          text-white px-3 py-2 sm:px-4 
          rounded-md text-xs sm:text-sm 
          transition-colors duration-200
          font-medium
        "
      >
        Add Single Log
      </button>
      <button
        onClick={clearLog}
        className="
          w-full bg-error hover:bg-error/90 
          text-white px-3 py-2 sm:px-4 
          rounded-md text-xs sm:text-sm 
          transition-colors duration-200
          font-medium
        "
      >
        Clear Logs
      </button>
    </div>
  );
};

const PredictPage = () => {
  return (
    <PredictSessionProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Responsive three-column layout container */}
        <div className="flex flex-col lg:flex-row h-screen lg:h-auto">
          
          {/* Left sidebar - Configuration panel */}
          <div className="
            w-full lg:w-[350px] 
            bg-white dark:bg-gray-800 
            border-b lg:border-b-0 lg:border-r 
            border-gray-200 dark:border-gray-700 
            flex-shrink-0
            order-1 lg:order-1
          ">
            <div className="p-3 sm:p-4 h-full">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-3 sm:mb-4">
                Configuration Panel
              </h3>
              {/* Left configuration panel placeholder */}
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4 sm:p-6 h-48 sm:h-64 lg:h-96">
                <p className="text-sm sm:text-base text-gray-600 dark:text-gray-300 text-center">
                  Configuration controls will be placed here
                </p>
              </div>
            </div>
          </div>

          {/* Center main content area - Prediction canvas */}
          <div className="
            flex-1 
            bg-white dark:bg-gray-800 
            min-w-0
            order-2 lg:order-2
          ">
            <div className="p-3 sm:p-4 h-full">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-3 sm:mb-4">
                Prediction Canvas
              </h3>
              {/* Main prediction canvas placeholder */}
              <div className="
                bg-gray-100 dark:bg-gray-700 
                rounded-lg 
                h-64 sm:h-80 lg:h-full 
                flex items-center justify-center
              ">
                <p className="text-sm sm:text-base text-gray-600 dark:text-gray-300 text-center px-4">
                  Mesh prediction visualization canvas will be placed here
                </p>
              </div>
            </div>
          </div>

          {/* Right sidebar - Control panel */}
          <div className="
            w-full lg:w-[320px] 
            bg-white dark:bg-gray-800 
            border-t lg:border-t-0 lg:border-l 
            border-gray-200 dark:border-gray-700 
            flex-shrink-0
            order-3 lg:order-3
          ">
            <div className="p-3 sm:p-4 h-full">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-3 sm:mb-4">
                Control Panel
              </h3>
              
              {/* Right control panel - Demo controls */}
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3 sm:p-4 mb-3 sm:mb-4">
                <h4 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 sm:mb-3">
                  Demo Controls
                </h4>
                <DemoControls />
              </div>
              
              {/* Operation log component - Responsive height */}
              <div className="h-48 sm:h-60 lg:h-auto lg:flex-1">
                <OperationLog height={240} className="h-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </PredictSessionProvider>
  );
};

export default PredictPage;
