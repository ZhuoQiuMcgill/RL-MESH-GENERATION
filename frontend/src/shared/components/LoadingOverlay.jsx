import React from 'react';

const LoadingOverlay = ({ isLoading, message = '正在加载...' }) => {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 shadow-lg max-w-sm mx-4">
        <div className="flex items-center space-x-3">
          {/* Loading Spinner */}
          <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-500 border-t-transparent"></div>
          <div className="text-gray-700">{message}</div>
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;
