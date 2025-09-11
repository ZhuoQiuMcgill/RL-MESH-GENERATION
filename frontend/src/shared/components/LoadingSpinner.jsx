import React from 'react';

const LoadingSpinner = ({ 
  size = 'md', 
  variant = 'primary', 
  message = '', 
  className = '',
  progress = null // For progress-based loading (0-100)
}) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8', 
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  const variants = {
    primary: 'border-blue-200 border-t-blue-600',
    secondary: 'border-gray-200 border-t-gray-600',
    success: 'border-green-200 border-t-green-600',
    warning: 'border-yellow-200 border-t-yellow-600',
    error: 'border-red-200 border-t-red-600'
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-3 ${className}`}>
      {/* Progress Ring for Progress-based Loading */}
      {progress !== null ? (
        <div className="relative">
          <div className={`${sizes[size]} border-4 border-gray-200 rounded-full`}></div>
          <div
            className={`absolute inset-0 ${sizes[size]} border-4 border-transparent border-t-blue-600 rounded-full animate-spin`}
            style={{
              background: `conic-gradient(from 0deg, rgb(37 99 235) ${progress * 3.6}deg, transparent ${progress * 3.6}deg)`
            }}
          ></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-medium text-gray-600">{Math.round(progress)}%</span>
          </div>
        </div>
      ) : (
        /* Standard Spinner */
        <div
          className={`${sizes[size]} border-4 ${variants[variant]} rounded-full animate-spin`}
        ></div>
      )}
      
      {/* Loading Message */}
      {message && (
        <p className="text-sm text-gray-600 text-center max-w-xs">
          {message}
        </p>
      )}
    </div>
  );
};

export default LoadingSpinner;
