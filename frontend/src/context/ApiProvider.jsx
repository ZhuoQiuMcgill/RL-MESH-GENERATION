import React, { createContext, useContext, useCallback, useEffect, useState, useRef } from 'react';
import { ApiClient, withErrorHandling, withRetry, CONSTANTS } from '../core/api/client.js';

// Create the API context
const ApiContext = createContext(null);

/**
 * API Provider - Normalized Context Provider
 * 
 * Uses the normalized ApiClient from core/api/client.js with environment-based configuration.
 * Maintains the same public API for backwards compatibility while providing enhanced features:
 * - Environment-based base URL (VITE_API_BASE_URL)
 * - Organized core API layer structure
 * - All existing methods preserved
 */
export function ApiProvider({ children }) {
  // Singleton instance
  const apiClientRef = useRef(null);
  
  if (!apiClientRef.current) {
    apiClientRef.current = new ApiClient();
  }

  const apiClient = apiClientRef.current;

  // Wrap all API methods with error handling and retry
  const createEnhancedMethod = useCallback((method) => {
    return withErrorHandling((...args) => withRetry(() => method.apply(apiClient, args)));
  }, [apiClient]);

  // Create enhanced API client with error handling and retry baked in
  const enhancedApiClient = useCallback(() => {
    const enhancedMethods = {};
    
    // Get all methods from the API client
    const methodNames = Object.getOwnPropertyNames(Object.getPrototypeOf(apiClient))
      .filter(name => name !== 'constructor' && typeof apiClient[name] === 'function');
    
    // Wrap each method with error handling and retry
    methodNames.forEach(methodName => {
      enhancedMethods[methodName] = createEnhancedMethod(apiClient[methodName]);
    });

    return enhancedMethods;
  }, [apiClient, createEnhancedMethod]);

  const contextValue = {
    apiClient: enhancedApiClient(),
    rawApiClient: apiClient // For cases where raw client is needed
  };

  return (
    <ApiContext.Provider value={contextValue}>
      {children}
    </ApiContext.Provider>
  );
}

/**
 * useApi Hook - Returns the enhanced API client with error handling and retry baked in
 * @returns {Object} Enhanced API client
 */
export function useApi() {
  const context = useContext(ApiContext);
  
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  
  return context.apiClient;
}

/**
 * usePolling Hook - Generic hook for live updates
 * @param {string} endpoint - API endpoint or method name
 * @param {number} interval - Polling interval in milliseconds (default: 2000)
 * @param {Object} options - Additional options
 * @param {boolean} options.enabled - Whether polling is enabled (default: true)
 * @param {Array} options.dependencies - Dependencies that trigger re-polling when changed
 * @param {Function} options.onSuccess - Callback for successful responses
 * @param {Function} options.onError - Callback for errors
 * @param {Array} options.methodArgs - Arguments to pass to the API method
 * @returns {Object} Polling state and controls
 */
export function usePolling(endpoint, interval = CONSTANTS.DEFAULT_POLLING_INTERVAL, options = {}) {
  const {
    enabled = true,
    dependencies = [],
    onSuccess = null,
    onError = null,
    methodArgs = []
  } = options;

  const api = useApi();
  const context = useContext(ApiContext);
  
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  
  const intervalRef = useRef(null);
  const mountedRef = useRef(true);

  // Determine if endpoint is a method name or direct endpoint
  const apiMethod = useCallback(() => {
    if (typeof endpoint === 'string' && api[endpoint] && typeof api[endpoint] === 'function') {
      // It's a method name
      return api[endpoint];
    } else if (typeof endpoint === 'string') {
      // It's a direct endpoint, use the raw request method
      return (...args) => context.rawApiClient.request(endpoint, ...args);
    } else if (typeof endpoint === 'function') {
      // It's already a function
      return endpoint;
    } else {
      throw new Error('Endpoint must be a string (method name or endpoint) or a function');
    }
  }, [endpoint, api, context]);

  const poll = useCallback(async () => {
    if (!mountedRef.current) return;
    
    try {
      setError(null);
      if (!isPolling) setIsLoading(true);
      
      const method = apiMethod();
      const result = await method(...methodArgs);
      
      if (!mountedRef.current) return;
      
      setData(result);
      if (onSuccess) onSuccess(result);
      
    } catch (err) {
      if (!mountedRef.current) return;
      
      setError(err);
      if (onError) onError(err);
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [apiMethod, methodArgs, onSuccess, onError, isPolling]);

  const startPolling = useCallback(() => {
    if (!enabled) return;
    
    setIsPolling(true);
    
    // Initial poll
    poll();
    
    // Set up interval
    intervalRef.current = setInterval(poll, interval);
  }, [enabled, poll, interval]);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const refresh = useCallback(() => {
    poll();
  }, [poll]);

  // Start/stop polling based on enabled flag
  useEffect(() => {
    if (enabled) {
      startPolling();
    } else {
      stopPolling();
    }

    return () => {
      stopPolling();
    };
  }, [enabled, startPolling, stopPolling, ...dependencies]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      stopPolling();
    };
  }, [stopPolling]);

  return {
    data,
    error,
    isLoading,
    isPolling,
    refresh,
    startPolling,
    stopPolling
  };
}

// Export the singleton instance for direct access if needed
export const apiClientInstance = new ApiClient();

export default ApiProvider;
