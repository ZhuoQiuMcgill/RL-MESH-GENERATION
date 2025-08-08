import { useState, useCallback } from 'react';
import { useApi, usePolling } from '../context/ApiProvider';

/**
 * Custom hook for managing mesh boundary data
 * @param {string} mesh - The selected mesh name
 * @returns {object} Boundary data, loading state, and load function
 */
export const useMeshBoundary = (mesh) => {
  const [boundaryData, setBoundaryData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();

  const loadBoundary = useCallback(async (meshName = mesh) => {
    if (!meshName) {
      setBoundaryData(null);
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await api.getMeshBoundary(meshName);
      
      if (data.success) {
        setBoundaryData(data.boundary_vertices);
      } else {
        setError(new Error(data.error || 'Failed to load boundary'));
        setBoundaryData(null);
      }
    } catch (err) {
      setError(err);
      setBoundaryData(null);
    } finally {
      setIsLoading(false);
    }
  }, [api, mesh]);

  const clearBoundary = useCallback(() => {
    setBoundaryData(null);
    setError(null);
  }, []);

  return {
    boundaryData,
    isLoading,
    error,
    loadBoundary,
    clearBoundary
  };
};

/**
 * Custom hook for managing mesh visualization data
 * @param {string} mesh - The selected mesh name
 * @returns {object} Mesh data, loading state, and load function
 */
export const useMeshData = (mesh) => {
  const [meshData, setMeshData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();

  const loadMeshData = useCallback(async (meshName = mesh) => {
    if (!meshName) {
      setMeshData(null);
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await api.getMeshData(meshName);
      
      if (data.success) {
        setMeshData(data.mesh_data);
      } else {
        setError(new Error(data.error || 'Failed to load mesh data'));
        setMeshData(null);
      }
    } catch (err) {
      setError(err);
      setMeshData(null);
    } finally {
      setIsLoading(false);
    }
  }, [api, mesh]);

  const clearMeshData = useCallback(() => {
    setMeshData(null);
    setError(null);
  }, []);

  return {
    meshData,
    isLoading,
    error,
    loadMeshData,
    clearMeshData
  };
};

/**
 * Custom hook for managing reference point data
 * @param {string} mesh - The selected mesh name
 * @returns {object} Reference point info, loading state, and find function
 */
export const useReferencePoint = (mesh) => {
  const [refPointInfo, setRefPointInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();

  const findReferencePoint = useCallback(async (meshName = mesh) => {
    if (!meshName) {
      setRefPointInfo(null);
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await api.getTrainingReferencePoint({ mesh: meshName });
      
      if (data.success) {
        setRefPointInfo(data.reference_point);
      } else {
        setError(new Error(data.error || 'Failed to find reference point'));
        setRefPointInfo(null);
      }
    } catch (err) {
      setError(err);
      setRefPointInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [api, mesh]);

  const clearReferencePoint = useCallback(() => {
    setRefPointInfo(null);
    setError(null);
  }, []);

  return {
    refPointInfo,
    isLoading,
    error,
    findReferencePoint,
    clearReferencePoint
  };
};

/**
 * Custom hook for managing training status with optional polling
 * @param {object} options - Configuration options
 * @param {boolean} options.polling - Enable/disable polling
 * @param {number} options.interval - Polling interval in milliseconds
 * @param {function} options.onStatusChange - Callback when status changes
 * @returns {object} Training status, controls, and polling state
 */
export const useTrainingStatus = ({ 
  polling = false, 
  interval = 2000, 
  onStatusChange = null 
} = {}) => {
  const [trainingStatus, setTrainingStatus] = useState({
    is_training: false,
    status: 'idle',
    episode: 0,
    total_episodes: 0,
    current_reward: 0,
    best_reward: 0,
    elapsed_time: 0,
    last_updated: null
  });
  
  const [trainingConfig, setTrainingConfig] = useState({
    algorithm: 'PPO',
    episodes: 1000,
    learning_rate: 0.001,
    mesh: ''
  });

  const api = useApi();

  // Use polling hook for automatic updates when training is active
  const {
    data: pollingData,
    isPolling,
    startPolling,
    stopPolling,
    refresh: refreshStatus
  } = usePolling('getTrainingStatus', interval, {
    enabled: polling && trainingStatus.is_training,
    onSuccess: (data) => {
      if (data.success) {
        const newStatus = data.status;
        setTrainingStatus(prev => {
          // Call onChange callback if status changed
          if (onStatusChange && prev.status !== newStatus.status) {
            onStatusChange(newStatus, prev);
          }
          return newStatus;
        });
      }
    },
    onError: (error) => {
      console.error('Failed to poll training status:', error);
    }
  });

  // Manual status refresh
  const getStatus = useCallback(async () => {
    try {
      const data = await api.getTrainingStatus();
      if (data.success) {
        setTrainingStatus(prev => {
          const newStatus = data.status;
          if (onStatusChange && prev.status !== newStatus.status) {
            onStatusChange(newStatus, prev);
          }
          return newStatus;
        });
        return data.status;
      }
      return null;
    } catch (error) {
      console.error('Failed to get training status:', error);
      throw error;
    }
  }, [api, onStatusChange]);

  // Start training
  const startTraining = useCallback(async (config = trainingConfig) => {
    try {
      const response = await api.startTraining(config);
      
      if (response.success) {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          status: 'training'
        }));
        
        // Start polling if enabled
        if (polling) {
          startPolling();
        }
        
        return response;
      } else {
        throw new Error(response.error || 'Failed to start training');
      }
    } catch (error) {
      console.error('Failed to start training:', error);
      throw error;
    }
  }, [api, trainingConfig, polling, startPolling]);

  // Stop training
  const stopTraining = useCallback(async () => {
    try {
      const response = await api.stopTraining();
      
      if (response.success) {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: false,
          status: 'stopped'
        }));
        
        // Stop polling
        stopPolling();
        
        return response;
      } else {
        throw new Error(response.error || 'Failed to stop training');
      }
    } catch (error) {
      console.error('Failed to stop training:', error);
      throw error;
    }
  }, [api, stopPolling]);

  // Update training configuration
  const updateConfig = useCallback((updates) => {
    setTrainingConfig(prev => ({ ...prev, ...updates }));
  }, []);

  return {
    // Status data
    trainingStatus,
    trainingConfig,
    
    // Polling state
    isPolling,
    
    // Actions
    getStatus,
    startTraining,
    stopTraining,
    refreshStatus,
    updateConfig,
    
    // Polling controls
    startPolling,
    stopPolling
  };
};
