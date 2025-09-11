import { api } from './client.js';

// Training service for handling all training-related API calls
export const trainingService = {
  // Start a new training session
  startTraining: async (config) => {
    try {
      const response = await api.post('/api/train/start', {
        algorithm: config.algorithm,
        episodes: config.episodes,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        gamma: config.gamma,
        environment: config.environment,
        modelName: config.modelName,
        additionalParams: config.additionalParams || {}
      });
      
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to start training. Please check your configuration and try again.'
      );
    }
  },

  // Get training status and progress
  getTrainingStatus: async (trainingId) => {
    try {
      const response = await api.get(`/api/train/status/${trainingId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve training status.'
      );
    }
  },

  // Stop/cancel training session
  stopTraining: async (trainingId) => {
    try {
      const response = await api.post(`/api/train/stop/${trainingId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to stop training session.'
      );
    }
  },

  // Pause training session
  pauseTraining: async (trainingId) => {
    try {
      const response = await api.post(`/api/train/pause/${trainingId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to pause training session.'
      );
    }
  },

  // Resume training session
  resumeTraining: async (trainingId) => {
    try {
      const response = await api.post(`/api/train/resume/${trainingId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to resume training session.'
      );
    }
  },

  // Get training metrics and logs
  getTrainingMetrics: async (trainingId, limit = 100) => {
    try {
      const response = await api.get(`/api/train/metrics/${trainingId}`, {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve training metrics.'
      );
    }
  },

  // Get available training algorithms
  getAvailableAlgorithms: async () => {
    try {
      const response = await api.get('/api/train/algorithms');
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve available algorithms.'
      );
    }
  },

  // Get training environments
  getAvailableEnvironments: async () => {
    try {
      const response = await api.get('/api/train/environments');
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve available environments.'
      );
    }
  },

  // Validate training configuration
  validateConfig: (config) => {
    const errors = [];
    
    if (!config.algorithm || typeof config.algorithm !== 'string') {
      errors.push('Algorithm is required');
    }
    
    if (!config.episodes || config.episodes <= 0) {
      errors.push('Episodes must be greater than 0');
    }
    
    if (!config.batchSize || config.batchSize <= 0) {
      errors.push('Batch size must be greater than 0');
    }
    
    // Check if batch size is power of 2
    if (config.batchSize && (config.batchSize & (config.batchSize - 1)) !== 0) {
      errors.push('Batch size should be a power of 2 for optimal performance');
    }
    
    if (!config.learningRate || config.learningRate <= 0 || config.learningRate > 1) {
      errors.push('Learning rate must be between 0 and 1');
    }
    
    if (config.gamma && (config.gamma < 0 || config.gamma > 1)) {
      errors.push('Gamma (discount factor) must be between 0 and 1');
    }
    
    if (!config.environment || typeof config.environment !== 'string') {
      errors.push('Environment is required');
    }
    
    if (!config.modelName || typeof config.modelName !== 'string') {
      errors.push('Model name is required');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  },

  // Save trained model
  saveModel: async (trainingId, modelName, description = '') => {
    try {
      const response = await api.post(`/api/train/save-model/${trainingId}`, {
        modelName,
        description
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to save trained model.'
      );
    }
  },

  // Load training configuration from template
  loadConfigTemplate: async (templateName) => {
    try {
      const response = await api.get(`/api/train/templates/${templateName}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to load configuration template.'
      );
    }
  },

  // Get available configuration templates
  getConfigTemplates: async () => {
    try {
      const response = await api.get('/api/train/templates');
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve configuration templates.'
      );
    }
  },

  // Export training data and results
  exportTrainingData: async (trainingId, format = 'json') => {
    try {
      const response = await api.get(`/api/train/export/${trainingId}`, {
        params: { format },
        responseType: 'blob'
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `training-data-${trainingId}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return { success: true };
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        `Failed to export training data in ${format} format.`
      );
    }
  }
};
