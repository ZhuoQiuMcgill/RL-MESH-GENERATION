import { api } from './client.js';

// Prediction service for handling all prediction-related API calls
export const predictionService = {
  // Start a new mesh prediction
  startPrediction: async (params) => {
    try {
      const response = await api.post('/api/predict/start', {
        modelId: params.modelId,
        inputDimensions: params.inputDimensions,
        quality: params.quality,
        iterations: params.iterations,
        learningRate: params.learningRate,
        additionalParams: params.additionalParams || {}
      });
      
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to start prediction. Please check your parameters and try again.'
      );
    }
  },

  // Get prediction status by ID
  getPredictionStatus: async (predictionId) => {
    try {
      const response = await api.get(`/api/predict/status/${predictionId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve prediction status.'
      );
    }
  },

  // Get prediction results by ID
  getPredictionResults: async (predictionId) => {
    try {
      const response = await api.get(`/api/predict/results/${predictionId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve prediction results.'
      );
    }
  },

  // Cancel a running prediction
  cancelPrediction: async (predictionId) => {
    try {
      const response = await api.post(`/api/predict/cancel/${predictionId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to cancel prediction.'
      );
    }
  },

  // Upload mesh file for prediction input
  uploadMeshFile: async (file, onProgress = null) => {
    try {
      const formData = new FormData();
      formData.append('meshFile', file);
      
      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        ...(onProgress && {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
          }
        })
      };
      
      const response = await api.post('/api/predict/upload', formData, config);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to upload mesh file. Please check the file format and try again.'
      );
    }
  },

  // Get available prediction models
  getAvailableModels: async () => {
    try {
      const response = await api.get('/api/predict/models');
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve available models.'
      );
    }
  },

  // Validate prediction parameters
  validateParams: (params) => {
    const errors = [];
    
    if (!params.modelId || typeof params.modelId !== 'string') {
      errors.push('Model ID is required');
    }
    
    if (!params.inputDimensions) {
      errors.push('Input dimensions are required');
    } else {
      const { width, height, depth } = params.inputDimensions;
      if (!width || width <= 0) errors.push('Width must be greater than 0');
      if (!height || height <= 0) errors.push('Height must be greater than 0');
      if (!depth || depth <= 0) errors.push('Depth must be greater than 0');
    }
    
    if (!params.quality || !['low', 'medium', 'high'].includes(params.quality)) {
      errors.push('Quality must be one of: low, medium, high');
    }
    
    if (!params.iterations || params.iterations <= 0) {
      errors.push('Iterations must be greater than 0');
    }
    
    if (!params.learningRate || params.learningRate <= 0 || params.learningRate > 1) {
      errors.push('Learning rate must be between 0 and 1');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  },

  // Export mesh results in various formats
  exportResults: async (predictionId, format = 'obj') => {
    try {
      const response = await api.get(`/api/predict/export/${predictionId}`, {
        params: { format },
        responseType: 'blob'
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `mesh-prediction-${predictionId}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return { success: true };
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        `Failed to export results in ${format} format.`
      );
    }
  }
};
