import { api } from './client.js';

// History service for handling all history-related API calls
export const historyService = {
  // Get historical data with pagination and filters
  getHistory: async (options = {}) => {
    try {
      const {
        page = 1,
        limit = 20,
        type = null, // 'training', 'prediction', or null for all
        status = null, // 'completed', 'failed', 'running', or null for all
        dateRange = null, // { start: Date, end: Date }
        sortBy = 'timestamp',
        sortOrder = 'desc'
      } = options;

      const params = {
        page,
        limit,
        sortBy,
        sortOrder
      };

      if (type) params.type = type;
      if (status) params.status = status;
      if (dateRange) {
        params.startDate = dateRange.start.toISOString();
        params.endDate = dateRange.end.toISOString();
      }

      const response = await api.get('/api/history', { params });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve history data.'
      );
    }
  },

  // Get detailed information about a specific history item
  getHistoryItem: async (itemId) => {
    try {
      const response = await api.get(`/api/history/item/${itemId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve history item details.'
      );
    }
  },

  // Delete a history item
  deleteHistoryItem: async (itemId) => {
    try {
      const response = await api.delete(`/api/history/item/${itemId}`);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to delete history item.'
      );
    }
  },

  // Delete multiple history items
  deleteHistoryItems: async (itemIds) => {
    try {
      const response = await api.post('/api/history/delete-batch', {
        itemIds
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to delete selected history items.'
      );
    }
  },

  // Get training history statistics
  getTrainingStats: async (timeRange = '30d') => {
    try {
      const response = await api.get('/api/history/training-stats', {
        params: { timeRange }
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve training statistics.'
      );
    }
  },

  // Get prediction history statistics
  getPredictionStats: async (timeRange = '30d') => {
    try {
      const response = await api.get('/api/history/prediction-stats', {
        params: { timeRange }
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve prediction statistics.'
      );
    }
  },

  // Search history items
  searchHistory: async (query, filters = {}) => {
    try {
      const params = {
        query,
        ...filters
      };

      const response = await api.get('/api/history/search', { params });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to search history items.'
      );
    }
  },

  // Export history data
  exportHistory: async (options = {}, format = 'csv') => {
    try {
      const {
        type = null,
        status = null,
        dateRange = null
      } = options;

      const params = { format };
      if (type) params.type = type;
      if (status) params.status = status;
      if (dateRange) {
        params.startDate = dateRange.start.toISOString();
        params.endDate = dateRange.end.toISOString();
      }

      const response = await api.get('/api/history/export', {
        params,
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      const timestamp = new Date().toISOString().split('T')[0];
      link.setAttribute('download', `history-export-${timestamp}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      return { success: true };
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        `Failed to export history data in ${format} format.`
      );
    }
  },

  // Get history summary for dashboard
  getHistorySummary: async () => {
    try {
      const response = await api.get('/api/history/summary');
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve history summary.'
      );
    }
  },

  // Archive old history items
  archiveHistoryItems: async (olderThanDays = 90) => {
    try {
      const response = await api.post('/api/history/archive', {
        olderThanDays
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to archive history items.'
      );
    }
  },

  // Get model performance comparison from history
  getModelComparison: async (modelIds, metric = 'accuracy') => {
    try {
      const response = await api.post('/api/history/model-comparison', {
        modelIds,
        metric
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve model comparison data.'
      );
    }
  },

  // Get performance trends over time
  getPerformanceTrends: async (timeRange = '30d', groupBy = 'day') => {
    try {
      const response = await api.get('/api/history/trends', {
        params: { timeRange, groupBy }
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve performance trends.'
      );
    }
  },

  // Add notes to a history item
  addNotes: async (itemId, notes) => {
    try {
      const response = await api.post(`/api/history/item/${itemId}/notes`, {
        notes
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to add notes to history item.'
      );
    }
  },

  // Get logs for a specific history item
  getItemLogs: async (itemId, logType = 'all') => {
    try {
      const response = await api.get(`/api/history/item/${itemId}/logs`, {
        params: { logType }
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || 
        'Failed to retrieve item logs.'
      );
    }
  }
};
