import apiClient from './client';

/**
 * 统一错误处理函数
 * @param {Error} error - API 错误对象
 * @param {Function} dispatch - Context dispatch函数
 * @param {Function} logFunction - 日志记录函数
 * @param {string} operation - 操作名称
 * @returns {Promise<never>} - 抛出格式化后的错误
 */
const handleApiError = (error, dispatch = null, logFunction = null, operation = '') => {
  let message = '网络请求失败';
  
  if (error.response) {
    // 服务器响应了错误状态
    const { status, data } = error.response;
    message = data?.message || data?.error || `服务器错误 (${status})`;
  } else if (error.request) {
    // 请求已发出但没有收到响应
    message = '网络连接失败，请检查网络连接';
  } else {
    // 请求配置错误
    message = error.message || '请求配置错误';
  }
  
  const formattedError = {
    message,
    originalError: error,
    status: error.response?.status,
    operation
  };
  
  // 使用dispatch触发API_ERROR
  if (dispatch) {
    dispatch({ type: 'API_ERROR', payload: formattedError });
  }
  
  // 记录API错误日志
  if (logFunction && operation) {
    logFunction('api_error', `${operation}失败: ${message}`);
  }
  
  return Promise.reject(formattedError);
};

/**
 * 创建带dispatch支持的API方法
 * @param {Function} dispatch - Context dispatch函数
 * @param {Function} addLog - 日志记录函数
 * @returns {Object} API 方法集合
 */
const createPredictApiWithDispatch = (dispatch = null, addLog = null) => ({
  /**
   * 获取组件列表
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 组件列表数据
   */
  listComponents: async (params = {}) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.get('/predict/components', { params });
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '获取组件列表');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 创建新的预测会话
   * @param {Object} sessionData - 会话创建数据
   * @returns {Promise<Object>} 创建的会话信息
   */
  createSession: async (sessionData) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.post('/predict/sessions', sessionData);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '创建预测会话');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 执行下一步预测
   * @param {string} sessionId - 会话ID
   * @param {Object} stepData - 步骤数据
   * @returns {Promise<Object>} 下一步预测结果
   */
  nextStep: async (sessionId, stepData = {}) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/next`, stepData);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '执行下一步预测');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 执行上一步预测
   * @param {string} sessionId - 会话ID
   * @param {Object} stepData - 步骤数据
   * @returns {Promise<Object>} 上一步预测结果
   */
  prevStep: async (sessionId, stepData = {}) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/prev`, stepData);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '执行上一步预测');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 处理所有步骤
   * @param {string} sessionId - 会话ID
   * @param {Object} processData - 处理数据
   * @returns {Promise<Object>} 处理结果
   */
  processAll: async (sessionId, processData = {}) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/process-all`, processData);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '处理所有步骤');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 重置预测会话
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 重置结果
   */
  resetSession: async (sessionId) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/reset`);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '重置预测会话');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 删除预测会话
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 删除结果
   */
  deleteSession: async (sessionId) => {
    dispatch && dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await apiClient.delete(`/predict/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '删除预测会话');
    } finally {
      dispatch && dispatch({ type: 'SET_LOADING', payload: false });
    }
  },

  /**
   * 获取会话状态
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 会话状态信息
   */
  getStatus: async (sessionId) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/status`);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '获取会话状态');
    }
  },

  /**
   * 获取质量评估
   * @param {string} sessionId - 会话ID
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 质量评估数据
   */
  getQuality: async (sessionId, params = {}) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/quality`, { params });
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '获取质量评估');
    }
  },

  /**
   * 获取参考点
   * @param {string} sessionId - 会话ID
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 参考点数据
   */
  getReferencePoint: async (sessionId, params = {}) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/reference-point`, { params });
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '获取参考点');
    }
  },

  /**
   * 预览参考点
   * @param {string} sessionId - 会话ID
   * @param {Object} previewData - 预览数据
   * @returns {Promise<Object>} 参考点预览数据
   */
  previewReferencePoint: async (sessionId, previewData = {}) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/reference-point/preview`, previewData);
      return response.data;
    } catch (error) {
      return handleApiError(error, dispatch, addLog, '预览参考点');
    }
  },
});

/**
 * 旧版本的Predict API 方法集合(向后兼容)
 */
const predictApi = {
  /**
   * 获取组件列表
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 组件列表数据
   */
  listComponents: async (params = {}) => {
    try {
      const response = await apiClient.get('/predict/components', { params });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 创建新的预测会话
   * @param {Object} sessionData - 会话创建数据
   * @returns {Promise<Object>} 创建的会话信息
   */
  createSession: async (sessionData) => {
    try {
      const response = await apiClient.post('/predict/sessions', sessionData);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 执行下一步预测
   * @param {string} sessionId - 会话ID
   * @param {Object} stepData - 步骤数据
   * @returns {Promise<Object>} 下一步预测结果
   */
  nextStep: async (sessionId, stepData = {}) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/next`, stepData);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 执行上一步预测
   * @param {string} sessionId - 会话ID
   * @param {Object} stepData - 步骤数据
   * @returns {Promise<Object>} 上一步预测结果
   */
  prevStep: async (sessionId, stepData = {}) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/prev`, stepData);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 处理所有步骤
   * @param {string} sessionId - 会话ID
   * @param {Object} processData - 处理数据
   * @returns {Promise<Object>} 处理结果
   */
  processAll: async (sessionId, processData = {}) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/process-all`, processData);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 重置预测会话
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 重置结果
   */
  resetSession: async (sessionId) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/reset`);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 删除预测会话
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 删除结果
   */
  deleteSession: async (sessionId) => {
    try {
      const response = await apiClient.delete(`/predict/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 获取会话状态
   * @param {string} sessionId - 会话ID
   * @returns {Promise<Object>} 会话状态信息
   */
  getStatus: async (sessionId) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/status`);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 获取质量评估
   * @param {string} sessionId - 会话ID
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 质量评估数据
   */
  getQuality: async (sessionId, params = {}) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/quality`, { params });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 获取参考点
   * @param {string} sessionId - 会话ID
   * @param {Object} params - 查询参数
   * @returns {Promise<Object>} 参考点数据
   */
  getReferencePoint: async (sessionId, params = {}) => {
    try {
      const response = await apiClient.get(`/predict/sessions/${sessionId}/reference-point`, { params });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  /**
   * 预览参考点
   * @param {string} sessionId - 会话ID
   * @param {Object} previewData - 预览数据
   * @returns {Promise<Object>} 参考点预览数据
   */
  previewReferencePoint: async (sessionId, previewData = {}) => {
    try {
      const response = await apiClient.post(`/predict/sessions/${sessionId}/reference-point/preview`, previewData);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },
};

// 导出所有方法
export const {
  listComponents,
  createSession,
  nextStep,
  prevStep,
  processAll,
  resetSession,
  deleteSession,
  getStatus,
  getQuality,
  getReferencePoint,
  previewReferencePoint,
} = predictApi;

// 导出创建API的工厂函数
export { createPredictApiWithDispatch };

// 默认导出整个 API 对象
export default predictApi;
