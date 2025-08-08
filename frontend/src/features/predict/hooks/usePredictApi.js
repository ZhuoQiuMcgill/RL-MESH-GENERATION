import { useState, useCallback, useRef } from 'react';
import { usePredictSession } from '../contexts/PredictSessionContext';
import { useOperationLog, LogType } from './useOperationLog';

// API 基础配置
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const PREDICTION_ENDPOINTS = {
  createSession: '/api/predict/session/create',
  getSession: '/api/predict/session',
  startPrediction: '/api/predict/start',
  pausePrediction: '/api/predict/pause',
  resumePrediction: '/api/predict/resume',
  stopPrediction: '/api/predict/stop',
  getStep: '/api/predict/step',
  getStatus: '/api/predict/status',
  getMeshData: '/api/predict/mesh'
};

/**
 * 预测API相关的自定义Hook
 * 提供与后端预测服务交互的所有功能
 */
export const usePredictApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { sessionId, configuration, actions } = usePredictSession();
  const { addLog } = useOperationLog();
  const abortControllerRef = useRef(null);

  // 通用API请求方法
  const apiRequest = useCallback(async (endpoint, options = {}) => {
    const url = `${API_BASE_URL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...(sessionId && { 'X-Session-Id': sessionId })
      },
      ...options
    };

    try {
      const response = await fetch(url, defaultOptions);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('请求被取消');
      }
      throw err;
    }
  }, [sessionId]);

  // 创建预测会话
  const createPredictionSession = useCallback(async (config = configuration) => {
    setLoading(true);
    setError(null);
    addLog(LogType.SYSTEM, '开始创建预测会话...');

    try {
      const response = await apiRequest(PREDICTION_ENDPOINTS.createSession, {
        method: 'POST',
        body: JSON.stringify({
          configuration: config,
          timestamp: new Date().toISOString()
        })
      });

      actions.createSession(response.sessionId);
      actions.configureSession(config);
      addLog(LogType.API_SUCCESS, `预测会话创建成功，ID: ${response.sessionId}`);

      return response;
    } catch (err) {
      setError(err.message);
      actions.setError(err);
      addLog(LogType.API_ERROR, `创建预测会话失败: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [configuration, apiRequest, actions, addLog]);

  // 开始预测
  const startPrediction = useCallback(async (params = {}) => {
    if (!sessionId) {
      throw new Error('请先创建预测会话');
    }

    setLoading(true);
    setError(null);
    addLog(LogType.USER_ACTION, '用户开始预测任务');

    // 创建新的AbortController用于取消请求
    abortControllerRef.current = new AbortController();

    try {
      const response = await apiRequest(PREDICTION_ENDPOINTS.startPrediction, {
        method: 'POST',
        body: JSON.stringify({
          sessionId,
          parameters: params,
          configuration
        }),
        signal: abortControllerRef.current.signal
      });

      actions.startPrediction(params);
      actions.updateProgress({
        totalSteps: response.totalSteps || 1000,
        currentStep: 0
      });
      addLog(LogType.API_SUCCESS, `预测任务开始，预计步数: ${response.totalSteps || 1000}`);

      return response;
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
        actions.setError(err);
        addLog(LogType.API_ERROR, `开始预测失败: ${err.message}`);
      } else {
        addLog(LogType.SYSTEM, '预测任务被用户取消');
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, [sessionId, configuration, apiRequest, actions, addLog]);

  // 暂停预测
  const pausePrediction = useCallback(async () => {
    if (!sessionId) return;

    try {
      await apiRequest(PREDICTION_ENDPOINTS.pausePrediction, {
        method: 'POST',
        body: JSON.stringify({ sessionId })
      });
      
      actions.pausePrediction();
    } catch (err) {
      setError(err.message);
      actions.setError(err);
      throw err;
    }
  }, [sessionId, apiRequest, actions]);

  // 恢复预测
  const resumePrediction = useCallback(async () => {
    if (!sessionId) return;

    try {
      await apiRequest(PREDICTION_ENDPOINTS.resumePrediction, {
        method: 'POST',
        body: JSON.stringify({ sessionId })
      });
      
      actions.resumePrediction();
    } catch (err) {
      setError(err.message);
      actions.setError(err);
      throw err;
    }
  }, [sessionId, apiRequest, actions]);

  // 停止预测
  const stopPrediction = useCallback(async () => {
    if (!sessionId) return;

    // 取消正在进行的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    try {
      await apiRequest(PREDICTION_ENDPOINTS.stopPrediction, {
        method: 'POST',
        body: JSON.stringify({ sessionId })
      });
      
      actions.stopPrediction();
    } catch (err) {
      setError(err.message);
      actions.setError(err);
      throw err;
    }
  }, [sessionId, apiRequest, actions]);

  // 获取当前步骤数据
  const fetchStepData = useCallback(async (stepNumber) => {
    if (!sessionId) return null;

    try {
      const response = await apiRequest(
        `${PREDICTION_ENDPOINTS.getStep}/${stepNumber}?sessionId=${sessionId}`
      );
      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, [sessionId, apiRequest]);

  // 获取预测状态
  const fetchPredictionStatus = useCallback(async () => {
    if (!sessionId) return null;

    try {
      const response = await apiRequest(
        `${PREDICTION_ENDPOINTS.getStatus}?sessionId=${sessionId}`
      );
      
      // 更新本地状态
      actions.updateProgress({
        currentStep: response.currentStep,
        totalSteps: response.totalSteps,
        progress: response.progress,
        status: response.status
      });

      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, [sessionId, apiRequest, actions]);

  // 获取网格数据
  const fetchMeshData = useCallback(async (stepNumber = null) => {
    if (!sessionId) return null;

    try {
      const url = stepNumber 
        ? `${PREDICTION_ENDPOINTS.getMeshData}?sessionId=${sessionId}&step=${stepNumber}`
        : `${PREDICTION_ENDPOINTS.getMeshData}?sessionId=${sessionId}`;
        
      const response = await apiRequest(url);
      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, [sessionId, apiRequest]);

  // 轮询状态更新
  const startStatusPolling = useCallback((interval = 1000) => {
    const pollInterval = setInterval(async () => {
      try {
        await fetchPredictionStatus();
      } catch (err) {
        console.error('状态轮询失败:', err);
        clearInterval(pollInterval);
      }
    }, interval);

    return () => clearInterval(pollInterval);
  }, [fetchPredictionStatus]);

  // 清理方法
  const cleanup = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setError(null);
    setLoading(false);
  }, []);

  return {
    // 状态
    loading,
    error,
    
    // API方法
    createPredictionSession,
    startPrediction,
    pausePrediction,
    resumePrediction,
    stopPrediction,
    fetchStepData,
    fetchPredictionStatus,
    fetchMeshData,
    startStatusPolling,
    
    // 工具方法
    cleanup,
    
    // 内部方法（可选暴露）
    apiRequest
  };
};

export default usePredictApi;
