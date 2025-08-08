import { useState, useCallback, useRef } from 'react';
import { useApi } from '../context/ApiProvider';

/**
 * Custom hook for mesh generation state machine
 * Manages the complete lifecycle of mesh generation sessions
 */
export const useMeshGenerator = () => {
  const [state, setState] = useState({
    // Configuration state
    selectedMesh: '',
    meshInfo: null,
    selectedPredictor: '',
    selectedRefSelector: '',
    selectedQualityMethod: '',
    predictorConfig: { n: 2, g: 3, beta: 6, modelPath: '' },
    refSelectorConfig: { n: 2 },
    
    // Session state
    sessionId: null,
    currentStep: 0,
    sessionData: null,
    actionInfo: null,
    referencePointInfo: null,
    elementQuality: null,
    
    // UI state
    isLoading: false,
    error: null,
    log: []
  });

  const api = useApi();
  const canvasRef = useRef(null);

  const addLogEntry = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setState(prev => ({
      ...prev,
      log: [...prev.log, { message: `[${timestamp}] ${message}`, type }]
    }));
  }, []);

  const setLoading = useCallback((isLoading) => {
    setState(prev => ({ ...prev, isLoading }));
  }, []);

  const setError = useCallback((error) => {
    setState(prev => ({ ...prev, error }));
  }, []);

  // Configuration methods
  const updateMesh = useCallback(async (meshName) => {
    if (!meshName) {
      setState(prev => ({
        ...prev,
        selectedMesh: '',
        meshInfo: null
      }));
      if (canvasRef.current) {
        canvasRef.current.clearCanvas();
      }
      return;
    }

    try {
      setLoading(true);
      const info = await api.getMeshInfo(meshName);
      const boundaryData = await api.getMeshBoundary(meshName);
      
      setState(prev => ({
        ...prev,
        selectedMesh: meshName,
        meshInfo: info
      }));

      if (boundaryData.success && canvasRef.current) {
        canvasRef.current.renderBoundaryPreview(boundaryData.boundary_vertices, meshName);
      }
      
      addLogEntry(`Selected mesh: ${meshName}`, 'info');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load mesh: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [api, addLogEntry, setLoading, setError]);

  const updatePredictor = useCallback((predictorType) => {
    setState(prev => ({
      ...prev,
      selectedPredictor: predictorType
    }));
    addLogEntry(`Selected predictor: ${predictorType}`, 'info');
  }, [addLogEntry]);

  const updatePredictorConfig = useCallback((config) => {
    setState(prev => ({
      ...prev,
      predictorConfig: { ...prev.predictorConfig, ...config }
    }));
  }, []);

  const updateRefSelector = useCallback((selectorType) => {
    setState(prev => ({
      ...prev,
      selectedRefSelector: selectorType
    }));
    addLogEntry(`Selected reference selector: ${selectorType}`, 'info');
  }, [addLogEntry]);

  const updateRefSelectorConfig = useCallback((config) => {
    setState(prev => ({
      ...prev,
      refSelectorConfig: { ...prev.refSelectorConfig, ...config }
    }));
  }, []);

  const updateQualityMethod = useCallback((method) => {
    setState(prev => ({
      ...prev,
      selectedQualityMethod: method
    }));
    addLogEntry(`Selected quality method: ${method}`, 'info');
  }, [addLogEntry]);

  // Session management methods
  const createSession = useCallback(async () => {
    try {
      setLoading(true);
      
      const sessionConfig = {
        initial_mesh: state.selectedMesh,
        predictor_type: state.selectedPredictor,
        predictor_config: state.predictorConfig,
        reference_selector: state.selectedRefSelector,
        reference_selector_config: state.refSelectorConfig,
        quality_method: state.selectedQualityMethod
      };
      
      // Note: This would need to be implemented in the predict API
      // const response = await api.createPredictionSession(sessionConfig);
      
      setState(prev => ({
        ...prev,
        sessionId: 'mock-session-id', // response.session_id
        sessionData: { boundarySize: 10, generatedElements: 0 } // response
      }));
      
      addLogEntry('Session created successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to create session: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [state, api, addLogEntry, setLoading, setError]);

  const executeNextStep = useCallback(async () => {
    if (!state.sessionId) return;
    
    try {
      setLoading(true);
      // Mock implementation - would call predict API
      
      setState(prev => ({
        ...prev,
        currentStep: prev.currentStep + 1,
        actionInfo: {
          type: 'type0-left',
          referenceVertex: prev.currentStep,
          status: 'valid',
          newCoords: `(${Math.random().toFixed(3)}, ${Math.random().toFixed(3)})`
        }
      }));
      
      addLogEntry(`Executed step ${state.currentStep + 1}`, 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to execute step: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [state.sessionId, state.currentStep, api, addLogEntry, setLoading, setError]);

  const executePreviousStep = useCallback(async () => {
    if (!state.sessionId || state.currentStep <= 0) return;
    
    try {
      setLoading(true);
      
      setState(prev => ({
        ...prev,
        currentStep: Math.max(0, prev.currentStep - 1)
      }));
      
      addLogEntry(`Reverted to step ${state.currentStep - 1}`, 'info');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to revert step: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [state.sessionId, state.currentStep, addLogEntry, setLoading, setError]);

  const resetSession = useCallback(async () => {
    if (!state.sessionId) return;
    
    try {
      setLoading(true);
      
      setState(prev => ({
        ...prev,
        currentStep: 0,
        actionInfo: null,
        referencePointInfo: null,
        elementQuality: null
      }));
      
      addLogEntry('Session reset successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to reset session: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [state.sessionId, addLogEntry, setLoading, setError]);

  const deleteSession = useCallback(async () => {
    if (!state.sessionId) return;
    
    try {
      setLoading(true);
      
      setState(prev => ({
        ...prev,
        sessionId: null,
        sessionData: null,
        currentStep: 0,
        actionInfo: null,
        referencePointInfo: null,
        elementQuality: null
      }));
      
      addLogEntry('Session deleted successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to delete session: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [state.sessionId, addLogEntry, setLoading, setError]);

  const clearLog = useCallback(() => {
    setState(prev => ({
      ...prev,
      log: []
    }));
    addLogEntry('Log cleared', 'info');
  }, [addLogEntry]);

  // Validation helpers
  const isConfigurationValid = useCallback(() => {
    return state.selectedMesh && 
           state.selectedPredictor && 
           state.selectedRefSelector && 
           state.selectedQualityMethod && 
           state.predictorConfig.modelPath;
  }, [state]);

  return {
    // State
    ...state,
    canvasRef,
    
    // Configuration methods
    updateMesh,
    updatePredictor,
    updatePredictorConfig,
    updateRefSelector,
    updateRefSelectorConfig,
    updateQualityMethod,
    
    // Session management
    createSession,
    executeNextStep,
    executePreviousStep,
    resetSession,
    deleteSession,
    
    // Utilities
    clearLog,
    isConfigurationValid
  };
};

export default useMeshGenerator;
