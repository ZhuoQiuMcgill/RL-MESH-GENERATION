import React, { createContext, useContext, useReducer, useCallback } from 'react';

// Status definition
export const PredictSessionStatus = {
  IDLE: 'idle',
  CONFIGURING: 'configuring', 
  INITIALIZING: 'initializing',
  RUNNING: 'running',
  PAUSED: 'paused',
  COMPLETED: 'completed',
  ERROR: 'error'
};

// Action Types
export const PredictSessionActions = {
  CREATE_SESSION: 'CREATE_SESSION',
  CONFIGURE_SESSION: 'CONFIGURE_SESSION',
  CONFIG_UPDATED: 'CONFIG_UPDATED',
  START_PREDICTION: 'START_PREDICTION',
  PAUSE_PREDICTION: 'PAUSE_PREDICTION',
  RESUME_PREDICTION: 'RESUME_PREDICTION',
  STOP_PREDICTION: 'STOP_PREDICTION',
  NEXT_STEP: 'NEXT_STEP',
  UPDATE_PROGRESS: 'UPDATE_PROGRESS',
  SET_REF_POINT: 'SET_REF_POINT',
  ADD_LOG: 'ADD_LOG',
  CLEAR_LOGS: 'CLEAR_LOGS',
  SET_ERROR: 'SET_ERROR',
  API_ERROR: 'API_ERROR',
  SET_LOADING: 'SET_LOADING',
  CLEAR_ERROR: 'CLEAR_ERROR',
  RESET_SESSION: 'RESET_SESSION'
};

// Initial state
const initialState = {
  sessionId: null,
  status: PredictSessionStatus.IDLE,
  configuration: {
    geometry: {
      type: 'rectangle',
      width: 10,
      height: 10,
      complexity: 1
    },
    mesh: {
      maxElementSize: 0.5,
      minElementSize: 0.1,
      quality: 0.8
    },
    algorithm: {
      method: 'rl_ddpg',
      maxSteps: 1000,
      learningRate: 0.001
    }
  },
  refPoint: null,
  currentStep: 0,
  totalSteps: 0,
  progress: 0,
  meshData: null,
  logs: [],
  error: null,
  loading: false,
  startTime: null,
  endTime: null
};

// Reducer
const predictSessionReducer = (state, action) => {
  switch (action.type) {
    case PredictSessionActions.CREATE_SESSION:
      return {
        ...state,
        sessionId: action.payload.sessionId,
        status: PredictSessionStatus.CONFIGURING,
        startTime: new Date().toISOString(),
        error: null
      };

    case PredictSessionActions.CONFIGURE_SESSION:
      return {
        ...state,
        configuration: {
          ...state.configuration,
          ...action.payload.configuration
        }
      };

    case PredictSessionActions.CONFIG_UPDATED:
      return {
        ...state,
        configuration: {
          ...state.configuration,
          ...action.payload
        },
        status: action.payload.isValid ? PredictSessionStatus.CONFIGURING : state.status,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Configuration updated',
            data: action.payload
          }
        ]
      };

    case PredictSessionActions.START_PREDICTION:
      return {
        ...state,
        status: PredictSessionStatus.INITIALIZING,
        currentStep: 0,
        progress: 0,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Prediction started',
            data: action.payload
          }
        ]
      };

    case PredictSessionActions.PAUSE_PREDICTION:
      return {
        ...state,
        status: PredictSessionStatus.PAUSED,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Prediction paused'
          }
        ]
      };

    case PredictSessionActions.RESUME_PREDICTION:
      return {
        ...state,
        status: PredictSessionStatus.RUNNING,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Prediction resumed'
          }
        ]
      };

    case PredictSessionActions.STOP_PREDICTION:
      return {
        ...state,
        status: PredictSessionStatus.IDLE,
        endTime: new Date().toISOString(),
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'warning',
            message: 'Prediction stopped by user'
          }
        ]
      };

    case PredictSessionActions.NEXT_STEP:
      const newStep = state.currentStep + 1;
      const newProgress = state.totalSteps > 0 ? (newStep / state.totalSteps) * 100 : 0;
      const isCompleted = newStep >= state.totalSteps && state.totalSteps > 0;
      
      return {
        ...state,
        currentStep: newStep,
        progress: newProgress,
        status: isCompleted ? PredictSessionStatus.COMPLETED : PredictSessionStatus.RUNNING,
        meshData: action.payload.meshData,
        endTime: isCompleted ? new Date().toISOString() : state.endTime,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Step ${newStep} completed`,
            data: action.payload
          }
        ]
      };

    case PredictSessionActions.UPDATE_PROGRESS:
      return {
        ...state,
        ...action.payload,
        status: state.status === PredictSessionStatus.INITIALIZING 
          ? PredictSessionStatus.RUNNING 
          : state.status
      };

    case PredictSessionActions.SET_REF_POINT:
      return {
        ...state,
        refPoint: action.payload.refPoint,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Reference point updated',
            data: action.payload.refPoint
          }
        ]
      };

    case PredictSessionActions.ADD_LOG:
      const newLog = {
        id: Date.now() + Math.random(),
        timestamp: new Date().toISOString(),
        ...action.payload
      };
      
      // Maintain maximum 200 logs
      const updatedLogs = [...state.logs, newLog];
      const logsToKeep = updatedLogs.length > 200 
        ? updatedLogs.slice(-200) 
        : updatedLogs;
      
      return {
        ...state,
        logs: logsToKeep
      };

    case PredictSessionActions.CLEAR_LOGS:
      return {
        ...state,
        logs: []
      };

    case PredictSessionActions.SET_ERROR:
      return {
        ...state,
        status: PredictSessionStatus.ERROR,
        error: action.payload.error,
        loading: false,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'error',
            message: action.payload.error.message || 'An error occurred',
            data: action.payload.error
          }
        ]
      };

    case PredictSessionActions.API_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false,
        logs: [
          ...state.logs,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            level: 'error',
            message: `API Error: ${action.payload.message || 'Unknown error'}`,
            data: action.payload
          }
        ]
      };

    case PredictSessionActions.SET_LOADING:
      return {
        ...state,
        loading: action.payload
      };

    case PredictSessionActions.CLEAR_ERROR:
      return {
        ...state,
        error: null
      };

    case PredictSessionActions.RESET_SESSION:
      return {
        ...initialState,
        configuration: state.configuration // Keep configuration
      };

    default:
      return state;
  }
};

// Context
const PredictSessionContext = createContext();

// Provider Component
export const PredictSessionProvider = ({ children }) => {
  const [state, dispatch] = useReducer(predictSessionReducer, initialState);

  // Action Creators
  const actions = {
    createSession: useCallback((sessionId) => {
      dispatch({
        type: PredictSessionActions.CREATE_SESSION,
        payload: { sessionId }
      });
    }, []),

    configureSession: useCallback((configuration) => {
      dispatch({
        type: PredictSessionActions.CONFIGURE_SESSION,
        payload: { configuration }
      });
    }, []),

    configUpdate: useCallback((config) => {
      dispatch({
        type: PredictSessionActions.CONFIG_UPDATED,
        payload: config
      });
    }, []),

    startPrediction: useCallback((params = {}) => {
      dispatch({
        type: PredictSessionActions.START_PREDICTION,
        payload: params
      });
    }, []),

    pausePrediction: useCallback(() => {
      dispatch({ type: PredictSessionActions.PAUSE_PREDICTION });
    }, []),

    resumePrediction: useCallback(() => {
      dispatch({ type: PredictSessionActions.RESUME_PREDICTION });
    }, []),

    stopPrediction: useCallback(() => {
      dispatch({ type: PredictSessionActions.STOP_PREDICTION });
    }, []),

    nextStep: useCallback((stepData) => {
      dispatch({
        type: PredictSessionActions.NEXT_STEP,
        payload: stepData
      });
    }, []),

    updateProgress: useCallback((progressData) => {
      dispatch({
        type: PredictSessionActions.UPDATE_PROGRESS,
        payload: progressData
      });
    }, []),

    setRefPoint: useCallback((refPoint) => {
      dispatch({
        type: PredictSessionActions.SET_REF_POINT,
        payload: { refPoint }
      });
    }, []),

    addLog: useCallback((log) => {
      dispatch({
        type: PredictSessionActions.ADD_LOG,
        payload: log
      });
    }, []),

    clearLogs: useCallback(() => {
      dispatch({ type: PredictSessionActions.CLEAR_LOGS });
    }, []),

    setError: useCallback((error) => {
      dispatch({
        type: PredictSessionActions.SET_ERROR,
        payload: { error }
      });
    }, []),

    apiError: useCallback((error) => {
      dispatch({
        type: PredictSessionActions.API_ERROR,
        payload: error
      });
    }, []),

    setLoading: useCallback((loading) => {
      dispatch({
        type: PredictSessionActions.SET_LOADING,
        payload: loading
      });
    }, []),

    clearError: useCallback(() => {
      dispatch({ type: PredictSessionActions.CLEAR_ERROR });
    }, []),

    resetSession: useCallback(() => {
      dispatch({ type: PredictSessionActions.RESET_SESSION });
    }, [])
  };

  const value = {
    ...state,
    actions
  };

  return (
    <PredictSessionContext.Provider value={value}>
      {children}
    </PredictSessionContext.Provider>
  );
};

// Custom Hook
export const usePredictSession = () => {
  const context = useContext(PredictSessionContext);
  if (!context) {
    throw new Error('usePredictSession must be used within a PredictSessionProvider');
  }
  return context;
};

export default PredictSessionContext;
