import { useCallback } from 'react';
import { usePredictSession } from '../contexts/PredictSessionContext';

// Log type constants
export const LogType = {
  SYSTEM: 'system',
  USER_ACTION: 'user_action',
  API_SUCCESS: 'api_success',
  API_ERROR: 'api_error',
  PREDICTION: 'prediction',
  MESH: 'mesh',
  ERROR: 'error'
};

/**
 * Operation log management custom Hook
 * Provides simplified addLog(type, msg) and clearLog() methods
 */
export const useOperationLog = () => {
  const { logs, actions } = usePredictSession();

  // Main method: add log - addLog(type, msg)
  const addLog = useCallback((type, msg) => {
    const logEntry = {
      type,
      message: msg,
      level: type === LogType.API_ERROR || type === LogType.ERROR ? 'error' : 'info'
    };
    
    actions.addLog(logEntry);
  }, [actions]);

  // Main method: clear logs - clearLog()
  const clearLog = useCallback(() => {
    actions.clearLogs();
  }, [actions]);

  return {
    // Core methods
    addLog,
    clearLog,
    
    // Log data
    logs,
    
    // Constants
    LogType
  };
};

export default useOperationLog;
