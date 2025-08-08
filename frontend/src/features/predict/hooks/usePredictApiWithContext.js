import { useMemo } from 'react';
import { createPredictApiWithDispatch } from '../../../shared/api/predict';
import { usePredictSession } from '../contexts/PredictSessionContext';

/**
 * 使用Context集成的Predict API Hook
 * 自动处理loading状态和错误，集成日志记录
 * @returns {Object} 包含所有API方法的对象
 */
const usePredictApiWithContext = () => {
  const { actions } = usePredictSession();

  const api = useMemo(() => {
    return createPredictApiWithDispatch(
      // dispatch函数 - 直接使用actions中的内部dispatch
      (action) => {
        switch (action.type) {
          case 'SET_LOADING':
            actions.setLoading(action.payload);
            break;
          case 'API_ERROR':
            actions.apiError(action.payload);
            break;
          default:
            // 其他action可以直接使用对应的action creator
            break;
        }
      },
      // addLog函数 - 使用actions中的addLog
      actions.addLog
    );
  }, [actions]);

  return api;
};

export default usePredictApiWithContext;
