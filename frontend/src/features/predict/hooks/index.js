/**
 * Predict Feature Hooks 索引文件
 * 统一导出所有预测相关的自定义Hook
 */

export { usePredictApi, default as usePredictApiDefault } from './usePredictApi';
export { default as usePredictApiWithContext } from './usePredictApiWithContext';
export { 
  useOperationLog, 
  LogLevel, 
  LogType,
  default as useOperationLogDefault 
} from './useOperationLog';
export { 
  useCanvasRenderer,
  RenderMode,
  ViewMode,
  CameraControl,
  default as useCanvasRendererDefault 
} from './useCanvasRenderer';

// 导出常用组合Hook
export const usePredictFeature = () => {
  // 可以在这里组合多个Hook，提供更高层次的抽象
  // 例如：同时使用API、日志和渲染功能的组合Hook
};
