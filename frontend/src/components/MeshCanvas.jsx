import React, { useRef, forwardRef, useImperativeHandle, useCallback, useMemo } from 'react';
import useCanvasRenderer from '../hooks/useCanvasRenderer';

/**
 * MeshCanvas 组件 - 2D 网格渲染画布
 * 
 * 提供网格可视化功能，包括边界预览、完整场景渲染和交互支持
 * 使用 forwardRef 暴露命令式 API 供父组件使用
 * 
 * @param {Object} props 组件属性
 * @param {string} props.className - 额外的 CSS 类名
 * @param {Object} props.style - 内联样式
 * @param {Function} props.onCanvasClick - 画布点击回调函数
 * @param {Object} props...canvasProps - 其他传递给 canvas 元素的属性
 * @param {React.RefObject} ref - 组件引用
 * @returns {JSX.Element} Canvas 元素
 */
const MeshCanvas = forwardRef((props, ref) => {
  const { 
    className = '', 
    style = {}, 
    onCanvasClick = null,
    ...canvasProps 
  } = props;

  const canvasRef = useRef(null);
  const canvasRenderer = useCanvasRenderer(canvasRef);

  // 处理画布点击事件
  const handleCanvasClick = useCallback((event) => {
    if (!onCanvasClick || !canvasRef.current) {
      return;
    }

    try {
      // 获取画布边界矩形
      const rect = canvasRef.current.getBoundingClientRect();
      
      // 计算相对于画布的坐标
      const screenX = event.clientX - rect.left;
      const screenY = event.clientY - rect.top;

      // 转换为世界坐标
      const worldCoords = canvasRenderer.screenToWorld(screenX, screenY);
      
      // 只有当坐标转换有效时才调用回调
      const transform = canvasRenderer.getCurrentTransform();
      if (transform) {
        onCanvasClick(worldCoords, event);
      } else {
        onCanvasClick(null, event);
      }
    } catch (error) {
      console.error('Error handling canvas click:', error);
      onCanvasClick(null, event);
    }
  }, [onCanvasClick, canvasRenderer]);

  // 定义暴露给父组件的命令式 API
  useImperativeHandle(ref, () => ({
    // 核心渲染方法
    clearCanvas: canvasRenderer.clearCanvas,
    renderBoundaryPreview: canvasRenderer.renderBoundaryPreview,
    renderScene: canvasRenderer.drawScene, // 使用 drawScene 作为 renderScene 的别名
    drawScene: canvasRenderer.drawScene,

    // 坐标转换方法
    getCurrentTransform: canvasRenderer.getCurrentTransform,
    screenToWorld: canvasRenderer.screenToWorld,
    worldToScreen: canvasRenderer.worldToScreen,

    // 实用方法
    onResize: canvasRenderer.onResize,
    getRenderer: canvasRenderer.getRenderer,
    getCanvas: canvasRenderer.getCanvas,

    // 调试和测试方法（生产环境可以考虑移除）
    setSizeOverrides: canvasRenderer.setSizeOverrides,
    getCurrentSizes: canvasRenderer.getCurrentSizes,
  }), [canvasRenderer]);

  // 合并 CSS 类名
  const finalClassName = useMemo(() => {
    const baseClass = 'mesh-canvas';
    const classes = [baseClass];
    
    if (className) {
      classes.push(className);
    }

    return classes.join(' ');
  }, [className]);

  // 合并样式
  const finalStyle = useMemo(() => {
    const baseStyle = {
      display: 'block',
      maxWidth: '100%',
      maxHeight: '100%',
    };

    // 如果有点击回调，显示十字光标
    if (onCanvasClick) {
      baseStyle.cursor = 'crosshair';
    }

    return {
      ...baseStyle,
      ...style
    };
  }, [style, onCanvasClick]);

  return (
    <div className="mesh-canvas-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        className={finalClassName}
        style={finalStyle}
        onClick={onCanvasClick ? handleCanvasClick : undefined}
        {...canvasProps}
      />
      
      {/* 空状态覆盖层（可选，用于显示加载状态或错误信息） */}
      {/* 可以根据需要添加加载指示器或错误提示 */}
    </div>
  );
});

MeshCanvas.displayName = 'MeshCanvas';

export default MeshCanvas;
