# MeshCanvas Hook 实现文档

## 概述

本文档介绍了将 `CanvasRenderer` 类重构为 React Hook (`useCanvasRenderer`) 和组件 (`MeshCanvas`) 的新实现。

## 架构设计

```
MeshCanvas (React Component)
    ├── useCanvasRenderer (React Hook)
    │   ├── 包装 CanvasRenderer 类
    │   ├── 管理生命周期和清理
    │   ├── 处理 ResizeObserver
    │   └── 提供响应式 API
    └── 提供命令式接口 (forwardRef)
```

## 主要文件

### 1. `src/hooks/useCanvasRenderer.js`

React Hook 实现，将 `CanvasRenderer` 类包装成响应式的 Hook。

**核心功能:**
- 自动初始化和清理 CanvasRenderer 实例
- ResizeObserver 集成，自动响应容器大小变化
- 错误处理和边界检查
- memoized 返回值防止不必要的重新渲染

**主要方法:**
```javascript
const canvasRenderer = useCanvasRenderer(canvasRef);

// 核心渲染方法
canvasRenderer.drawScene(meshData, boundaryVertices, refPointInfo);
canvasRenderer.renderBoundaryPreview(boundaryVertices, meshName);
canvasRenderer.clearCanvas();

// 坐标转换
const worldCoords = canvasRenderer.screenToWorld(screenX, screenY);
const screenCoords = canvasRenderer.worldToScreen([x, y]);
const transform = canvasRenderer.getCurrentTransform();

// 实用方法
canvasRenderer.onResize();
const renderer = canvasRenderer.getRenderer();
const canvas = canvasRenderer.getCanvas();
```

### 2. `src/components/MeshCanvas.jsx`

React 组件实现，使用 `useCanvasRenderer` Hook 并提供声明式接口。

**特性:**
- 使用 `forwardRef` 暴露命令式 API
- 自动处理点击事件和坐标转换
- 响应式样式和类名管理
- 与 CanvasRenderer 完全兼容的 API

**使用方式:**
```javascript
import MeshCanvas from './components/MeshCanvas';

const MyComponent = () => {
  const canvasRef = useRef(null);

  const handleCanvasClick = (worldCoords, event) => {
    if (worldCoords) {
      console.log('Clicked at:', worldCoords);
    }
  };

  const renderData = () => {
    canvasRef.current?.renderScene(meshData, boundaryData, refPointInfo);
  };

  return (
    <div style={{ width: '800px', height: '600px' }}>
      <MeshCanvas
        ref={canvasRef}
        onCanvasClick={handleCanvasClick}
        className="my-canvas"
        style={{ border: '1px solid #ccc' }}
      />
    </div>
  );
};
```

## API 参考

### useCanvasRenderer Hook

```javascript
const canvasRenderer = useCanvasRenderer(canvasRef);
```

**参数:**
- `canvasRef`: React ref 对象，指向 canvas 元素

**返回值:**
```javascript
{
  // 核心渲染方法
  drawScene: (meshData, boundaryVertices, refPointInfo?) => void,
  renderBoundaryPreview: (boundaryVertices, meshName?) => void,
  clearCanvas: () => void,
  
  // 坐标转换
  screenToWorld: (screenX, screenY) => [number, number],
  worldToScreen: (worldCoords) => [number, number], 
  getCurrentTransform: () => Object | null,
  
  // 实用方法
  onResize: () => void,
  getRenderer: () => CanvasRenderer | null,
  getCanvas: () => HTMLCanvasElement | null,
  
  // 调试方法
  setSizeOverrides: (overrides) => void,
  getCurrentSizes: () => Object,
  cleanup: () => void
}
```

### MeshCanvas 组件

```javascript
<MeshCanvas
  ref={canvasRef}
  className={string}
  style={Object}
  onCanvasClick={(worldCoords, event) => void}
  {...canvasProps}
/>
```

**Props:**
- `className`: 额外的 CSS 类名
- `style`: 内联样式对象
- `onCanvasClick`: 点击事件回调函数
- `...canvasProps`: 传递给 canvas 元素的其他属性

**暴露的方法 (via ref):**
```javascript
// 与原始 CanvasRenderer 完全兼容的 API
canvasRef.current.clearCanvas();
canvasRef.current.renderBoundaryPreview(vertices, meshName);
canvasRef.current.renderScene(meshData, boundaries, refPoint);
canvasRef.current.screenToWorld(x, y);
canvasRef.current.worldToScreen([x, y]);
canvasRef.current.getCurrentTransform();
canvasRef.current.onResize();
```

## 关键改进

### 1. 自动生命周期管理

```javascript
// Hook 自动处理初始化和清理
useEffect(() => {
  if (canvasRef?.current) {
    initRenderer();
  }
  return cleanup; // 自动清理
}, [canvasRef, initRenderer, cleanup]);
```

### 2. ResizeObserver 集成

```javascript
// 自动响应容器大小变化
const resizeObserver = new ResizeObserver((entries) => {
  setTimeout(() => {
    if (rendererRef.current) {
      rendererRef.current.onResize();
    }
  }, 50);
});

resizeObserver.observe(container);
```

### 3. 错误边界处理

```javascript
// 所有方法都包含错误处理
const drawScene = useCallback((meshData, boundaryVertices, refPointInfo) => {
  if (!rendererRef.current) {
    console.warn('CanvasRenderer not initialized');
    return;
  }

  try {
    rendererRef.current.renderScene(meshData, boundaryVertices, refPointInfo);
  } catch (error) {
    console.error('Failed to render scene:', error);
  }
}, []);
```

### 4. 性能优化

```javascript
// 使用 useMemo 防止不必要的重新渲染
return useMemo(() => ({
  drawScene,
  renderBoundaryPreview,
  clearCanvas,
  // ... 其他方法
}), [
  drawScene,
  renderBoundaryPreview,
  clearCanvas,
  // ... 依赖数组
]);
```

## 迁移指南

### 从原始 CanvasRenderer 迁移

**之前:**
```javascript
// 手动管理 CanvasRenderer 实例
const canvasRef = useRef(null);
const rendererRef = useRef(null);

useEffect(() => {
  if (canvasRef.current) {
    rendererRef.current = new CanvasRenderer(canvasRef.current);
  }
  
  return () => {
    if (rendererRef.current) {
      rendererRef.current.destroy();
    }
  };
}, []);

// 手动处理 resize
useEffect(() => {
  const handleResize = () => rendererRef.current?.onResize();
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);
```

**现在:**
```javascript
// 使用 MeshCanvas 组件，自动处理一切
const canvasRef = useRef(null);

return (
  <MeshCanvas
    ref={canvasRef}
    onCanvasClick={handleClick}
    style={{ width: '100%', height: '500px' }}
  />
);
```

### 向后兼容性

新实现保持了与原始 CanvasRenderer API 的完全兼容性：

```javascript
// 原始 API 调用方式保持不变
canvasRef.current.clearCanvas();
canvasRef.current.renderBoundaryPreview(vertices, 'Mesh Name');
canvasRef.current.renderScene(meshData, boundaries, refPoint);

// 坐标转换方法签名相同
const worldCoords = canvasRef.current.screenToWorld(x, y);
const screenCoords = canvasRef.current.worldToScreen([x, y]);
```

## 最佳实践

### 1. 容器尺寸设置

```javascript
// 确保父容器有明确的尺寸
<div style={{ width: '800px', height: '600px' }}>
  <MeshCanvas ref={canvasRef} />
</div>
```

### 2. 点击事件处理

```javascript
const handleCanvasClick = useCallback((worldCoords, event) => {
  if (worldCoords) {
    // 有效的坐标转换
    console.log('World coordinates:', worldCoords);
  } else {
    // 无效的坐标转换（例如未渲染数据）
    console.log('No valid coordinate transform available');
  }
}, []);
```

### 3. 渲染数据前的检查

```javascript
const renderData = useCallback(() => {
  if (canvasRef.current && meshData && boundaryData) {
    canvasRef.current.renderScene(meshData, boundaryData, refPointInfo);
  }
}, [meshData, boundaryData, refPointInfo]);
```

### 4. 错误处理

```javascript
const renderSafely = useCallback(() => {
  try {
    canvasRef.current?.renderScene(meshData, boundaryData);
  } catch (error) {
    console.error('Rendering failed:', error);
    // 显示错误提示给用户
  }
}, [meshData, boundaryData]);
```

## 示例代码

查看 `src/components/MeshCanvasExample.jsx` 获取完整的使用示例，包括：
- 基本渲染操作
- 点击交互处理
- 坐标转换演示
- 动态数据更新

## 故障排除

### 常见问题

1. **画布不显示内容**
   - 确保父容器有明确的宽高
   - 检查数据格式是否正确
   - 查看控制台错误日志

2. **点击坐标不准确**
   - 确保画布正确初始化
   - 检查是否调用了数据渲染方法
   - 验证坐标转换是否有效

3. **resize 不工作**
   - Hook 会自动处理 ResizeObserver
   - 如果需要手动触发，调用 `canvasRef.current.onResize()`

4. **内存泄露**
   - Hook 会自动清理资源
   - 不需要手动调用 cleanup 方法

## 总结

新的 Hook 实现提供了：
- ✅ 更好的 React 集成
- ✅ 自动生命周期管理
- ✅ 改进的错误处理
- ✅ 响应式设计支持
- ✅ 完全向后兼容
- ✅ 更简洁的使用方式

通过将 CanvasRenderer 包装为 React Hook，我们获得了现代 React 应用所需的响应式和声明式特性，同时保持了原有的强大功能。
