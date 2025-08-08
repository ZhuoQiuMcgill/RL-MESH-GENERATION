# CanvasRenderer 重构为 React Hook

## 概述

本项目将原始的 `CanvasRenderer` 类重构为现代的 React Hook (`useCanvasRenderer`) 和组件 (`MeshCanvas`)，提供更好的 React 集成和响应式设计支持。

## 重构内容

### ✅ 1. useCanvasRenderer Hook (src/hooks/useCanvasRenderer.js)

将 `CanvasRenderer` 类包装为 React Hook，提供：
- 自动初始化和清理 CanvasRenderer 实例
- ResizeObserver 集成，自动响应容器大小变化
- 错误处理和边界检查
- 性能优化 (memoized 返回值)

**主要 API:**
```javascript
const canvasRenderer = useCanvasRenderer(canvasRef);

// 核心渲染方法
canvasRenderer.drawScene(meshData, boundaryVertices, refPointInfo);
canvasRenderer.renderBoundaryPreview(boundaryVertices, meshName);
canvasRenderer.clearCanvas();

// 坐标转换
const worldCoords = canvasRenderer.screenToWorld(screenX, screenY);
const screenCoords = canvasRenderer.worldToScreen([x, y]);
```

### ✅ 2. MeshCanvas 组件 (src/components/MeshCanvas.jsx)

React 组件实现，使用 `useCanvasRenderer` Hook：
- 使用 `forwardRef` 暴露命令式 API
- 自动处理点击事件和坐标转换
- 响应式样式和类名管理
- 与原始 CanvasRenderer 完全兼容

**使用方式:**
```jsx
<MeshCanvas
  ref={canvasRef}
  onCanvasClick={handleCanvasClick}
  className="my-canvas"
  style={{ width: '100%', height: '500px' }}
/>
```

### ✅ 3. 确保在 useEffect 中处理 resize 监听并清理

Hook 自动管理 ResizeObserver：
```javascript
// 自动设置
const resizeObserver = new ResizeObserver((entries) => {
  setTimeout(() => {
    if (rendererRef.current) {
      rendererRef.current.onResize();
    }
  }, 50);
});

// 自动清理
useEffect(() => {
  return () => {
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect();
    }
  };
}, []);
```

### ✅ 4. MeshCanvas 组件负责挂载 canvas 元素及空状态覆盖层

组件结构：
```jsx
<div className="mesh-canvas-container">
  <canvas
    ref={canvasRef}
    className={finalClassName}
    style={finalStyle}
    onClick={onCanvasClick ? handleCanvasClick : undefined}
    {...canvasProps}
  />
  {/* 空状态覆盖层（可选） */}
</div>
```

## 文件结构

```
src/
├── hooks/
│   ├── useCanvasRenderer.js       # React Hook 实现
│   ├── index.js                   # Hook 导出
│   └── __tests__/
│       └── useCanvasRenderer.test.js  # Hook 测试
├── components/
│   ├── MeshCanvas.jsx            # React 组件
│   ├── MeshCanvasExample.jsx     # 使用示例
│   └── index.js                  # 组件导出（已更新）
└── docs/
    └── MeshCanvas-Hook-Usage.md  # 详细使用文档
```

## 关键特性

### 🚀 自动生命周期管理
- Hook 自动处理 CanvasRenderer 初始化和清理
- 无需手动管理 useEffect 生命周期

### 📱 响应式设计
- ResizeObserver 自动监听容器大小变化
- 支持高 DPI 显示器
- 自动重绘缓存内容

### 🛡️ 错误处理
- 所有方法都包含错误边界
- 优雅处理未初始化状态
- 详细的错误日志

### ⚡ 性能优化
- useMemo 防止不必要的重新渲染
- 稳定的方法引用
- 自动去抖动 resize 事件

### 🔄 向后兼容
- 保持与原始 CanvasRenderer API 完全兼容
- 现有代码无需修改 API 调用

## API 保持兼容

所有原始 CanvasRenderer 方法都可以通过组件 ref 访问：

```javascript
// 原来的调用方式仍然有效
canvasRef.current.clearCanvas();
canvasRef.current.renderBoundaryPreview(vertices, 'Mesh Name');
canvasRef.current.renderScene(meshData, boundaries, refPoint);
canvasRef.current.screenToWorld(x, y);
canvasRef.current.worldToScreen([x, y]);
```

## 使用示例

查看 `src/components/MeshCanvasExample.jsx` 获取完整示例：

```jsx
import React, { useRef, useCallback } from 'react';
import { MeshCanvas } from './components';

const MyApp = () => {
  const canvasRef = useRef(null);

  const handleCanvasClick = useCallback((worldCoords, event) => {
    if (worldCoords) {
      console.log('Clicked at:', worldCoords);
    }
  }, []);

  const renderMesh = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.renderBoundaryPreview(boundaryData, 'My Mesh');
    }
  }, []);

  return (
    <div style={{ width: '800px', height: '600px' }}>
      <MeshCanvas
        ref={canvasRef}
        onCanvasClick={handleCanvasClick}
        style={{ border: '1px solid #ccc' }}
      />
      <button onClick={renderMesh}>Render Mesh</button>
    </div>
  );
};
```

## 测试

Hook 包含完整的单元测试：
- 初始化和清理测试
- 错误处理测试
- 坐标转换测试
- ResizeObserver 集成测试
- 性能优化验证

运行测试：
```bash
npm test src/hooks/__tests__/useCanvasRenderer.test.js
```

## 迁移指南

### 从手动 CanvasRenderer 迁移到 MeshCanvas

**之前:**
```javascript
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
```

**现在:**
```javascript
const canvasRef = useRef(null);

return (
  <MeshCanvas ref={canvasRef} />
);
```

## 优势

1. **简化使用**: 不再需要手动管理 CanvasRenderer 生命周期
2. **响应式**: 自动处理容器大小变化
3. **错误安全**: 所有操作都有错误处理
4. **性能优化**: 防止不必要的重新渲染
5. **向后兼容**: 现有代码无需修改
6. **测试友好**: Hook 和组件都易于测试

## 下一步

- 考虑添加更多可视化选项到组件 props
- 支持主题化和自定义样式
- 添加更多交互功能（缩放、平移等）
- 优化大型网格数据的性能

---

这次重构成功地将传统的类组件架构升级为现代的 React Hook 架构，同时保持了完全的向后兼容性和改进的开发体验。
