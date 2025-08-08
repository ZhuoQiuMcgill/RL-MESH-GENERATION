# Predict Feature Architecture

## 项目结构

```
frontend/src/features/predict/
├── components/
│   └── component-tree.md          # 组件树结构设计
├── contexts/
│   └── PredictSessionContext.jsx  # 全局状态管理
├── hooks/
│   ├── index.js                   # Hooks导出索引
│   ├── usePredictApi.js          # API交互Hook
│   ├── useOperationLog.js        # 日志管理Hook
│   └── useCanvasRenderer.js      # 3D渲染Hook
├── index.jsx                      # 功能入口
├── routes.js                     # 路由配置
└── README.md                     # 本文件

frontend/docs/
└── predict-architecture.md       # 详细架构文档
```

## 核心架构组件

### 1. 状态管理 (PredictSessionContext)
- **使用 React Context + useReducer** 统一管理预测会话状态
- **支持的状态**：sessionId, status, configuration, refPoint, logs 等
- **Action系统**：CREATE_SESSION, NEXT_STEP, START_PREDICTION 等

### 2. 自定义Hooks

#### usePredictApi
- 封装所有预测相关的API调用
- 处理网络错误和重试逻辑
- 状态轮询和请求取消

#### useOperationLog  
- 日志的增删改查和过滤
- 多格式导出（JSON, CSV, TXT）
- 性能优化的日志管理

#### useCanvasRenderer
- 3D渲染器的初始化和管理
- 渲染循环和性能监控
- 相机控制和交互处理

### 3. 组件树设计

```
PredictPage
├── ConfigurationPanel (配置面板)
├── MeshCanvas (网格画布)  
├── ControlButtons (控制按钮)
├── StatusDisplay (状态显示)
└── OperationLog (操作日志)
```

## 使用示例

### 基础使用
```jsx
import { PredictSessionProvider, usePredictSession } from './contexts/PredictSessionContext';
import { usePredictApi, useOperationLog, useCanvasRenderer } from './hooks';

function PredictPage() {
  return (
    <PredictSessionProvider>
      <div className="predict-page">
        <ConfigurationPanel />
        <MeshCanvas />
        <ControlButtons />
        <StatusDisplay />
        <OperationLog />
      </div>
    </PredictSessionProvider>
  );
}

function ConfigurationPanel() {
  const { configuration, actions } = usePredictSession();
  const { createPredictionSession } = usePredictApi();
  
  const handleCreateSession = async () => {
    await createPredictionSession(configuration);
  };
  
  // ... 组件实现
}
```

### Hook使用
```jsx
function MeshCanvas() {
  const { 
    canvasRef, 
    initializeRenderer, 
    updateMeshData 
  } = useCanvasRenderer();
  
  const { meshData } = usePredictSession();
  
  useEffect(() => {
    initializeRenderer();
  }, []);
  
  useEffect(() => {
    if (meshData) {
      updateMeshData(meshData);
    }
  }, [meshData]);
  
  return <canvas ref={canvasRef} />;
}
```

## 开发指南

### 状态管理
- 使用 `usePredictSession()` 访问全局状态
- 通过 `actions` 对象分发状态更新
- 所有状态变更都通过reducer处理

### API调用
- 使用 `usePredictApi()` 进行网络请求
- 自动处理loading状态和错误
- 支持请求取消和重试

### 日志系统
- 使用 `useOperationLog()` 管理日志
- 支持多级别日志：debug, info, warning, error
- 内置过滤和导出功能

### 3D渲染
- 使用 `useCanvasRenderer()` 管理3D场景
- 支持多种渲染模式和相机控制
- 性能监控和资源管理

## 下一步计划

1. **UI组件实现** - 基于设计实现具体的UI组件
2. **Three.js集成** - 完整的3D渲染管道
3. **测试覆盖** - 单元测试和集成测试
4. **性能优化** - 渲染和状态管理优化

## 相关文档

- [详细架构文档](../../docs/predict-architecture.md)
- [组件树设计](./components/component-tree.md)
- [Legacy系统分析](../../docs/predict-legacy-analysis.md)
