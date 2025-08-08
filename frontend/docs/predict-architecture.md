# RL-MESH-GENERATION 预测功能架构设计

## 目录

1. [架构概览](#架构概览)
2. [组件树设计](#组件树设计)
3. [状态管理架构](#状态管理架构)
4. [自定义Hooks设计](#自定义hooks设计)
5. [数据流设计](#数据流设计)
6. [API交互设计](#api交互设计)
7. [性能优化策略](#性能优化策略)
8. [错误处理机制](#错误处理机制)
9. [扩展性考虑](#扩展性考虑)

## 架构概览

### 设计原则
- **单一职责原则**：每个组件和Hook都有明确的职责
- **关注点分离**：UI组件、状态管理、API调用、3D渲染分离
- **可测试性**：通过依赖注入和纯函数提高可测试性
- **可扩展性**：模块化设计，便于功能扩展
- **性能优化**：通过合理的状态管理和渲染优化提升性能

### 技术栈
- **React 18**: 主要框架，利用并发特性
- **React Context + useReducer**: 全局状态管理
- **Custom Hooks**: 业务逻辑封装
- **Three.js**: 3D渲染引擎（待集成）
- **WebGL**: 硬件加速渲染

## 组件树设计

### 组件层次结构

```
PredictPage (主页面容器)
├── PredictSessionProvider (状态管理提供者)
    ├── ConfigurationPanel (配置面板)
    │   ├── GeometryConfig (几何配置)
    │   ├── MeshConfig (网格配置)
    │   └── AlgorithmConfig (算法配置)
    ├── MeshCanvas (网格画布)
    │   ├── CanvasRenderer (渲染器)
    │   ├── ViewportControls (视口控制)
    │   └── MeshVisualization (网格可视化)
    ├── ControlButtons (控制按钮组)
    │   ├── StartPredictionButton (开始预测)
    │   ├── PauseResumeButton (暂停/恢复)
    │   ├── StopButton (停止)
    │   └── ResetButton (重置)
    ├── StatusDisplay (状态显示)
    │   ├── SessionInfo (会话信息)
    │   ├── ProgressIndicator (进度指示器)
    │   └── CurrentStepInfo (当前步骤信息)
    └── OperationLog (操作日志)
        ├── LogList (日志列表)
        ├── LogFilter (日志过滤器)
        └── LogExport (日志导出)
```

### 组件职责分析

#### PredictPage (主页面容器)
**职责：**
- 作为预测功能的根组件
- 管理整体布局和导航
- 协调子组件间的交互
- 处理页面级的事件和生命周期

**Props接口：**
```typescript
interface PredictPageProps {
  initialConfig?: PredictConfiguration;
  onSessionComplete?: (sessionId: string, result: PredictResult) => void;
  onError?: (error: Error) => void;
}
```

#### ConfigurationPanel (配置面板)
**职责：**
- 提供网格生成的配置选项
- 验证配置参数的有效性
- 响应配置变化并更新全局状态

**状态：**
- 几何参数配置
- 网格参数配置  
- 算法参数配置
- 配置验证结果

#### MeshCanvas (网格画布)
**职责：**
- 3D网格的可视化显示
- 支持交互式视图操作
- 实时显示网格生成过程
- 处理用户的3D交互输入

**核心功能：**
- 网格渲染
- 相机控制
- 交互处理
- 性能监控

#### ControlButtons (控制按钮组)
**职责：**
- 提供预测流程控制功能
- 根据当前状态动态启用/禁用按钮
- 处理用户操作输入

**状态逻辑：**
```javascript
const buttonStates = {
  IDLE: { start: true, pause: false, stop: false, reset: true },
  RUNNING: { start: false, pause: true, stop: true, reset: false },
  PAUSED: { start: false, pause: false, stop: true, reset: false },
  COMPLETED: { start: true, pause: false, stop: false, reset: true }
};
```

#### StatusDisplay (状态显示)
**职责：**
- 显示当前预测会话状态
- 展示进度和步骤信息
- 提供实时状态更新

#### OperationLog (操作日志)
**职责：**
- 记录和显示操作日志
- 提供日志过滤和搜索功能
- 支持日志导出功能

## 状态管理架构

### PredictSessionContext 设计

#### 状态结构
```javascript
const SessionState = {
  // 会话信息
  sessionId: string | null,
  status: 'idle' | 'configuring' | 'initializing' | 'running' | 'paused' | 'completed' | 'error',
  
  // 配置信息
  configuration: {
    geometry: {
      type: 'rectangle' | 'circle' | 'polygon',
      width: number,
      height: number,
      complexity: number
    },
    mesh: {
      maxElementSize: number,
      minElementSize: number,
      quality: number
    },
    algorithm: {
      method: 'rl_ddpg' | 'rl_ppo' | 'traditional',
      maxSteps: number,
      learningRate: number
    }
  },
  
  // 运行时状态
  refPoint: { x: number, y: number, z: number } | null,
  currentStep: number,
  totalSteps: number,
  progress: number,
  meshData: MeshData | null,
  
  // 日志和错误
  logs: Array<LogEntry>,
  error: Error | null,
  
  // 时间信息
  startTime: string | null,
  endTime: string | null
};
```

#### Action Types
```javascript
const PredictSessionActions = {
  // 会话管理
  CREATE_SESSION: 'CREATE_SESSION',
  CONFIGURE_SESSION: 'CONFIGURE_SESSION',
  RESET_SESSION: 'RESET_SESSION',
  
  // 预测控制
  START_PREDICTION: 'START_PREDICTION',
  PAUSE_PREDICTION: 'PAUSE_PREDICTION',
  RESUME_PREDICTION: 'RESUME_PREDICTION',
  STOP_PREDICTION: 'STOP_PREDICTION',
  
  // 进度更新
  NEXT_STEP: 'NEXT_STEP',
  UPDATE_PROGRESS: 'UPDATE_PROGRESS',
  SET_REF_POINT: 'SET_REF_POINT',
  
  // 日志管理
  ADD_LOG: 'ADD_LOG',
  CLEAR_LOGS: 'CLEAR_LOGS',
  
  // 错误处理
  SET_ERROR: 'SET_ERROR'
};
```

#### Reducer 设计模式
使用纯函数reducer处理状态变更，确保状态的不可变性和可预测性：

```javascript
const predictSessionReducer = (state, action) => {
  switch (action.type) {
    case 'CREATE_SESSION':
      return {
        ...state,
        sessionId: action.payload.sessionId,
        status: 'configuring',
        startTime: new Date().toISOString(),
        error: null
      };
    
    case 'START_PREDICTION':
      return {
        ...state,
        status: 'initializing',
        currentStep: 0,
        progress: 0,
        logs: [...state.logs, createStartLog(action.payload)]
      };
    
    // ... 其他action处理
  }
};
```

### 状态管理最佳实践

1. **状态规范化**：复杂对象使用规范化存储
2. **选择性更新**：只更新变化的部分
3. **异步状态处理**：使用中间件模式处理异步操作
4. **状态持久化**：关键状态的本地存储

## 自定义Hooks设计

### usePredictApi Hook

**职责：**
- 封装所有与预测API相关的操作
- 处理网络请求的错误和重试
- 管理请求状态（loading, error）

**接口设计：**
```javascript
const {
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
  
  // 工具方法
  startStatusPolling,
  cleanup
} = usePredictApi();
```

**错误处理策略：**
- 网络错误自动重试
- 超时处理
- 错误状态统一管理

### useOperationLog Hook

**职责：**
- 日志的增删改查操作
- 日志过滤和搜索
- 日志导出功能

**特性：**
- 多种日志级别支持
- 性能优化的过滤算法
- 多格式导出（JSON, CSV, TXT）

### useCanvasRenderer Hook

**职责：**
- 3D渲染器的初始化和管理
- 渲染循环控制
- 性能监控

**核心功能：**
```javascript
const {
  // Refs
  canvasRef,
  
  // 状态
  renderState,
  performanceState,
  cameraState,
  
  // 控制方法
  initializeRenderer,
  updateMeshData,
  updateRenderSettings,
  
  // 相机控制
  setCameraPosition,
  resetCamera,
  fitCameraToMesh,
  
  // 工具方法
  captureScreenshot,
  exportMeshData
} = useCanvasRenderer();
```

## 数据流设计

### 单向数据流
遵循React的单向数据流原则：

```
用户操作 → Action Dispatch → Reducer → State Update → Component Re-render
```

### 异步数据流
```
用户操作 → Hook API调用 → 网络请求 → 响应处理 → State更新 → UI更新
```

### 3D渲染数据流
```
网格数据更新 → useCanvasRenderer → Three.js渲染 → Canvas显示
```

## API交互设计

### API端点设计
```javascript
const PREDICTION_ENDPOINTS = {
  createSession: '/api/predict/session/create',
  startPrediction: '/api/predict/start',
  pausePrediction: '/api/predict/pause',
  resumePrediction: '/api/predict/resume',
  stopPrediction: '/api/predict/stop',
  getStep: '/api/predict/step/:stepNumber',
  getStatus: '/api/predict/status',
  getMeshData: '/api/predict/mesh'
};
```

### 请求/响应格式

#### 创建会话请求
```json
{
  "configuration": {
    "geometry": { "type": "rectangle", "width": 10, "height": 10 },
    "mesh": { "maxElementSize": 0.5, "quality": 0.8 },
    "algorithm": { "method": "rl_ddpg", "maxSteps": 1000 }
  },
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### 状态查询响应
```json
{
  "sessionId": "sess_123456",
  "status": "running",
  "currentStep": 150,
  "totalSteps": 1000,
  "progress": 15.0,
  "meshData": { ... },
  "timestamp": "2024-01-01T00:01:30.000Z"
}
```

### 错误处理
```javascript
const API_ERROR_CODES = {
  SESSION_NOT_FOUND: 'SESSION_NOT_FOUND',
  INVALID_CONFIGURATION: 'INVALID_CONFIGURATION',
  PREDICTION_FAILED: 'PREDICTION_FAILED',
  RESOURCE_LIMIT_EXCEEDED: 'RESOURCE_LIMIT_EXCEEDED'
};
```

## 性能优化策略

### React性能优化
1. **memo化组件**：使用React.memo包装纯组件
2. **useCallback优化**：缓存回调函数
3. **useMemo优化**：缓存计算结果
4. **虚拟化**：长列表使用react-window

### 3D渲染优化
1. **LOD系统**：根据距离调整细节级别
2. **视锥剔除**：只渲染可见对象
3. **批处理渲染**：合并draw calls
4. **WebWorker**：将计算密集型操作移至worker

### 内存管理
1. **对象池**：复用Three.js几何体和材质
2. **资源清理**：及时清理不用的3D对象
3. **纹理压缩**：使用压缩纹理格式

## 错误处理机制

### 分层错误处理

1. **UI层错误**：使用Error Boundary捕获
2. **业务逻辑错误**：在Hook中处理和上报
3. **网络错误**：统一的retry机制
4. **渲染错误**：WebGL context丢失处理

### 错误恢复策略

```javascript
const ErrorRecoveryStrategies = {
  NETWORK_ERROR: 'retry_with_backoff',
  VALIDATION_ERROR: 'show_form_errors',
  RENDER_ERROR: 'fallback_to_2d',
  MEMORY_ERROR: 'reduce_quality_and_retry'
};
```

## 扩展性考虑

### 插件化架构
设计插件接口，支持：
- 自定义渲染器
- 算法扩展
- 导出格式扩展

### 主题系统
支持多种视觉主题：
```javascript
const ThemeConfig = {
  colors: { ... },
  spacing: { ... },
  typography: { ... },
  animations: { ... }
};
```

### 国际化支持
```javascript
const i18nKeys = {
  'predict.start': 'Start Prediction',
  'predict.pause': 'Pause',
  'predict.stop': 'Stop',
  // ... 更多翻译键
};
```

### 移动端适配
- 响应式设计
- 触摸操作优化
- 性能降级策略

## 测试策略

### 单元测试
- Hook测试：使用@testing-library/react-hooks
- 组件测试：使用@testing-library/react
- 工具函数测试：Jest

### 集成测试
- API交互测试
- 状态管理测试
- 端到端用户流程测试

### 性能测试
- 渲染性能监控
- 内存泄漏检测
- 长时间运行稳定性测试

## 部署和监控

### 构建优化
- 代码分割
- Tree shaking
- 资源压缩

### 监控指标
- 页面加载时间
- 3D渲染性能
- API响应时间
- 错误率统计

---

## 实现优先级

### Phase 1: 核心架构 ✅ (已完成)
- [x] PredictSessionContext状态管理
- [x] 基础Hook实现（usePredictApi, useOperationLog, useCanvasRenderer）
- [x] 组件树结构设计
- [x] 架构文档

### Phase 2: UI组件实现
- [ ] 基础组件开发
- [ ] 样式系统
- [ ] 交互逻辑

### Phase 3: 3D渲染集成
- [ ] Three.js集成
- [ ] 渲染管道
- [ ] 性能优化

### Phase 4: 高级特性
- [ ] 错误恢复
- [ ] 性能监控
- [ ] 扩展功能

---

*文档版本：v1.0*  
*最后更新：2024-08-07*  
*维护者：开发团队*
