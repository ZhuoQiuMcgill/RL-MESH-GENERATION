# Predict Feature 组件树结构

## 组件层次结构

```
PredictPage (主页面容器)
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

## 组件职责

### PredictPage
- 作为预测功能的主容器
- 管理整个预测流程的布局
- 协调各子组件间的交互

### ConfigurationPanel
- 提供网格生成的配置选项
- 包含几何、网格和算法参数设置
- 验证配置参数的有效性

### MeshCanvas
- 3D网格的可视化显示
- 支持交互式视图操作
- 实时显示网格生成过程

### ControlButtons
- 提供预测流程控制功能
- 包含开始、暂停、停止、重置等操作
- 根据当前状态动态启用/禁用按钮

### StatusDisplay
- 显示当前预测会话状态
- 展示进度和步骤信息
- 提供实时状态更新

### OperationLog
- 记录和显示操作日志
- 提供日志过滤和搜索功能
- 支持日志导出功能
