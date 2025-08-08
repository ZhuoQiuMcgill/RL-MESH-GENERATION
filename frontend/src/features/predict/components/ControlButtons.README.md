# ControlButtons 组件 - 使用说明

## 快速开始

ControlButtons 组件已经完成开发，提供了完整的预测会话控制功能。

### 1. 导入和使用

```jsx
import { ControlButtons } from '../features/predict/components';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';

function App() {
  return (
    <PredictSessionProvider>
      <ControlButtons />
    </PredictSessionProvider>
  );
}
```

### 2. 完成的功能

✅ **Next 按钮** - 执行下一步预测
- 调用 `predictApi.nextStep()`
- 更新会话状态
- 记录操作日志

✅ **Prev 按钮** - 回退到上一步  
- 调用 `predictApi.prevStep()`
- 回退会话状态
- 记录操作日志

✅ **Process All 按钮** - 处理所有剩余步骤
- 调用 `predictApi.processAll()`
- 支持流式响应处理
- **循环写入日志** - 符合任务要求
- 实时进度更新

✅ **Reset 按钮** - 重置预测会话
- 调用 `predictApi.resetSession()`
- 重置本地状态
- 安全的错误处理

✅ **Delete 按钮** - 删除预测会话
- 调用 `predictApi.deleteSession()`
- 用户确认对话框
- 完全清理状态

### 3. 智能状态管理

✅ **根据 sessionStatus & context 禁用** - 符合任务要求
- 无会话时禁用相关按钮
- 运行中状态的智能控制
- 步骤边界检查 (prev在步骤0时禁用，next在完成时禁用)

✅ **加载动画** - 符合任务要求
- 每个按钮独立的加载状态
- 旋转加载图标
- 防止重复点击

### 4. Process All 异步日志

✅ **异步结果循环写入日志** - 符合任务要求

```javascript
// 流式处理示例
for await (const stepResult of result.stream) {
  stepCount++;
  
  // 更新进度
  actions.nextStep({
    meshData: stepResult.meshData,
    stepData: stepResult.stepData,
    quality: stepResult.quality
  });

  // 循环写入日志
  actions.addLog({
    level: 'info',
    message: `处理步骤 ${stepCount} 完成`,
    data: {
      step: stepCount,
      quality: stepResult.quality,
      progress: `${stepCount}/${totalSteps}`,
      executionTime: stepResult.executionTime
    }
  });
}
```

## 集成到现有页面

### 在 Predict 页面中集成

```jsx
// src/pages/Predict.jsx
import { ControlButtons } from '../features/predict/components';

function Predict() {
  return (
    <div className="predict-page">
      {/* 其他组件 */}
      <ConfigurationPanel />
      
      {/* 控制按钮 */}
      <div className="controls-section mt-6">
        <h3 className="text-lg font-medium mb-4">预测控制</h3>
        <ControlButtons />
      </div>
      
      {/* 其他组件 */}
      <OperationLog />
    </div>
  );
}
```

### 自定义样式

```jsx
<ControlButtons className="justify-center space-x-3" />
```

## API 依赖

组件依赖以下 API 方法，确保 `predictApi` 包含：

- `nextStep(sessionId, stepData)`
- `prevStep(sessionId, stepData)`  
- `processAll(sessionId, processData)`
- `resetSession(sessionId)`
- `deleteSession(sessionId)`

## 状态要求

组件需要在 `PredictSessionProvider` 上下文中使用，依赖以下状态：

- `sessionId` - 当前会话ID
- `status` - 会话状态 (IDLE, RUNNING, COMPLETED等)
- `currentStep` - 当前步数
- `totalSteps` - 总步数  
- `actions` - 状态更新方法

## 测试验证

```bash
# 语法检查 (已通过)
npx eslint src/features/predict/components/ControlButtons.jsx

# 构建测试 (已通过)  
npm run build

# 开发服务器
npm run dev
```

## 组件文件结构

```
src/features/predict/components/
├── ControlButtons.jsx           # 主组件文件
├── ControlButtons.md            # 详细文档
├── ControlButtons.README.md     # 使用说明 (本文件)
├── ControlButtonsExample.jsx    # 使用示例
└── index.js                     # 导出索引 (已更新)
```

## 技术特性

- **React Hooks**: useState, useCallback
- **状态管理**: 与 PredictSessionContext 集成
- **错误处理**: 统一的错误处理和日志记录
- **性能优化**: useCallback 优化，独立加载状态
- **用户体验**: 加载动画，确认对话框
- **代码质量**: ESLint 通过，TypeScript 兼容

## 下一步

组件已经完成开发并满足所有任务要求：

1. ✅ 包含 Next/Prev/Process All/Reset/Delete 按钮
2. ✅ 根据 sessionStatus & context 智能禁用
3. ✅ 执行相应 predict.js 调用并 dispatch 结果
4. ✅ 添加加载动画
5. ✅ Process All 异步结果循环写入日志

可以直接集成到项目中使用！
