# ControlButtons 组件开发文档

## 概述

ControlButtons 组件是预测功能的核心控制界面，提供了 Next、Prev、Process All、Reset、Delete 五个主要操作按钮。组件与 PredictSessionContext 深度集成，能够根据会话状态智能控制按钮的启用/禁用状态，并提供完整的加载动画和错误处理。

## 功能特性

### 1. 按钮功能

#### Next 按钮
- **功能**: 执行下一步预测
- **API调用**: `predictApi.nextStep(sessionId, stepData)`
- **状态更新**: 更新 `currentStep`，添加日志记录
- **禁用条件**: 无会话、正在运行、已完成且到达最大步数

#### Prev 按钮  
- **功能**: 回退到上一步
- **API调用**: `predictApi.prevStep(sessionId, stepData)`
- **状态更新**: 回退 `currentStep`，添加日志记录
- **禁用条件**: 无会话、正在运行、当前步数为0

#### Process All 按钮
- **功能**: 自动处理所有剩余步骤
- **API调用**: `predictApi.processAll(sessionId, processData)`
- **特殊处理**: 
  - 支持流式响应处理
  - 循环写入进度日志
  - 实时更新会话状态
- **禁用条件**: 无会话、正在运行、已完成

#### Reset 按钮
- **功能**: 重置当前预测会话
- **API调用**: `predictApi.resetSession(sessionId)`
- **状态更新**: 调用 `actions.resetSession()`
- **禁用条件**: 正在删除或处理全部时

#### Delete 按钮
- **功能**: 删除当前预测会话
- **API调用**: `predictApi.deleteSession(sessionId)`
- **安全措施**: 用户确认对话框
- **状态更新**: 重置所有状态
- **禁用条件**: 正在运行、正在处理全部时

### 2. 状态管理

#### 加载状态
每个按钮都有独立的加载状态管理：
```javascript
const [loadingStates, setLoadingStates] = useState({
  next: false,
  prev: false,
  processAll: false,
  reset: false,
  delete: false
});
```

#### 按钮禁用逻辑
基于以下条件动态计算：
- 会话存在性 (`sessionId`)
- 会话状态 (`status`)
- 当前步数 (`currentStep`)
- 总步数 (`totalSteps`)
- 其他按钮的加载状态 (`loadingStates`)

### 3. 错误处理

#### 统一错误处理
```javascript
const handleError = useCallback((error, operation) => {
  console.error(`${operation} failed:`, error);
  actions.setError(error);
  actions.addLog({
    level: 'error',
    message: `${operation} 操作失败: ${error.message}`,
    data: { error: error.message }
  });
}, [actions]);
```

#### 错误恢复策略
- Next/Prev: 记录错误，停止加载状态
- Process All: 暂停预测，记录错误
- Reset: 即使失败也尝试本地重置
- Delete: 记录错误，保持原状态

### 4. Process All 流式处理

Process All 按钮实现了复杂的异步流处理逻辑：

```javascript
// 流式响应处理
if (result.stream) {
  let stepCount = currentStep;
  
  for await (const stepResult of result.stream) {
    stepCount++;
    
    // 更新进度
    actions.nextStep({
      meshData: stepResult.meshData,
      stepData: stepResult.stepData,
      quality: stepResult.quality
    });

    // 写入日志
    actions.addLog({
      level: 'info',
      message: `处理步骤 ${stepCount} 完成`,
      data: {
        step: stepCount,
        quality: stepResult.quality,
        progress: `${stepCount}/${totalSteps || stepCount}`,
        executionTime: stepResult.executionTime
      }
    });

    // 避免UI阻塞
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
```

## 使用示例

### 基本使用
```jsx
import { ControlButtons } from '../features/predict/components';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';

function PredictPage() {
  return (
    <PredictSessionProvider>
      <div className="p-4">
        <ControlButtons />
      </div>
    </PredictSessionProvider>
  );
}
```

### 自定义样式
```jsx
<ControlButtons className="my-custom-class justify-center" />
```

### 与其他组件集成
```jsx
<PredictSessionProvider>
  <ConfigurationPanel />
  <ControlButtons className="mt-4" />
  <OperationLog />
</PredictSessionProvider>
```

## 依赖关系

### 核心依赖
- `PredictSessionContext`: 会话状态管理
- `predictApi`: API 调用接口
- `Button`: 基础按钮组件

### 状态依赖
- `sessionId`: 会话标识
- `status`: 会话状态 (PredictSessionStatus)
- `currentStep`: 当前步数
- `totalSteps`: 总步数
- `actions`: 状态更新方法

## 测试策略

### 单元测试要点
1. 按钮禁用逻辑测试
2. API 调用参数验证
3. 错误处理流程测试
4. 加载状态管理测试
5. 日志写入验证

### 集成测试要点
1. 与 PredictSessionContext 的集成
2. 流式处理的完整流程
3. 用户交互流程测试
4. 状态同步验证

### 测试用例示例
```javascript
describe('ControlButtons', () => {
  it('should disable Next button when no session exists', () => {
    // 测试无会话时按钮禁用
  });
  
  it('should handle Process All stream response correctly', () => {
    // 测试流式响应处理
  });
  
  it('should show confirmation dialog before delete', () => {
    // 测试删除确认
  });
});
```

## 性能考虑

### 优化措施
1. 使用 `useCallback` 优化函数引用
2. 独立的加载状态避免不必要的重渲染
3. Process All 中的延迟防止 UI 阻塞
4. 合理的错误边界处理

### 内存管理
- 组件卸载时清理定时器
- 避免内存泄漏的异步操作
- 合理的错误对象清理

## 扩展性

### 可扩展的设计
- 按钮配置可外部化
- 样式主题可定制
- API 接口可替换
- 状态管理可插拔

### 未来增强
- 批量操作支持
- 键盘快捷键
- 进度指示器
- 操作历史记录

## 注意事项

1. **会话依赖**: 组件必须在 PredictSessionProvider 内使用
2. **API 兼容性**: 依赖 predictApi 的特定接口格式
3. **浏览器兼容**: 使用了现代浏览器 API (async/await)
4. **用户体验**: Delete 操作需要用户确认，不可撤销
