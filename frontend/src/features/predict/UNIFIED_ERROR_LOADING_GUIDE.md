# 统一错误处理与 Loading 反馈系统

## 概述

这个系统提供了统一的错误处理和Loading状态管理，确保用户界面的一致性和良好的用户体验。

## 功能特点

- ✅ 统一的API错误处理
- ✅ 全局Loading状态管理  
- ✅ 自动错误Toast提示
- ✅ 自动日志记录
- ✅ 页面顶层组件挂载
- ✅ 向后兼容的API设计

## 核心组件

### 1. Context State 增强

在 `PredictSessionContext` 中增加了两个新字段：
- `error: null` - 当前错误状态
- `loading: false` - 当前加载状态

### 2. 全局组件

#### LoadingOverlay
显示全局Loading覆盖层，阻止用户操作并提供视觉反馈。

```jsx
<LoadingOverlay isLoading={loading} message="正在处理..." />
```

#### ErrorToast  
显示错误信息的Toast提示，支持自动关闭。

```jsx
<ErrorToast 
  error={error} 
  onClose={() => actions.clearError()}
  autoClose={true}
  duration={5000}
/>
```

### 3. API集成

#### createPredictApiWithDispatch
创建集成了Context的API方法，自动处理Loading和错误状态。

```javascript
const api = createPredictApiWithDispatch(dispatch, addLog);
```

#### usePredictApiWithContext Hook
简化API使用的Hook，自动集成Context功能。

```javascript
const api = usePredictApiWithContext();
```

## 使用方法

### 1. 基本使用

```javascript
import { usePredictApiWithContext } from '../hooks';
import { usePredictSession } from '../contexts/PredictSessionContext';

const MyComponent = () => {
  const { loading, error } = usePredictSession();
  const api = usePredictApiWithContext();
  
  const handleApiCall = async () => {
    try {
      // API会自动设置loading状态
      const result = await api.createSession(sessionData);
      // 成功处理
    } catch (error) {
      // 错误已自动处理，这里可做额外处理
    }
  };

  return (
    <div>
      <button 
        onClick={handleApiCall}
        disabled={loading}
      >
        {loading ? '处理中...' : '开始处理'}
      </button>
    </div>
  );
};
```

### 2. 手动控制状态

```javascript
const { actions } = usePredictSession();

// 手动设置Loading状态
actions.setLoading(true);

// 手动设置错误
actions.apiError({
  message: '自定义错误',
  operation: '自定义操作'
});

// 清除错误
actions.clearError();
```

## 新增Action Types

```javascript
{
  API_ERROR: 'API_ERROR',      // API错误
  SET_LOADING: 'SET_LOADING',  // 设置Loading状态
  CLEAR_ERROR: 'CLEAR_ERROR'   // 清除错误
}
```

## 错误处理流程

1. API调用开始 → 设置 `loading: true`
2. API调用失败 → 触发 `API_ERROR` action
3. Context更新状态 → `error` 设置为错误信息，`loading: false`  
4. ErrorToast显示错误 → 用户看到错误提示
5. 自动/手动清除错误 → `error: null`

## 项目集成

系统已自动集成到 `AppRouter` 中：

```jsx
<PredictSessionProvider>
  <AppContent />
  <LoadingOverlay isLoading={loading} />
  <ErrorToast error={error} onClose={actions.clearError} />
</PredictSessionProvider>
```

## 向后兼容

原有的API方法仍然可用，新系统不会影响现有代码：

```javascript
// 旧API - 仍然可用
import predictApi from '../../../shared/api/predict';
const result = await predictApi.createSession(data);

// 新API - 带Context集成
const api = usePredictApiWithContext();  
const result = await api.createSession(data);
```

## 最佳实践

1. **统一使用Hook**: 推荐使用 `usePredictApiWithContext()` 获取API方法
2. **利用Loading状态**: 在UI中使用loading状态禁用按钮、显示加载指示器
3. **错误处理**: 依赖自动错误处理，必要时添加业务特定的错误处理
4. **日志记录**: 所有API错误会自动记录到操作日志中
5. **用户体验**: LoadingOverlay会阻止用户操作，避免重复提交

## 示例组件

参考 `ApiUsageExample.jsx` 组件查看完整的使用示例。
