# ConfigurationPanel 组件实现文档

## 概述

ConfigurationPanel 是一个完整的配置面板组件，实现了网格生成预测会话的配置功能。它将旧的 HTML 表单拆分为可复用的 React 受控组件，并与 PredictSessionContext 集成以管理状态。

## 实现的功能

### 1. 可复用的受控组件

#### Select 组件
- **位置**: `src/components/Select.jsx`
- **功能**: 通用下拉选择组件
- **特性**:
  - 受控组件模式
  - 支持选项数据格式化
  - 加载状态显示
  - 禁用状态
  - 工具提示支持

#### NumberInput 组件
- **位置**: `src/components/NumberInput.jsx`
- **功能**: 数字输入组件
- **特性**:
  - 受控组件模式
  - 最小/最大值限制
  - 步进控制
  - 居中显示
  - 输入验证

#### Button 组件
- **位置**: `src/components/Button.jsx`（已存在）
- **特性**:
  - 多种变体（primary, secondary, outline, danger）
  - 尺寸变化
  - 禁用状态
  - 过渡动画

### 2. ConfigurationPanel 主组件

#### 位置
`src/features/predict/components/ConfigurationPanel.jsx`

#### 核心功能

**数据获取**:
- 组件挂载时调用 `listComponents` API
- 加载网格、预测器、模型、质量方法等数据
- 错误处理和重试机制

**配置管理**:
- 使用 useReducer 管理本地配置状态
- 支持网格、预测器、参考选择器、质量方法选择
- 动态显示/隐藏配置项（基于选择的预测器类型）

**Context 集成**:
- 与 PredictSessionContext 集成
- 实时 dispatch CONFIG_UPDATED action
- 传递配置有效性状态

**表单验证**:
- 实时验证配置完整性
- 根据预测器类型调整验证逻辑
- 按钮状态管理

### 3. Context 更新

#### PredictSessionContext 增强
- 添加了 `CONFIG_UPDATED` action type
- 实现了 `configUpdate` action creator
- 支持配置状态的集中管理

## 样式和 UI

### Tailwind CSS 应用
- 使用 Tailwind 实用类进行样式设计
- 响应式布局
- 现代化的表单控件样式

### 设计特性
- 清晰的信息层次
- 一致的间距和颜色方案
- 交互反馈（hover、focus、disabled 状态）
- 错误状态显示

### 复刻旧 CSS 变量
- 保持与现有设计系统的一致性
- 使用 CSS 变量进行主题化

## 使用方式

### 基本使用
```jsx
import { ConfigurationPanel } from '../features/predict/components';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';

function App() {
  return (
    <PredictSessionProvider>
      <ConfigurationPanel />
    </PredictSessionProvider>
  );
}
```

### Props
- `disabled` (boolean): 禁用整个表单
- `className` (string): 额外的 CSS 类名

### Context 集成
组件自动与 PredictSessionContext 集成，无需额外配置。配置更新会实时传递到上下文中。

## 文件结构

```
src/
├── components/
│   ├── Select.jsx           # 下拉选择组件
│   ├── NumberInput.jsx      # 数字输入组件
│   └── index.js            # 组件导出
├── features/predict/
│   ├── components/
│   │   ├── ConfigurationPanel.jsx  # 主配置面板
│   │   └── index.js                # 组件导出
│   └── contexts/
│       └── PredictSessionContext.jsx # 更新的上下文
└── pages/
    └── ConfigDemo.jsx       # 演示页面
```

## API 集成

### 使用的 API
- `listComponents()`: 获取可用组件列表
  - 网格列表 (initial_meshes)
  - 预测器列表 (predictors)
  - 参考选择器列表 (reference_selectors)
  - 质量方法列表 (quality_methods)
  - 训练模型列表 (trained_models)

### 数据格式
API 返回的数据会被转换为组件所需的 `{ value, label, description }` 格式。

## 特殊功能

### 动态配置显示
- 只有选择了 RL 预测器时才显示模型选择和参数配置
- 只有选择了非默认参考选择器时才显示其配置

### 实时验证
- 配置完整性实时检查
- 按钮状态动态更新
- Context 状态同步更新

### 错误处理
- API 调用失败时显示错误信息
- 提供重新加载选项
- 优雅的降级体验

## 测试

可以通过访问 ConfigDemo 页面来测试组件功能：
```jsx
import ConfigDemo from './pages/ConfigDemo';
```

## 后续扩展

1. 添加更多配置项
2. 实现配置保存/加载
3. 添加配置模板功能
4. 国际化支持
5. 更多的验证规则
