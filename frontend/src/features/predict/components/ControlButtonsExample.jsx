import React from 'react';
import { PredictSessionProvider } from '../contexts/PredictSessionContext';
import ControlButtons from './ControlButtons';
import Card from '../../../components/Card';

/**
 * ControlButtons 使用示例组件
 * 展示如何在 PredictSessionProvider 上下文中使用 ControlButtons
 */
const ControlButtonsExample = () => {
  return (
    <PredictSessionProvider>
      <div className="p-6 space-y-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Control Buttons Example</h2>
          <p className="text-gray-600 mb-6">
            这些按钮根据当前会话状态自动启用/禁用，并提供完整的预测会话控制功能。
          </p>
          
          {/* Control Buttons */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">预测控制按钮</h3>
              <ControlButtons />
            </div>
            
            <div className="text-sm text-gray-500">
              <p><strong>Next:</strong> 执行下一步预测</p>
              <p><strong>Prev:</strong> 回退到上一步</p>
              <p><strong>Process All:</strong> 自动处理所有剩余步骤</p>
              <p><strong>Reset:</strong> 重置当前预测会话</p>
              <p><strong>Delete:</strong> 删除当前预测会话</p>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">功能特性</h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li>✅ 根据会话状态智能禁用/启用按钮</li>
            <li>✅ 集成加载动画和状态反馈</li>
            <li>✅ 完整的错误处理和日志记录</li>
            <li>✅ Process All 支持流式处理和进度日志</li>
            <li>✅ Reset 和 Delete 操作的安全确认</li>
            <li>✅ 与 PredictSessionContext 完全集成</li>
          </ul>
        </Card>

        <Card className="p-6 bg-blue-50">
          <h3 className="text-lg font-medium mb-2 text-blue-900">使用方式</h3>
          <pre className="text-sm bg-white p-3 rounded border overflow-x-auto">
{`import { ControlButtons } from '../features/predict/components';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';

function MyApp() {
  return (
    <PredictSessionProvider>
      <ControlButtons />
    </PredictSessionProvider>
  );
}`}
          </pre>
        </Card>
      </div>
    </PredictSessionProvider>
  );
};

export default ControlButtonsExample;
