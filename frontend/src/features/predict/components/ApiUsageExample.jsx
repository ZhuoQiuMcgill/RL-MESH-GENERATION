import React from 'react';
import { usePredictApiWithContext } from '../hooks';
import { usePredictSession } from '../contexts/PredictSessionContext';

/**
 * API使用示例组件
 * 演示如何使用新的统一错误处理和Loading反馈系统
 */
const ApiUsageExample = () => {
  const { loading, error } = usePredictSession();
  const api = usePredictApiWithContext();

  const handleListComponents = async () => {
    try {
      const result = await api.listComponents();
      console.log('组件列表:', result);
    } catch (error) {
      // 错误已经由Context自动处理，这里可以做额外处理
      console.log('API调用失败，但已通过Context处理');
    }
  };

  const handleCreateSession = async () => {
    try {
      const sessionData = {
        name: 'Test Session',
        configuration: {
          geometry: { type: 'rectangle', width: 5, height: 5 },
          mesh: { maxElementSize: 0.5, quality: 0.8 }
        }
      };
      const result = await api.createSession(sessionData);
      console.log('会话创建成功:', result);
    } catch (error) {
      console.log('会话创建失败，但已通过Context处理');
    }
  };

  const handleNextStep = async () => {
    try {
      const result = await api.nextStep('test-session-id', {
        action: 'refine',
        parameters: { refinementLevel: 1 }
      });
      console.log('下一步执行成功:', result);
    } catch (error) {
      console.log('下一步执行失败，但已通过Context处理');
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">API使用示例</h2>
      
      {/* 状态显示 */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <p className="text-sm">
          <strong>Loading状态:</strong> {loading ? '加载中...' : '空闲'}
        </p>
        <p className="text-sm">
          <strong>错误状态:</strong> {error ? error.message : '无错误'}
        </p>
      </div>

      {/* API调用按钮 */}
      <div className="space-y-4">
        <button
          onClick={handleListComponents}
          disabled={loading}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          获取组件列表
        </button>

        <button
          onClick={handleCreateSession}
          disabled={loading}
          className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          创建预测会话
        </button>

        <button
          onClick={handleNextStep}
          disabled={loading}
          className="w-full px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          执行下一步（会报错演示）
        </button>
      </div>

      {/* 说明文本 */}
      <div className="mt-8 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-semibold mb-2">功能说明:</h3>
        <ul className="text-sm space-y-1">
          <li>• 自动显示全局Loading覆盖层</li>
          <li>• 自动处理API错误并显示Toast提示</li>
          <li>• 错误信息会自动记录到操作日志</li>
          <li>• 支持自动清除错误状态</li>
          <li>• 所有按钮在Loading时自动禁用</li>
        </ul>
      </div>
    </div>
  );
};

export default ApiUsageExample;
