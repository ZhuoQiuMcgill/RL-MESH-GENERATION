import React from 'react';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';
import { ConfigurationPanel, OperationLog } from '../features/predict/components';
import { useOperationLog, LogType } from '../features/predict/hooks/useOperationLog';

// 演示控制按钮组件
const DemoButtons = () => {
  const { addLog, clearLog } = useOperationLog();

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h3 className="text-lg font-semibold mb-4">操作日志演示</h3>
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={() => addLog(LogType.SYSTEM, '系统初始化完成')}
          className="bg-gray-500 text-white px-3 py-2 rounded text-sm hover:bg-gray-600"
        >
          系统日志
        </button>
        <button
          onClick={() => addLog(LogType.USER_ACTION, '用户修改配置参数')}
          className="bg-blue-500 text-white px-3 py-2 rounded text-sm hover:bg-blue-600"
        >
          用户操作
        </button>
        <button
          onClick={() => addLog(LogType.API_SUCCESS, 'API调用成功')}
          className="bg-green-500 text-white px-3 py-2 rounded text-sm hover:bg-green-600"
        >
          API成功
        </button>
        <button
          onClick={() => addLog(LogType.API_ERROR, 'API调用失败: 网络超时')}
          className="bg-red-500 text-white px-3 py-2 rounded text-sm hover:bg-red-600"
        >
          API失败
        </button>
        <button
          onClick={() => addLog(LogType.PREDICTION, '开始预测计算...')}
          className="bg-purple-500 text-white px-3 py-2 rounded text-sm hover:bg-purple-600"
        >
          预测日志
        </button>
        <button
          onClick={() => addLog(LogType.MESH, '网格生成完成')}
          className="bg-orange-500 text-white px-3 py-2 rounded text-sm hover:bg-orange-600"
        >
          网格日志
        </button>
      </div>
      <button
        onClick={clearLog}
        className="w-full mt-3 bg-gray-800 text-white px-3 py-2 rounded text-sm hover:bg-gray-900"
      >
        清空所有日志
      </button>
    </div>
  );
};

const ConfigDemo = () => {
  return (
    <PredictSessionProvider>
      <div className="min-h-screen bg-gray-100 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              配置面板和操作日志演示
            </h1>
            <p className="text-gray-600">
              测试 ConfigurationPanel 组件和 OperationLog 系统的功能
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 左侧：配置面板 */}
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              <ConfigurationPanel />
            </div>
            
            {/* 右侧：操作日志区域 */}
            <div>
              <DemoButtons />
              <OperationLog height={400} className="shadow-lg" />
            </div>
          </div>
        </div>
      </div>
    </PredictSessionProvider>
  );
};

export default ConfigDemo;
