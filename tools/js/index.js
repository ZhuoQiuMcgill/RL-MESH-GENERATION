/**
 * 模块索引文件
 * 统一导出所有模块，方便管理和使用
 */

// 工具模块
export * from './utils.js';

// API客户端模块
export { ApiClient, withErrorHandling, withRetry } from './api-client.js';

// Canvas渲染模块
export { CanvasRenderer } from './canvas-renderer.js';

// UI控制器模块
export { UIController } from './ui-controller.js';

// 训练管理器模块
export { TrainingManager } from './training-manager.js';

// 版本信息
export const VERSION = '1.0.0';

// 模块信息
export const MODULES = {
    utils: '工具函数模块',
    apiClient: 'API客户端模块',
    canvasRenderer: 'Canvas渲染模块',
    uiController: 'UI控制器模块',
    trainingManager: '训练管理器主模块'
};

// 检查所有模块是否正确加载
export function checkModules() {
    const results = {
        utils: typeof CONSTANTS !== 'undefined',
        apiClient: typeof ApiClient !== 'undefined',
        canvasRenderer: typeof CanvasRenderer !== 'undefined',
        uiController: typeof UIController !== 'undefined',
        trainingManager: typeof TrainingManager !== 'undefined'
    };

    console.log('模块加载状态:', results);
    return results;
}