/**
 * Module index file
 * Unified export of all modules for convenient management and usage
 */

// Utility module
export * from './utils.js';

// API client module
export { ApiClient, withErrorHandling, withRetry } from './api-client.js';

// Canvas renderer module
export { CanvasRenderer } from './canvas-renderer.js';

// UI controller module
export { UIController } from './ui-controller.js';

// Training manager module
export { TrainingManager } from './training-manager.js';

// Version information
export const VERSION = '1.0.0';

// Module information
export const MODULES = {
    utils: 'Utility module',
    apiClient: 'API client module',
    canvasRenderer: 'Canvas renderer module',
    uiController: 'UI controller module',
    trainingManager: 'Training manager module'
};

// Check if all modules are loaded correctly
export function checkModules() {
    const results = {
        utils: typeof CONSTANTS !== 'undefined',
        apiClient: typeof ApiClient !== 'undefined',
        canvasRenderer: typeof CanvasRenderer !== 'undefined',
        uiController: typeof UIController !== 'undefined',
        trainingManager: typeof TrainingManager !== 'undefined'
    };

    console.log('Module load status:', results);
    return results;
}