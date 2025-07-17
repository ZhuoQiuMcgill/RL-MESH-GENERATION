/**
 * API客户端模块
 * 负责与后端API的所有通信
 */

import {CONSTANTS, delay} from './utils.js';

/**
 * API错误处理装饰器
 * @param {Function} apiMethod - API方法
 * @returns {Function} 包装后的方法
 */
export function withErrorHandling(apiMethod) {
    return async function (...args) {
        try {
            return await apiMethod.apply(this, args);
        } catch (error) {
            console.error(`API调用失败:`, error);

            // 根据错误类型返回不同的错误信息
            if (error.message.includes('timeout') || error.message.includes('超时')) {
                throw new Error('请求超时，请检查网络连接');
            } else if (error.message.includes('Failed to fetch')) {
                throw new Error('网络连接失败，请检查服务器状态');
            } else {
                throw error;
            }
        }
    };
}

/**
 * 带重试机制的API调用
 * @param {Function} apiCall - API调用函数
 * @param {number} maxRetries - 最大重试次数
 * @param {number} retryDelay - 重试延迟时间（毫秒）
 * @returns {Promise<any>} API响应
 */
export async function withRetry(apiCall, maxRetries = 1, retryDelay = 3000) {
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await apiCall();
        } catch (error) {
            lastError = error;

            if (attempt < maxRetries) {
                console.warn(`API调用失败，尝试重试 (${attempt + 1}/${maxRetries}):`, error.message);
                await delay(retryDelay * Math.pow(2, attempt)); // 指数退避
            }
        }
    }

    throw lastError;
}

export class ApiClient {
    constructor() {
        this.baseUrl = CONSTANTS.API_BASE_URL;
    }

    /**
     * 通用的API请求方法
     * @param {string} endpoint - API端点
     * @param {Object} options - 请求选项
     * @param {number} customTimeout - 自定义超时时间（毫秒），如果不提供则使用默认值
     * @returns {Promise<any>} API响应
     */
    async request(endpoint, options = {}, customTimeout = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const requestOptions = {...defaultOptions, ...options};

        // 确定超时时间
        const timeout = customTimeout || CONSTANTS.CONNECTION_TIMEOUT;

        try {
            // 创建超时控制器
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(url, {
                ...requestOptions,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error(`请求超时（${timeout / 1000}秒），请检查网络连接`);
            }
            throw error;
        }
    }

    /**
     * 检查后端连接状态
     * @returns {Promise<boolean>} 连接状态
     */
    async checkConnection() {
        try {
            await this.request('/training/health');
            return true;
        } catch (error) {
            console.error('后端连接失败:', error);
            return false;
        }
    }

    /**
     * 获取训练状态
     * @returns {Promise<Object>} 训练状态信息
     */
    async getTrainingStatus() {
        return await this.request('/training/status');
    }

    /**
     * 启动训练
     * @param {Object} config - 训练配置
     * @returns {Promise<Object>} 启动结果
     */
    async startTraining(config) {
        console.log('=== API客户端发送请求 ===');
        console.log('请求配置:', config);
        console.log('请求体:', JSON.stringify(config));
        console.log('========================');

        return await this.request('/training/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * 停止训练 - 使用较长的超时时间
     * @returns {Promise<Object>} 停止结果
     */
    async stopTraining() {
        return await this.request('/training/stop', {
            method: 'POST'
        }, CONSTANTS.TRAINING_STOP_TIMEOUT); // 使用30秒超时
    }

    /**
     * 获取可用的Mesh列表
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Object>} Mesh列表
     */
    async getMeshList(subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/list?${params}`);
    }

    /**
     * 获取指定Mesh的信息
     * @param {string} meshName - Mesh名称
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Object>} Mesh信息
     */
    async getMeshInfo(meshName, subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/info/${meshName}?${params}`);
    }

    /**
     * 获取指定Mesh的边界数据（新增）
     * @param {string} meshName - Mesh名称
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Object>} Mesh边界数据
     */
    async getMeshBoundary(meshName, subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/boundary/${meshName}?${params}`);
    }

    /**
     * 检查Mesh API健康状态
     * @returns {Promise<Object>} 健康状态
     */
    async checkMeshHealth() {
        return await this.request('/mesh/health');
    }

    /**
     * 检查训练API健康状态
     * @returns {Promise<Object>} 健康状态
     */
    async checkTrainingHealth() {
        return await this.request('/training/health');
    }

    // ========== Checkpoint相关API ==========

    /**
     * 获取可用的Checkpoint列表
     * @returns {Promise<Object>} Checkpoint列表
     */
    async getCheckpointList() {
        return await this.request('/checkpoint/list');
    }

    /**
     * 获取指定Checkpoint的信息
     * @param {string} checkpointName - Checkpoint名称
     * @returns {Promise<Object>} Checkpoint信息
     */
    async getCheckpointInfo(checkpointName) {
        return await this.request(`/checkpoint/info/${checkpointName}`);
    }

    /**
     * 验证Checkpoint是否有效
     * @param {string} checkpointName - Checkpoint名称
     * @returns {Promise<Object>} 验证结果
     */
    async validateCheckpoint(checkpointName) {
        return await this.request(`/checkpoint/validate/${checkpointName}`);
    }

    /**
     * 删除指定的Checkpoint
     * @param {string} checkpointName - Checkpoint名称
     * @returns {Promise<Object>} 删除结果
     */
    async deleteCheckpoint(checkpointName) {
        return await this.request(`/checkpoint/delete/${checkpointName}`, {
            method: 'DELETE'
        });
    }

    /**
     * 从历史训练目录复制Checkpoint
     * @param {string} trainingId - 训练会话ID
     * @param {string} checkpointName - 目标Checkpoint名称（可选）
     * @returns {Promise<Object>} 复制结果
     */
    async copyCheckpointFromHistory(trainingId, checkpointName = null) {
        const body = {training_id: trainingId};
        if (checkpointName) {
            body.checkpoint_name = checkpointName;
        }

        return await this.request('/checkpoint/copy', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    /**
     * 检查Checkpoint API健康状态
     * @returns {Promise<Object>} 健康状态
     */
    async checkCheckpointHealth() {
        return await this.request('/checkpoint/health');
    }
}