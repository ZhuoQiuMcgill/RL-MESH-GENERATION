/**
 * API客户端模块 - 严格按照API文档版本
 * 负责与后端API的所有通信
 * 严格遵循 api-doc.md 中定义的所有API路径和数据格式
 */

import {CONSTANTS, delay} from './utils.js';

export class ApiClient {
    constructor() {
        this.baseUrl = CONSTANTS.API_BASE_URL;
    }

    /**
     * 通用的API请求方法
     * @param {string} endpoint - API端点
     * @param {Object} options - 请求选项
     * @returns {Promise<any>} API响应
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const requestOptions = {...defaultOptions, ...options};

        try {
            // 创建超时控制器
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), CONSTANTS.CONNECTION_TIMEOUT);

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
                throw new Error('请求超时，请检查网络连接');
            }
            throw error;
        }
    }

    // ============ Training API ============

    /**
     * 启动训练
     * 路径: POST /training/start
     * @param {Object} config - 训练配置
     * @param {string} config.mesh_name - 要使用的网格文件名称
     * @param {string} config.subfolder - 网格文件所在的子文件夹，默认为"mesh"
     * @param {number} config.max_episodes - 最大训练轮数
     * @param {number} config.max_steps - 每轮最大步数
     * @returns {Promise<Object>} 启动结果
     *
     * 成功响应格式:
     * {
     *   "message": "training_started",
     *   "success": true,
     *   "config": { ... }
     * }
     */
    async startTraining(config) {
        return await this.request('/training/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * 停止训练
     * 路径: POST /training/stop
     * @returns {Promise<Object>} 停止结果
     *
     * 成功响应格式:
     * {
     *   "message": "stop_requested",
     *   "success": true
     * }
     */
    async stopTraining() {
        return await this.request('/training/stop', {
            method: 'POST'
        });
    }

    /**
     * 获取训练状态
     * 路径: GET /training/status
     * @returns {Promise<Object>} 训练状态信息
     *
     * 成功响应格式:
     * {
     *   "running": true,
     *   "status": "training",
     *   "stats": {
     *     "episode": 150,
     *     "total_steps": 75000,
     *     "episode_reward": 125.5,
     *     "average_reward": 98.2,
     *     "buffer_size": 10000
     *   },
     *   "progress": {
     *     "current_episode": 150,
     *     "total_steps": 75000,
     *     "latest_reward": 125.5,
     *     "average_reward": 98.2,
     *     "buffer_utilization": 10000
     *   },
     *   "timestamp": 1720285200.123
     * }
     */
    async getTrainingStatus() {
        return await this.request('/training/status');
    }

    /**
     * 训练健康检查
     * 路径: GET /training/health
     * @returns {Promise<Object>} 健康状态
     *
     * 成功响应格式:
     * {
     *   "status": "healthy",
     *   "service": "training-api",
     *   "manager_running": false,
     *   "timestamp": 1720285200.123
     * }
     */
    async checkTrainingHealth() {
        return await this.request('/training/health');
    }

    // ============ Mesh API ============

    /**
     * 获取可用的Mesh列表
     * 路径: GET /mesh/list
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Array>} Mesh名称数组（为了兼容现有代码，返回meshes数组）
     *
     * 原始API响应格式:
     * {
     *   "meshes": ["1", "simple_square", "triangle", "rectangle", "pentagon", "hexagon"],
     *   "count": 6
     * }
     *
     * 此方法返回meshes数组以保持向后兼容性
     */
    async getMeshList(subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        const response = await this.request(`/mesh/list?${params}`);

        // 根据API文档，响应格式是 {meshes: [...], count: 6}
        // 为了兼容现有前端代码，直接返回meshes数组
        if (response && Array.isArray(response.meshes)) {
            console.log(`成功获取${response.count}个mesh文件:`, response.meshes);
            return response.meshes;
        } else {
            console.warn('API响应格式不符合预期:', response);
            return [];
        }
    }

    /**
     * 获取原始Mesh列表响应（包含完整信息）
     * 路径: GET /mesh/list
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Object>} 完整的API响应
     */
    async getMeshListRaw(subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/list?${params}`);
    }

    /**
     * 获取指定Mesh的信息
     * 路径: GET /mesh/info/<mesh_name>
     * @param {string} meshName - Mesh名称
     * @param {string} subfolder - 子文件夹名称，默认为'mesh'
     * @returns {Promise<Object>} Mesh信息
     *
     * 成功响应格式:
     * {
     *   "name": "simple_square",
     *   "subfolder": "mesh",
     *   "exists": true,
     *   "vertex_count": 4,
     *   "file_size": 128,
     *   "error": null
     * }
     */
    async getMeshInfo(meshName, subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        const response = await this.request(`/mesh/info/${meshName}?${params}`);

        // 为了兼容现有前端代码，将API字段映射到前端期望的字段
        if (response) {
            return {
                name: response.name,
                subfolder: response.subfolder,
                exists: response.exists,
                vertices: response.vertex_count,  // 映射 vertex_count -> vertices
                file_size: response.file_size,
                boundary_vertices: response.vertex_count, // 暂时使用vertex_count作为boundary_vertices
                error: response.error
            };
        }

        return response;
    }

    /**
     * Mesh健康检查
     * 路径: GET /mesh/health
     * @returns {Promise<Object>} 健康状态
     *
     * 成功响应格式:
     * {
     *   "status": "healthy",
     *   "service": "mesh-api",
     *   "timestamp": 1720285200.123
     * }
     */
    async checkMeshHealth() {
        return await this.request('/mesh/health');
    }

    // ============ 兼容性和工具方法 ============

    /**
     * 检查后端连接状态（优先使用训练API）
     * @returns {Promise<boolean>} 连接状态
     */
    async checkConnection() {
        try {
            await this.checkTrainingHealth();
            return true;
        } catch (error) {
            console.error('后端连接失败:', error);
            return false;
        }
    }

    /**
     * 综合健康检查（检查所有API模块）
     * @returns {Promise<Object>} 详细的健康状态
     */
    async checkAllHealth() {
        const healthStatus = {
            training: {healthy: false, error: null},
            mesh: {healthy: false, error: null},
            overall: false
        };

        // 检查训练API
        try {
            const trainingHealth = await this.checkTrainingHealth();
            healthStatus.training.healthy = trainingHealth && trainingHealth.status === 'healthy';
        } catch (error) {
            healthStatus.training.error = error.message;
        }

        // 检查Mesh API
        try {
            const meshHealth = await this.checkMeshHealth();
            healthStatus.mesh.healthy = meshHealth && meshHealth.status === 'healthy';
        } catch (error) {
            healthStatus.mesh.error = error.message;
        }

        // 整体健康状态
        healthStatus.overall = healthStatus.training.healthy || healthStatus.mesh.healthy;

        return healthStatus;
    }

    /**
     * 验证训练配置参数
     * @param {Object} config - 训练配置
     * @returns {Object} 验证结果
     */
    validateTrainingConfig(config) {
        const errors = [];

        if (config.max_episodes && (!Number.isInteger(config.max_episodes) || config.max_episodes <= 0)) {
            errors.push('max_episodes必须是正整数');
        }

        if (config.max_steps && (!Number.isInteger(config.max_steps) || config.max_steps <= 0)) {
            errors.push('max_steps必须是正整数');
        }

        if (config.subfolder && typeof config.subfolder !== 'string') {
            errors.push('subfolder必须是字符串');
        }

        if (config.mesh_name && typeof config.mesh_name !== 'string') {
            errors.push('mesh_name必须是字符串');
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    /**
     * 格式化API错误信息
     * @param {Error} error - 错误对象
     * @returns {string} 格式化的错误信息
     */
    formatError(error) {
        if (error.message.includes('timeout') || error.message.includes('超时')) {
            return '请求超时，请检查网络连接';
        } else if (error.message.includes('Failed to fetch')) {
            return '网络连接失败，请检查服务器状态';
        } else if (error.message.includes('404')) {
            return '请求的资源不存在';
        } else if (error.message.includes('500')) {
            return '服务器内部错误';
        } else {
            return error.message || '未知错误';
        }
    }
}

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
export async function withRetry(apiCall, maxRetries = 3, retryDelay = 1000) {
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