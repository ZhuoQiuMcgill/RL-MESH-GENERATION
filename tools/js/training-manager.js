/**
 * 强化学习网格生成训练管理系统
 * 主要的TrainingManager类，整合所有功能模块
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {ApiClient, withErrorHandling, withRetry} from './api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';
import {UIController} from './ui-controller.js';

export class TrainingManager {
    constructor() {
        // 初始化各个模块
        this.apiClient = new ApiClient();
        this.uiController = new UIController();
        this.canvasRenderer = null; // 延迟初始化

        // 状态管理
        this.isTraining = false;
        this.updateInterval = null;
        this.progressCheckInterval = null;

        // 创建带错误处理的API方法
        this.safeApiCall = withErrorHandling.bind(this);

        this.init();
    }

    /**
     * 初始化应用程序
     */
    async init() {
        try {
            this.setupCanvas();
            this.bindEvents();

            // 检查后端连接
            const isConnected = await this.checkBackendConnection();

            // 无论连接状态如何，都尝试加载mesh列表
            this.uiController.logMessage('尝试加载Mesh列表...', LOG_TYPES.INFO);
            await this.loadMeshList();

            if (!isConnected) {
                this.uiController.logMessage('无法连接到后端服务器，请确保Flask应用正在运行在 http://localhost:5000', LOG_TYPES.ERROR);
                this.uiController.logMessage('如果后端未运行，请启动后端服务器，然后点击"刷新Mesh列表"按钮', LOG_TYPES.INFO);
            }

            this.uiController.updateButtonStates(false);
            this.uiController.logMessage('系统初始化完成', LOG_TYPES.INFO);

            // 添加调试信息
            console.log('系统初始化完成, 后端连接状态:', isConnected);

        } catch (error) {
            console.error('初始化失败:', error);
            this.uiController.showError('系统初始化失败: ' + error.message);
        }
    }

    /**
     * 设置Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('mesh-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
        } else {
            console.error('未找到Canvas元素');
        }
    }

    /**
     * 绑定事件监听器
     */
    bindEvents() {
        // 开始训练按钮
        const startBtn = document.getElementById('start-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startTraining());
        }

        // 停止训练按钮
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopTraining());
        }

        // 刷新按钮
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                console.log('手动刷新mesh列表');
                this.loadMeshList();
            });
        }

        // 清除日志按钮
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.uiController.clearLog());
        }

        // Mesh选择变化
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', () => this.handleMeshSelection());
        }

        // Canvas点击事件
        const canvas = document.getElementById('mesh-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', (event) => this.handleCanvasClick(event));
        }

        // 监听更新间隔输入变化，实时更新进度条
        const updateIntervalInput = document.getElementById('update-interval');
        if (updateIntervalInput) {
            updateIntervalInput.addEventListener('input', () => {
                if (this.isTraining) {
                    const newInterval = parseInt(updateIntervalInput.value) || 10;
                    this.restartProgressBar(newInterval);
                }
            });
        }
    }

    /**
     * 重启进度条（当更新间隔改变时）
     * @param {number} intervalSeconds - 新的间隔秒数
     */
    restartProgressBar(intervalSeconds) {
        this.uiController.startUpdateProgressBar(intervalSeconds);
        this.setupProgressCheckInterval();
    }

    /**
     * 设置进度条检查间隔
     */
    setupProgressCheckInterval() {
        // 清除现有的检查间隔
        if (this.progressCheckInterval) {
            clearInterval(this.progressCheckInterval);
        }

        // 每100ms检查一次进度条是否完成
        this.progressCheckInterval = setInterval(() => {
            if (this.isTraining) {
                const completed = this.uiController.updateProgressBar();
                if (completed) {
                    // 进度条完成一个周期，触发数据更新
                    this.fetchTrainingData();
                }
            }
        }, 100);
    }

    /**
     * 检查后端连接状态
     * @returns {Promise<boolean>} 连接状态
     */
    async checkBackendConnection() {
        try {
            console.log('开始检查后端连接...');

            // 先检查mesh API
            try {
                const meshHealth = await this.apiClient.checkMeshHealth();
                console.log('Mesh API健康检查结果:', meshHealth);
                if (meshHealth && meshHealth.status === 'healthy') {
                    this.uiController.logMessage('Mesh API连接正常', LOG_TYPES.SUCCESS);
                    return true;
                }
            } catch (meshError) {
                console.warn('Mesh API健康检查失败:', meshError);
            }

            // 再检查训练API
            try {
                const trainingHealth = await this.apiClient.checkTrainingHealth();
                console.log('Training API健康检查结果:', trainingHealth);
                if (trainingHealth && trainingHealth.status === 'healthy') {
                    this.uiController.logMessage('Training API连接正常', LOG_TYPES.SUCCESS);
                    return true;
                }
            } catch (trainingError) {
                console.warn('Training API健康检查失败:', trainingError);
            }

            // 如果以上都失败，尝试基本连接测试
            const connected = await this.apiClient.checkConnection();
            if (connected) {
                this.uiController.logMessage('后端基本连接正常', LOG_TYPES.SUCCESS);
                return true;
            }

            this.uiController.logMessage('所有后端API连接失败', LOG_TYPES.ERROR);
            return false;

        } catch (error) {
            console.error('后端连接检查失败:', error);
            this.uiController.logMessage('后端连接失败: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * 加载Mesh列表
     */
    async loadMeshList() {
        try {
            this.uiController.showLoading(true);
            this.uiController.logMessage('正在加载Mesh列表...', LOG_TYPES.INFO);

            // 直接调用API方法，增加调试信息
            console.log('开始调用getMeshList API...');
            const data = await this.apiClient.getMeshList();
            console.log('API响应数据:', data);

            // 检查返回的数据格式
            if (data && data.meshes) {
                this.uiController.populateMeshList(data.meshes);
                this.uiController.logMessage(`成功加载 ${data.meshes.length} 个Mesh文件`, LOG_TYPES.SUCCESS);
                console.log('成功填充mesh列表:', data.meshes);
            } else if (data && Array.isArray(data)) {
                // 如果直接返回数组
                this.uiController.populateMeshList(data);
                this.uiController.logMessage(`成功加载 ${data.length} 个Mesh文件`, LOG_TYPES.SUCCESS);
                console.log('成功填充mesh列表(数组格式):', data);
            } else {
                console.warn('API返回的数据格式不正确:', data);
                this.uiController.logMessage('未找到可用的Mesh文件', LOG_TYPES.WARNING);
                this.uiController.populateMeshList([]);
            }

        } catch (error) {
            console.error('加载Mesh列表失败:', error);
            this.uiController.showError('加载Mesh列表失败: ' + error.message);
            this.uiController.logMessage('加载Mesh列表失败: ' + error.message, LOG_TYPES.ERROR);

            // 尝试添加一些默认的mesh选项用于测试
            const defaultMeshes = ['简单正方形', '三角形', '矩形', '五边形', '六边形'];
            this.uiController.populateMeshList(defaultMeshes);
            this.uiController.logMessage('已加载默认Mesh列表用于测试', LOG_TYPES.WARNING);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 处理Mesh选择
     */
    async handleMeshSelection() {
        const meshName = this.uiController.getElementValue('mesh-select');

        if (!meshName) {
            this.uiController.hideMeshInfo();
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
            }
            return;
        }

        try {
            this.uiController.showLoading(true);

            const meshInfo = await withRetry(() => this.apiClient.getMeshInfo(meshName));

            if (meshInfo && !meshInfo.error) {
                this.uiController.updateMeshInfo({
                    vertices: meshInfo.vertex_count,
                    size: meshInfo.file_size,
                    boundary_vertices: meshInfo.boundary_vertices || 0
                });

                this.uiController.logMessage(`已加载Mesh: ${meshName}`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.showError(`加载Mesh失败: ${meshInfo?.error || '未知错误'}`);
            }
        } catch (error) {
            this.uiController.showError('加载Mesh失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 开始训练
     */
    async startTraining() {
        // 验证配置
        const validation = this.uiController.validateTrainingConfig();
        if (!validation.valid) {
            this.uiController.showError(validation.message);
            return;
        }

        try {
            this.uiController.showLoading(true);

            const config = this.uiController.getTrainingConfig();
            const response = await this.safeApiCall(async () => {
                return await this.apiClient.startTraining(config);
            });

            if (response && !response.error) {
                this.isTraining = true;
                this.uiController.updateButtonStates(true);
                this.uiController.updateStatusIndicator(STATUS.RUNNING);

                // 开始进度条和数据更新循环
                this.startUpdateLoop();

                this.uiController.logMessage('训练已开始', LOG_TYPES.SUCCESS);
            } else {
                this.uiController.showError('启动训练失败: ' + (response?.error || '未知错误'));
            }
        } catch (error) {
            this.uiController.showError('启动训练失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 停止训练
     */
    async stopTraining() {
        try {
            this.uiController.updateStatusIndicator(STATUS.STOPPING);

            const response = await this.safeApiCall(async () => {
                return await this.apiClient.stopTraining();
            });

            if (response && !response.error) {
                this.stopUpdateLoop();
                this.isTraining = false;
                this.uiController.updateButtonStates(false);
                this.uiController.updateStatusIndicator(STATUS.STOPPED);
                this.uiController.logMessage('训练已停止', LOG_TYPES.WARNING);
            } else {
                this.uiController.showError('停止训练失败: ' + (response?.error || '未知错误'));
            }
        } catch (error) {
            this.uiController.showError('停止训练失败: ' + error.message);
        }
    }

    /**
     * 开始更新循环
     */
    startUpdateLoop() {
        // 启动进度条
        const intervalSeconds = this.uiController.getUpdateInterval() / 1000;
        this.uiController.startUpdateProgressBar(intervalSeconds);

        // 设置进度条检查
        this.setupProgressCheckInterval();

        // 立即获取一次数据
        this.fetchTrainingData();

        this.uiController.logMessage(`开始数据更新循环，间隔: ${intervalSeconds}秒`, LOG_TYPES.INFO);
    }

    /**
     * 停止更新循环
     */
    stopUpdateLoop() {
        this.uiController.stopUpdateProgressBar();

        if (this.progressCheckInterval) {
            clearInterval(this.progressCheckInterval);
            this.progressCheckInterval = null;
        }

        this.uiController.logMessage('数据更新循环已停止', LOG_TYPES.INFO);
    }

    /**
     * 获取训练数据
     */
    async fetchTrainingData() {
        if (!this.isTraining) return;

        try {
            const data = await this.safeApiCall(async () => {
                return await this.apiClient.getTrainingStatus();
            });

            if (data && !data.error) {
                // 更新统计数据
                if (data.stats) {
                    this.uiController.updateTrainingStats(data.stats);
                }

                // 更新进度数据
                if (data.progress) {
                    this.uiController.updateProgress(data.progress);
                }

                // 更新Canvas渲染
                if (this.canvasRenderer) {
                    if (data.mesh_data || data.boundary_vertices) {
                        this.canvasRenderer.renderScene(
                            data.mesh_data || {},
                            data.boundary_vertices || [],
                            data.ref_point_info
                        );
                    }
                }

                // 检查训练状态
                if (data.status) {
                    if (data.status === 'completed') {
                        this.handleTrainingCompleted();
                    } else if (data.status === 'error') {
                        this.handleTrainingError(data.error_message);
                    }
                }

            } else {
                console.warn('获取训练数据失败:', data?.error);
            }
        } catch (error) {
            console.error('获取训练数据错误:', error);
            // 不显示错误给用户，避免过多提示
        }
    }

    /**
     * 处理训练完成
     */
    handleTrainingCompleted() {
        this.stopUpdateLoop();
        this.isTraining = false;
        this.uiController.updateButtonStates(false);
        this.uiController.updateStatusIndicator(STATUS.COMPLETED);
        this.uiController.logMessage('训练已完成！', LOG_TYPES.SUCCESS);
    }

    /**
     * 处理训练错误
     * @param {string} errorMessage - 错误消息
     */
    handleTrainingError(errorMessage) {
        this.stopUpdateLoop();
        this.isTraining = false;
        this.uiController.updateButtonStates(false);
        this.uiController.updateStatusIndicator(STATUS.ERROR);
        this.uiController.logMessage(`训练出错: ${errorMessage}`, LOG_TYPES.ERROR);
    }

    /**
     * 处理Canvas点击
     * @param {MouseEvent} event - 点击事件
     */
    handleCanvasClick(event) {
        if (!this.canvasRenderer || !this.canvasRenderer.currentTransform) {
            return;
        }

        const rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // 将屏幕坐标转换为世界坐标
        const worldCoords = this.canvasRenderer.screenToWorld(x, y);

        if (worldCoords) {
            this.uiController.updateClickCoordinates([worldCoords.x, worldCoords.y]);
        }
    }

    /**
     * 销毁方法
     */
    destroy() {
        this.stopUpdateLoop();
        this.isTraining = false;

        if (this.canvasRenderer) {
            this.canvasRenderer.clearCanvas();
        }

        this.uiController.logMessage('系统已关闭', LOG_TYPES.INFO);
    }

    /**
     * 调试方法 - 手动测试mesh列表加载
     */
    async debugLoadMeshList() {
        console.log('=== 开始调试mesh列表加载 ===');

        // 1. 检查DOM元素
        const selectElement = document.getElementById('mesh-select');
        console.log('mesh-select元素:', selectElement);

        // 2. 直接测试API调用
        try {
            console.log('测试API调用...');
            const response = await fetch('http://localhost:5000/mesh/list');
            console.log('原始API响应状态:', response.status);
            const data = await response.json();
            console.log('原始API响应数据:', data);
        } catch (error) {
            console.error('直接API调用失败:', error);
        }

        // 3. 测试通过API客户端调用
        try {
            console.log('通过API客户端测试...');
            const data = await this.apiClient.getMeshList();
            console.log('API客户端返回数据:', data);
        } catch (error) {
            console.error('API客户端调用失败:', error);
        }

        // 4. 强制添加测试数据
        console.log('强制添加测试数据...');
        const testMeshes = ['test1', 'test2', 'test3'];
        this.uiController.populateMeshList(testMeshes);

        console.log('=== 调试完成 ===');
    }
}