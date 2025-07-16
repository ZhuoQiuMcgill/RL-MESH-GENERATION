/**
 * 强化学习网格生成训练管理系统 - 支持checkpoint的版本
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
        this.immediateUpdateTimer = null; // 新增：立即更新定时器

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
            if (isConnected) {
                await this.loadMeshList();
                await this.loadCheckpointList(); // 新增：加载checkpoint列表
            } else {
                this.uiController.logMessage('无法连接到后端服务器，请确保Flask应用正在运行在 http://localhost:5000', LOG_TYPES.ERROR);
            }

            this.uiController.updateButtonStates(false);
            this.uiController.logMessage('系统初始化完成', LOG_TYPES.INFO);
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

        // 刷新状态按钮
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshStatus());
        }

        // 清除日志按钮
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.uiController.clearLogs());
        }

        // Mesh选择变化 - 增强版本，支持预览
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', (e) => this.onMeshSelectionChange(e.target.value));
        }

        // 新增：Checkpoint模式切换
        const checkpointMode = document.getElementById('checkpoint-mode');
        if (checkpointMode) {
            checkpointMode.addEventListener('change', (e) => this.onCheckpointModeChange(e.target.checked));
        }

        // 新增：Checkpoint选择变化
        const checkpointSelect = document.getElementById('checkpoint-select');
        if (checkpointSelect) {
            checkpointSelect.addEventListener('change', (e) => this.onCheckpointSelectionChange(e.target.value));
        }

        // Canvas点击事件
        const canvas = document.getElementById('mesh-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * 检查后端连接状态
     * @returns {Promise<boolean>} 连接状态
     */
    async checkBackendConnection() {
        try {
            const connected = await this.apiClient.checkConnection();
            if (connected) {
                this.uiController.logMessage('后端连接正常', LOG_TYPES.SUCCESS);
            }
            return connected;
        } catch (error) {
            this.uiController.logMessage('后端连接失败: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * 加载可用的Mesh列表
     */
    async loadMeshList() {
        try {
            this.uiController.showLoading(true);

            const data = await withRetry(() => this.apiClient.getMeshList());

            this.uiController.populateMeshList(data.meshes || []);

            if (data.meshes && data.meshes.length > 0) {
                this.uiController.logMessage(`成功加载 ${data.meshes.length} 个Mesh文件`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('未找到可用的Mesh文件', LOG_TYPES.WARNING);
            }

        } catch (error) {
            console.error('加载Mesh列表失败:', error);
            this.uiController.showError('加载Mesh列表失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 新增：加载可用的Checkpoint列表
     */
    async loadCheckpointList() {
        try {
            const data = await withRetry(() => this.apiClient.getCheckpointList());

            this.uiController.populateCheckpointList(data.checkpoints || []);

            if (data.checkpoints && data.checkpoints.length > 0) {
                this.uiController.logMessage(`成功加载 ${data.checkpoints.length} 个Checkpoint文件`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('未找到可用的Checkpoint文件', LOG_TYPES.INFO);
            }

        } catch (error) {
            console.error('加载Checkpoint列表失败:', error);
            this.uiController.logMessage('加载Checkpoint列表失败: ' + error.message, LOG_TYPES.WARNING);
        }
    }

    /**
     * Mesh选择变化事件处理 - 增强版本，支持预览
     * @param {string} meshName - 选中的mesh名称
     */
    async onMeshSelectionChange(meshName) {
        if (!meshName) {
            this.uiController.hideMeshInfo();
            // 清空canvas，显示默认提示
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
            }
            return;
        }

        try {
            this.uiController.showLoading(true);

            // 同时获取mesh信息和边界数据
            const [info, boundaryData] = await Promise.all([
                this.apiClient.getMeshInfo(meshName),
                this.apiClient.getMeshBoundary(meshName)
            ]);

            // 更新UI信息
            this.uiController.showMeshInfo(info);
            this.uiController.logMessage(`选择了Mesh: ${meshName}`, LOG_TYPES.INFO);

            // 在canvas中渲染边界预览
            if (this.canvasRenderer && boundaryData.success) {
                this.canvasRenderer.renderBoundaryPreview(
                    boundaryData.boundary_vertices,
                    meshName
                );
                this.uiController.logMessage(
                    `已加载边界预览: ${boundaryData.vertex_count} 个顶点`,
                    LOG_TYPES.SUCCESS
                );
            } else if (!boundaryData.success) {
                this.uiController.logMessage(
                    `无法加载边界数据: ${boundaryData.error}`,
                    LOG_TYPES.WARNING
                );
            }

        } catch (error) {
            console.error('获取Mesh信息失败:', error);
            this.uiController.showError('获取Mesh信息失败: ' + error.message);
            this.uiController.hideMeshInfo();

            // 清空canvas
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
            }
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 新增：Checkpoint模式切换事件处理
     * @param {boolean} useCheckpoint - 是否使用checkpoint
     */
    onCheckpointModeChange(useCheckpoint) {
        this.uiController.showCheckpointSelection(useCheckpoint);

        if (useCheckpoint) {
            this.uiController.logMessage('已启用Checkpoint模式', LOG_TYPES.INFO);
        } else {
            this.uiController.logMessage('已禁用Checkpoint模式', LOG_TYPES.INFO);
            this.uiController.hideCheckpointInfo();
        }
    }

    /**
     * 新增：Checkpoint选择变化事件处理
     * @param {string} checkpointName - 选中的checkpoint名称
     */
    async onCheckpointSelectionChange(checkpointName) {
        if (!checkpointName) {
            this.uiController.hideCheckpointInfo();
            return;
        }

        try {
            this.uiController.showLoading(true);

            // 获取checkpoint详细信息
            const response = await this.apiClient.getCheckpointInfo(checkpointName);

            if (response.success && response.checkpoint_info) {
                this.uiController.showCheckpointInfo(response.checkpoint_info);
                this.uiController.logMessage(`选择了Checkpoint: ${checkpointName}`, LOG_TYPES.INFO);
            } else {
                this.uiController.showError('获取Checkpoint信息失败');
            }

        } catch (error) {
            console.error('获取Checkpoint信息失败:', error);
            this.uiController.showError('获取Checkpoint信息失败: ' + error.message);
            this.uiController.hideCheckpointInfo();
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 处理Canvas点击事件
     * @param {MouseEvent} event - 鼠标事件
     */
    handleCanvasClick(event) {
        if (!this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) {
            this.uiController.updateClickCoordinates(null);
            return;
        }

        // 获取鼠标相对于canvas的位置
        const rect = event.target.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;

        // 转换为世界坐标
        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);

        // 更新显示
        this.uiController.updateClickCoordinates(worldCoords);

        // 记录到日志
        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
        this.uiController.logMessage(`点击坐标: ${coordText}`, LOG_TYPES.INFO);
    }

    /**
     * 开始训练 - 修复版本，支持checkpoint，立即获取状态
     */
    async startTraining() {
        // 验证配置
        const validation = this.uiController.validateTrainingConfig();
        if (!validation.valid) {
            this.uiController.showError(validation.message);
            return;
        }

        const config = this.uiController.getTrainingConfig();

        try {
            this.uiController.showLoading(true);

            const result = await this.apiClient.startTraining(config);

            let successMessage = '训练已启动: ' + result.message;
            if (result.from_checkpoint && result.checkpoint_name) {
                successMessage += ` (继续训练自checkpoint: ${result.checkpoint_name})`;
            }

            this.uiController.logMessage(successMessage, LOG_TYPES.SUCCESS);

            this.isTraining = true;
            this.uiController.updateButtonStates(true);
            this.uiController.updateStatusIndicator(STATUS.RUNNING);

            // 修复：立即开始状态更新
            this.startPeriodicUpdate();

            // 新增：立即获取第一次状态更新，减少延迟
            this.scheduleImmediateUpdate();

        } catch (error) {
            console.error('启动训练失败:', error);
            this.uiController.showError('启动训练失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 新增：安排立即状态更新
     */
    scheduleImmediateUpdate() {
        // 清除之前的立即更新定时器
        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
        }

        // 500ms后立即获取一次状态，然后在1秒、2秒后再获取
        const immediateUpdates = [500, 1000, 2000];

        immediateUpdates.forEach((delay, index) => {
            setTimeout(async () => {
                if (this.isTraining) {
                    await this.updateTrainingStatus();
                    this.uiController.logMessage(`获取训练状态更新 #${index + 1}`, LOG_TYPES.INFO);
                }
            }, delay);
        });
    }

    /**
     * 停止训练
     */
    async stopTraining() {
        // 立即停止轮询和更新UI
        this.stopPeriodicUpdate();
        this.isTraining = false;
        this.uiController.updateButtonStates(false);
        this.uiController.updateStatusIndicator(STATUS.STOPPING);

        // 清除立即更新定时器
        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
        }

        try {
            this.uiController.showLoading(true);

            const result = await this.apiClient.stopTraining();
            this.uiController.logMessage('训练停止请求已发送: ' + result.message, LOG_TYPES.INFO);

        } catch (error) {
            console.error('停止训练失败:', error);
            this.uiController.showError('停止训练失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 刷新训练状态
     */
    async refreshStatus() {
        await this.updateTrainingStatus();
    }

    /**
     * 开始定期更新
     */
    startPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        const interval = this.uiController.getUpdateInterval();

        this.updateInterval = setInterval(async () => {
            await this.updateTrainingStatus();
        }, interval);

        // 立即执行一次更新
        this.updateTrainingStatus();
    }

    /**
     * 停止定期更新
     */
    stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }

        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
            this.immediateUpdateTimer = null;
        }
    }

    /**
     * 更新训练状态
     */
    async updateTrainingStatus() {
        try {
            const status = await this.apiClient.getTrainingStatus();
            this.handleStatusUpdate(status);
        } catch (error) {
            console.error('获取训练状态失败:', error);
            this.uiController.logMessage('获取训练状态失败: ' + error.message, LOG_TYPES.ERROR);
        }
    }

    /**
     * 处理状态更新
     * @param {Object} status - 状态数据
     */
    handleStatusUpdate(status) {
        // 更新运行状态
        this.isTraining = status.running;

        // 更新状态指示器
        this.uiController.updateStatusIndicator(status.status);

        // 更新统计数据
        if (status.stats) {
            this.uiController.updateTrainingStats(status.stats);
        }

        // 更新进度信息
        if (status.progress) {
            this.uiController.updateProgressInfo(status.progress);
        }

        // 更新渲染
        this.updateRendering();

        // 如果训练明确结束，停止定期更新
        const isFinished = !status.running || [STATUS.STOPPED, STATUS.COMPLETED, STATUS.ERROR].includes(status.status);
        if (isFinished && this.updateInterval) {
            this.stopPeriodicUpdate();
        }

        this.uiController.updateButtonStates(this.isTraining);
    }

    /**
     * 更新渲染
     */
    updateRendering() {
        if (!this.canvasRenderer) return;

        const renderData = this.uiController.getRenderData();

        if (renderData.meshData || renderData.boundaryData) {
            this.canvasRenderer.renderScene(
                renderData.meshData,
                renderData.boundaryData,
                renderData.refPointInfo
            );
        }
    }

    /**
     * 处理Canvas点击事件的节流版本
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);

    /**
     * 处理窗口大小变化 - 新增方法
     */
    handleResize() {
        if (this.canvasRenderer) {
            this.canvasRenderer.onResize();
        }
        this.uiController.logMessage('窗口大小已调整', LOG_TYPES.INFO);
    }

    /**
     * 新增：刷新Checkpoint列表
     */
    async refreshCheckpointList() {
        await this.loadCheckpointList();
    }

    /**
     * 获取应用程序状态
     * @returns {Object} 应用程序状态
     */
    getApplicationState() {
        return {
            isTraining: this.isTraining,
            hasUpdateInterval: !!this.updateInterval,
            canvasReady: !!this.canvasRenderer,
            uiReady: !!this.uiController
        };
    }

    /**
     * 销毁管理器，清理资源
     */
    destroy() {
        this.stopPeriodicUpdate();

        if (this.canvasRenderer) {
            this.canvasRenderer.destroy();
        }

        this.uiController.reset();

        // 清理事件监听器
        // 注：在实际应用中，应该存储事件监听器引用并在此处移除

        console.log('TrainingManager已销毁');
    }
}