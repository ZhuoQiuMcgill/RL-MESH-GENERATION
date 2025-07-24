/**
 * Reinforcement Learning Mesh Generation Training Management System - Checkpoint Support Version
 * Main TrainingManager class that integrates all functional modules
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {ApiClient, withErrorHandling, withRetry} from './api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';
import {UIController} from './ui-controller.js';

export class TrainingManager {
    constructor() {
        // Initialize all modules
        this.apiClient = new ApiClient();
        this.uiController = new UIController();
        this.canvasRenderer = null; // Delayed initialization

        // State management
        this.isTraining = false;
        this.updateInterval = null;
        this.immediateUpdateTimer = null; // New: immediate update timer

        // Create API methods with error handling
        this.safeApiCall = withErrorHandling.bind(this);

        this.init();
    }

    /**
     * Initialize application
     */
    async init() {
        try {
            this.setupCanvas();
            this.bindEvents();

            // Check backend connection
            const isConnected = await this.checkBackendConnection();
            if (isConnected) {
                await this.loadMeshList();
                await this.loadCheckpointList(); // New: load checkpoint list
            } else {
                this.uiController.logMessage('Cannot connect to backend server. Ensure the Flask app is running at http://localhost:5000', LOG_TYPES.ERROR);
            }

            this.uiController.updateButtonStates(false);
            this.uiController.logMessage('System initialization completed', LOG_TYPES.INFO);
        } catch (error) {
            console.error('Initialization failed:', error);
            this.uiController.showError('System initialization failed: ' + error.message);
        }
    }

    /**
     * Setup Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('mesh-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
        } else {
            console.error('Canvas element not found');
        }
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Start training button
        const startBtn = document.getElementById('start-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startTraining());
        }

        // Stop training button
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopTraining());
        }

        // Refresh status button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshStatus());
        }

        // Clear log button
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.uiController.clearLogs());
        }

        // Mesh selection change - enhanced version with preview support
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', (e) => this.onMeshSelectionChange(e.target.value));
        }

        // New: Checkpoint mode toggle
        const checkpointMode = document.getElementById('checkpoint-mode');
        if (checkpointMode) {
            checkpointMode.addEventListener('change', (e) => this.onCheckpointModeChange(e.target.checked));
        }

        // New: Checkpoint selection change
        const checkpointSelect = document.getElementById('checkpoint-select');
        if (checkpointSelect) {
            checkpointSelect.addEventListener('change', (e) => this.onCheckpointSelectionChange(e.target.value));
        }

        // Canvas click event
        const canvas = document.getElementById('mesh-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * Check backend connection status
     * @returns {Promise<boolean>} Connection status
     */
    async checkBackendConnection() {
        try {
            const connected = await this.apiClient.checkConnection();
            if (connected) {
                this.uiController.logMessage('Backend connection successful', LOG_TYPES.SUCCESS);
            }
            return connected;
        } catch (error) {
            this.uiController.logMessage('Backend connection failed: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * Load available mesh list
     */
    async loadMeshList() {
        try {
            this.uiController.showLoading(true);

            const data = await withRetry(() => this.apiClient.getMeshList());

            this.uiController.populateMeshList(data.meshes || []);

            if (data.meshes && data.meshes.length > 0) {
                this.uiController.logMessage(`Loaded ${data.meshes.length} mesh files`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('No mesh files found', LOG_TYPES.WARNING);
            }

        } catch (error) {
            console.error('Failed to load mesh list:', error);
            this.uiController.showError('Failed to load mesh list: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * New: Load available checkpoint list
     */
    async loadCheckpointList() {
        try {
            const data = await withRetry(() => this.apiClient.getCheckpointList());

            this.uiController.populateCheckpointList(data.checkpoints || []);

            if (data.checkpoints && data.checkpoints.length > 0) {
                this.uiController.logMessage(`Loaded ${data.checkpoints.length} checkpoint files`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('No checkpoint files found', LOG_TYPES.INFO);
            }

        } catch (error) {
            console.error('Failed to load checkpoint list:', error);
            this.uiController.logMessage('Failed to load checkpoint list: ' + error.message, LOG_TYPES.WARNING);
        }
    }

    /**
     * Mesh selection change event handler - enhanced version with preview support
     * @param {string} meshName - Selected mesh name
     */
    async onMeshSelectionChange(meshName) {
        if (!meshName) {
            this.uiController.hideMeshInfo();
            // Clear canvas and show default prompt
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
            }
            return;
        }

        try {
            this.uiController.showLoading(true);

            // Get mesh info and boundary data simultaneously
            const [info, boundaryData] = await Promise.all([
                this.apiClient.getMeshInfo(meshName),
                this.apiClient.getMeshBoundary(meshName)
            ]);

            // Update UI info
            this.uiController.showMeshInfo(info);
            this.uiController.logMessage(`Selected mesh: ${meshName}`, LOG_TYPES.INFO);

            // Render boundary preview in canvas
            if (this.canvasRenderer && boundaryData.success) {
                this.canvasRenderer.renderBoundaryPreview(
                    boundaryData.boundary_vertices,
                    meshName
                );
                this.uiController.logMessage(
                    `Loaded boundary preview: ${boundaryData.vertex_count} vertices`,
                    LOG_TYPES.SUCCESS
                );
            } else if (!boundaryData.success) {
                this.uiController.logMessage(
                    `Failed to load boundary data: ${boundaryData.error}`,
                    LOG_TYPES.WARNING
                );
            }

        } catch (error) {
            console.error('Failed to get mesh info:', error);
            this.uiController.showError('Failed to get mesh info: ' + error.message);
            this.uiController.hideMeshInfo();

            // Clear canvas
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
            }
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * New: Checkpoint mode toggle event handler
     * @param {boolean} useCheckpoint - Whether to use checkpoint
     */
    onCheckpointModeChange(useCheckpoint) {
        this.uiController.showCheckpointSelection(useCheckpoint);

        if (useCheckpoint) {
            this.uiController.logMessage('Checkpoint mode enabled', LOG_TYPES.INFO);
        } else {
            this.uiController.logMessage('Checkpoint mode disabled', LOG_TYPES.INFO);
            this.uiController.hideCheckpointInfo();
        }
    }

    /**
     * New: Checkpoint selection change event handler
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
                this.uiController.logMessage(`Selected checkpoint: ${checkpointName}`, LOG_TYPES.INFO);
            } else {
                this.uiController.showError('Failed to get checkpoint info');
            }

        } catch (error) {
            console.error('Failed to get checkpoint info:', error);
            this.uiController.showError('Failed to get checkpoint info: ' + error.message);
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
        this.uiController.logMessage(`Click coordinates: ${coordText}`, LOG_TYPES.INFO);
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

        // 添加调试日志
        console.log('=== 发送训练请求 ===');
        console.log('完整配置:', config);
        console.log('checkpoint_name:', config.checkpoint_name);
        console.log('====================');

        try {
            this.uiController.showLoading(true);

            const result = await this.apiClient.startTraining(config);

            let successMessage = 'Training started: ' + result.message;
            if (result.from_checkpoint && result.checkpoint_name) {
                successMessage += ` (continued from checkpoint: ${result.checkpoint_name})`;
            }

            this.uiController.logMessage(successMessage, LOG_TYPES.SUCCESS);

            this.isTraining = true;
            this.uiController.updateButtonStates(true);
            this.uiController.updateStatusIndicator(STATUS.RUNNING);

            this.startPeriodicUpdate();
            this.scheduleImmediateUpdate();

        } catch (error) {
            console.error('Failed to start training:', error);
            this.uiController.showError('Failed to start training: ' + error.message);
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
                    this.uiController.logMessage(`Training status update #${index + 1}`, LOG_TYPES.INFO);
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
            this.uiController.logMessage('Training stop request sent: ' + result.message, LOG_TYPES.INFO);

        } catch (error) {
            console.error('Failed to stop training:', error);
            this.uiController.showError('Failed to stop training: ' + error.message);
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
            console.error('Failed to get training status:', error);
            this.uiController.logMessage('Failed to get training status: ' + error.message, LOG_TYPES.ERROR);
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
        this.uiController.logMessage('Window resized', LOG_TYPES.INFO);
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

        console.log('TrainingManager destroyed');
    }
}