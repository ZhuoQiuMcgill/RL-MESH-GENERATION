/**
 * 训练管理器模块 - 修复真实状态更新版本
 * 彻底修复进度条假更新和按钮绑定错误的问题
 */

import {STATUS, LOG_TYPES, CONSTANTS, delay} from './utils.js';
import {ApiClient} from './api-client.js';
import {UIController} from './ui-controller.js';
import {CanvasRenderer} from './canvas-renderer.js';

export class TrainingManager {
    constructor() {
        this.apiClient = new ApiClient();
        this.uiController = new UIController();
        this.canvasRenderer = null; // 延迟初始化

        this.isTraining = false;
        this.progressCheckInterval = null;
        this.trainingVerificationTimeout = null;

        // 添加调试标记
        this.debugMode = true;
    }

    /**
     * 初始化训练管理器
     */
    async init() {
        console.log('初始化训练管理器...');

        try {
            // 先初始化Canvas渲染器
            await this.initializeCanvas();

            // 设置事件监听器
            this.setupEventListeners();

            // 检查后端连接
            const connected = await this.checkBackendConnection();
            if (connected) {
                this.uiController.logMessage('后端连接成功', LOG_TYPES.SUCCESS);

                // 加载初始数据
                await this.loadMeshList();
            } else {
                this.uiController.logMessage('后端连接失败，请检查服务是否启动', LOG_TYPES.ERROR);
            }

            console.log('训练管理器初始化完成');

        } catch (error) {
            console.error('初始化失败:', error);
            this.uiController.logMessage('初始化失败: ' + error.message, LOG_TYPES.ERROR);
        }
    }

    /**
     * 初始化Canvas渲染器
     */
    async initializeCanvas() {
        console.log('正在初始化Canvas...');

        try {
            // 创建CanvasRenderer实例
            this.canvasRenderer = new CanvasRenderer();

            // 尝试初始化，如果失败则等待重试
            let retryCount = 0;
            const maxRetries = 5;
            const retryDelay = 100;

            while (retryCount < maxRetries) {
                if (this.canvasRenderer.init()) {
                    console.log('Canvas初始化成功');
                    return;
                }

                console.log(`Canvas初始化失败，重试 ${retryCount + 1}/${maxRetries}`);
                retryCount++;

                if (retryCount < maxRetries) {
                    await delay(retryDelay);
                }
            }

            console.warn('Canvas初始化最终失败，将在没有Canvas的情况下继续');
            this.canvasRenderer = null;

        } catch (error) {
            console.error('Canvas初始化出错:', error);
            this.canvasRenderer = null;
        }
    }

    /**
     * 开始训练 - 修复重复启动问题
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
            this.uiController.logMessage('正在检查训练状态...', LOG_TYPES.INFO);

            // 首先检查训练是否已在运行
            const currentStatus = await this.safeApiCall(async () => {
                return await this.apiClient.getTrainingStatus();
            });

            if (currentStatus && currentStatus.running === true) {
                this.uiController.showError('训练已在运行中，请勿重复启动');
                this.uiController.logMessage('检测到训练正在运行，无需重复启动', LOG_TYPES.WARNING);
                return;
            }

            this.uiController.logMessage('正在启动训练...', LOG_TYPES.INFO);

            const config = this.uiController.getTrainingConfig();

            if (this.debugMode) {
                console.log('发送训练配置:', config);
            }

            // 发送启动请求
            const response = await this.safeApiCall(async () => {
                return await this.apiClient.startTraining(config);
            });

            if (this.debugMode) {
                console.log('训练启动API响应:', response);
            }

            // 按照API文档检查响应格式
            if (response && response.success === true && response.message === "training_started") {
                this.uiController.logMessage('训练启动请求已发送，等待后端确认...', LOG_TYPES.INFO);

                // 等待并验证训练真正开始
                const trainingStarted = await this.waitForTrainingToStart();

                if (trainingStarted) {
                    // 只有在确认训练真正开始后才更新UI状态
                    this.isTraining = true;
                    this.uiController.updateButtonStates(true);
                    this.uiController.updateStatusIndicator(STATUS.RUNNING);

                    // 开始真正的进度条和数据更新循环
                    this.startUpdateLoop();

                    this.uiController.logMessage('训练已确认开始', LOG_TYPES.SUCCESS);
                } else {
                    this.uiController.showError('训练启动失败：后端未能开始训练');
                    this.uiController.updateStatusIndicator(STATUS.ERROR);
                }
            } else {
                const errorMessage = response?.error || '启动训练失败';
                this.uiController.showError(errorMessage);
            }

        } catch (error) {
            console.error('启动训练失败:', error);
            this.uiController.showError('启动训练失败: ' + error.message);
            this.uiController.updateStatusIndicator(STATUS.ERROR);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 等待训练真正开始
     */
    async waitForTrainingToStart() {
        const maxWaitTime = 10000; // 最大等待10秒
        const checkInterval = 1000; // 每秒检查一次
        const startTime = Date.now();

        while (Date.now() - startTime < maxWaitTime) {
            try {
                if (this.debugMode) {
                    console.log('检查训练是否已开始...');
                }

                const status = await this.safeApiCall(async () => {
                    return await this.apiClient.getTrainingStatus();
                });

                if (this.debugMode) {
                    console.log('训练状态检查响应:', status);
                }

                // 检查训练是否真正在运行
                if (status && status.running === true) {
                    this.uiController.logMessage('后端确认训练已开始', LOG_TYPES.SUCCESS);
                    return true;
                }

                // 检查是否有错误状态
                if (status && status.status === 'error') {
                    this.uiController.logMessage('后端报告训练启动失败', LOG_TYPES.ERROR);
                    return false;
                }

                // 等待下一次检查
                await delay(checkInterval);
                this.uiController.logMessage(`等待训练开始... (${Math.ceil((Date.now() - startTime) / 1000)}s)`, LOG_TYPES.INFO);

            } catch (error) {
                console.error('检查训练状态失败:', error);
                await delay(checkInterval);
            }
        }

        this.uiController.logMessage('等待训练开始超时', LOG_TYPES.ERROR);
        return false;
    }

    /**
     * 停止训练
     */
    async stopTraining() {
        try {
            this.uiController.updateStatusIndicator(STATUS.STOPPING);
            this.uiController.logMessage('正在停止训练...', LOG_TYPES.INFO);

            const response = await this.safeApiCall(async () => {
                return await this.apiClient.stopTraining();
            });

            if (response && response.success === true && response.message === "stop_requested") {
                // 立即停止前端更新循环
                this.stopUpdateLoop();
                this.isTraining = false;
                this.uiController.updateButtonStates(false);
                this.uiController.updateStatusIndicator(STATUS.STOPPED);
                this.uiController.logMessage('训练已停止', LOG_TYPES.WARNING);
            } else {
                const errorMessage = response?.error || '停止训练失败';
                this.uiController.showError(errorMessage);
            }
        } catch (error) {
            console.error('停止训练失败:', error);
            this.uiController.showError('停止训练失败: ' + error.message);
        }
    }

    /**
     * 开始更新循环 - 确保真实发送API请求
     */
    startUpdateLoop() {
        console.log('🔄 开始真实更新循环');

        // 启动进度条
        const intervalSeconds = this.uiController.getUpdateInterval() / 1000;
        this.uiController.startUpdateProgressBar(intervalSeconds);

        // 设置进度条检查 - 关键修复
        this.setupProgressCheckInterval();

        // 立即获取一次数据
        this.fetchTrainingData();

        this.uiController.logMessage(`开始数据更新循环，间隔: ${intervalSeconds}秒`, LOG_TYPES.INFO);
    }

    /**
     * 停止更新循环
     */
    stopUpdateLoop() {
        console.log('⏹️ 停止更新循环');

        this.uiController.stopUpdateProgressBar();

        if (this.progressCheckInterval) {
            clearInterval(this.progressCheckInterval);
            this.progressCheckInterval = null;
        }

        if (this.trainingVerificationTimeout) {
            clearTimeout(this.trainingVerificationTimeout);
            this.trainingVerificationTimeout = null;
        }

        this.uiController.logMessage('数据更新循环已停止', LOG_TYPES.INFO);
    }

    /**
     * 设置事件监听器 - 修复按钮绑定
     */
    setupEventListeners() {
        console.log('🔗 设置事件监听器');

        // 开始训练按钮
        const startBtn = document.getElementById('start-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startTraining());
            console.log('✅ 绑定开始训练按钮');
        }

        // 停止训练按钮
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopTraining());
            console.log('✅ 绑定停止训练按钮');
        }

        // 刷新mesh列表按钮 - 确保正确绑定
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                console.log('🔄 手动刷新mesh列表');
                this.loadMeshList();
            });
            console.log('✅ 绑定刷新mesh列表按钮');
        }

        // 手动更新训练状态按钮 - 新增！
        const updateStatusBtn = document.getElementById('update-status-btn');
        if (updateStatusBtn) {
            updateStatusBtn.addEventListener('click', () => {
                console.log('🔄 手动更新训练状态');
                this.manualUpdateTrainingStatus();
            });
            console.log('✅ 绑定手动更新状态按钮');
        } else {
            console.warn('⚠️ 未找到update-status-btn元素');
        }

        // 清除日志按钮
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.uiController.clearLog());
            console.log('✅ 绑定清除日志按钮');
        }

        // Mesh选择变化
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', () => this.handleMeshSelection());
            console.log('✅ 绑定Mesh选择变化事件');
        }

        // Canvas点击事件
        const canvas = document.getElementById('mesh-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', (event) => this.handleCanvasClick(event));
            console.log('✅ 绑定Canvas点击事件');
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
            console.log('✅ 绑定更新间隔输入变化事件');
        }
    }

    /**
     * 手动更新训练状态 - 新增方法
     */
    async manualUpdateTrainingStatus() {
        this.uiController.logMessage('手动更新训练状态...', LOG_TYPES.INFO);

        try {
            await this.fetchTrainingData();
            this.uiController.logMessage('手动更新完成', LOG_TYPES.SUCCESS);
        } catch (error) {
            this.uiController.logMessage('手动更新失败: ' + error.message, LOG_TYPES.ERROR);
        }
    }

    /**
     * 重启进度条（当更新间隔改变时）
     */
    restartProgressBar(intervalSeconds) {
        console.log(`🔄 重启进度条，新间隔: ${intervalSeconds}秒`);
        this.uiController.startUpdateProgressBar(intervalSeconds);
        this.setupProgressCheckInterval();
    }

    /**
     * 设置进度条检查间隔 - 移除不必要的状态检查
     */
    setupProgressCheckInterval() {
        console.log('⏰ 设置进度条检查间隔');

        // 清除现有的检查间隔
        if (this.progressCheckInterval) {
            clearInterval(this.progressCheckInterval);
        }

        // 每100ms检查一次进度条是否完成，直接发送API请求
        this.progressCheckInterval = setInterval(() => {
            const completed = this.uiController.updateProgressBar();
            if (completed) {
                console.log('📡 进度条完成，发送训练状态请求...');
                this.fetchTrainingData().catch(error => {
                    console.error('获取训练数据失败:', error);
                });
            }
        }, 100);

        console.log('✅ 进度条检查间隔已设置');
    }

    /**
     * 检查后端连接状态
     */
    async checkBackendConnection() {
        try {
            console.log('🔍 开始检查后端连接...');

            // 使用综合健康检查
            const healthStatus = await this.apiClient.checkAllHealth();

            if (healthStatus.training.healthy) {
                this.uiController.logMessage('Training API连接正常', LOG_TYPES.SUCCESS);
            } else if (healthStatus.training.error) {
                this.uiController.logMessage(`Training API连接失败: ${healthStatus.training.error}`, LOG_TYPES.WARNING);
            }

            if (healthStatus.mesh.healthy) {
                this.uiController.logMessage('Mesh API连接正常', LOG_TYPES.SUCCESS);
            } else if (healthStatus.mesh.error) {
                this.uiController.logMessage(`Mesh API连接失败: ${healthStatus.mesh.error}`, LOG_TYPES.WARNING);
            }

            if (healthStatus.overall) {
                return true;
            } else {
                this.uiController.logMessage('所有后端API连接失败', LOG_TYPES.ERROR);
                return false;
            }

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

            console.log('📡 发送Mesh列表请求...');

            const meshes = await this.safeApiCall(async () => {
                return await this.apiClient.getMeshList();
            });

            if (Array.isArray(meshes) && meshes.length > 0) {
                this.uiController.populateMeshList(meshes);
                this.uiController.logMessage(`成功加载 ${meshes.length} 个Mesh`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('未找到可用的Mesh文件', LOG_TYPES.WARNING);
                this.uiController.populateMeshList([]);
            }
        } catch (error) {
            console.error('加载Mesh列表失败:', error);
            this.uiController.showError('加载Mesh列表失败: ' + error.message);
            this.uiController.populateMeshList([]);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 处理Mesh选择
     */
    async handleMeshSelection() {
        const meshName = this.uiController.getSelectedMesh();
        if (!meshName) {
            this.uiController.hideMeshInfo();
            return;
        }

        try {
            this.uiController.showLoading(true);

            const meshInfo = await this.safeApiCall(async () => {
                return await this.apiClient.getMeshInfo(meshName);
            });

            if (meshInfo && meshInfo.exists === true && !meshInfo.error) {
                this.uiController.updateMeshInfo({
                    vertices: meshInfo.vertices || 0,
                    size: meshInfo.file_size,
                    boundary_vertices: meshInfo.boundary_vertices || 0
                });

                this.uiController.logMessage(`已加载Mesh: ${meshName}`, LOG_TYPES.SUCCESS);
            } else {
                const errorMessage = meshInfo?.error || '获取Mesh信息失败';
                this.uiController.showError(`加载Mesh失败: ${errorMessage}`);
            }
        } catch (error) {
            this.uiController.showError('加载Mesh失败: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * 获取训练数据 - 移除状态检查，确保API请求发送
     */
    async fetchTrainingData() {
        try {
            console.log('📡 发送训练状态请求: GET /training/status');

            const data = await this.safeApiCall(async () => {
                return await this.apiClient.getTrainingStatus();
            });

            console.log('📨 收到训练状态响应:', data);

            if (data && typeof data === 'object') {
                // 检查训练是否仍在运行
                if (data.running === false) {
                    this.uiController.logMessage('检测到训练已停止', LOG_TYPES.WARNING);
                    this.handleTrainingCompleted(data);
                    return;
                }

                // 更新统计数据
                if (data.stats && typeof data.stats === 'object') {
                    if (this.debugMode) {
                        console.log('📊 更新统计数据:', data.stats);
                    }
                    this.uiController.updateTrainingStats(data.stats);
                }

                // 更新进度数据
                if (data.progress && typeof data.progress === 'object') {
                    if (this.debugMode) {
                        console.log('📈 更新进度数据:', data.progress);
                    }
                    this.uiController.updateProgress(data.progress);
                }

                // 更新Canvas渲染（如果Canvas可用）
                if (this.canvasRenderer && this.canvasRenderer.isInitialized()) {
                    if (data.mesh_data || data.boundary_vertices) {
                        this.canvasRenderer.renderScene(
                            data.mesh_data || {},
                            data.boundary_vertices || [],
                            data.ref_point_info
                        );
                    }
                }

                // 记录成功
                if (this.debugMode) {
                    console.log('✅ 训练状态更新成功');
                }

            } else {
                console.error('❌ 获取训练数据失败: 响应格式无效', data);
            }
        } catch (error) {
            console.error('❌ 获取训练数据失败:', error);
            this.uiController.logMessage('无法获取训练数据: ' + error.message, LOG_TYPES.WARNING);
        }
    }

    /**
     * 处理训练完成
     */
    handleTrainingCompleted(finalData) {
        this.stopUpdateLoop();
        this.isTraining = false;
        this.uiController.updateButtonStates(false);

        if (finalData && finalData.status === 'completed') {
            this.uiController.updateStatusIndicator(STATUS.COMPLETED);
            this.uiController.logMessage('训练已完成', LOG_TYPES.SUCCESS);
        } else {
            this.uiController.updateStatusIndicator(STATUS.STOPPED);
            this.uiController.logMessage('训练已停止', LOG_TYPES.WARNING);
        }
    }

    /**
     * 处理Canvas点击事件
     */
    handleCanvasClick(event) {
        if (this.canvasRenderer && this.canvasRenderer.isInitialized()) {
            const coordinates = this.canvasRenderer.getClickCoordinates(event);
            this.uiController.updateClickCoordinates(coordinates);
        }
    }

    /**
     * 安全的API调用包装器
     */
    async safeApiCall(apiCall) {
        try {
            return await apiCall();
        } catch (error) {
            console.error('API调用失败:', error);
            throw error;
        }
    }

    /**
     * 调试功能：加载mesh列表
     */
    debugLoadMeshList() {
        console.log('=== 调试：加载Mesh列表 ===');
        this.loadMeshList().then(() => {
            console.log('调试：Mesh列表加载完成');
        }).catch(error => {
            console.error('调试：Mesh列表加载失败:', error);
        });
    }

    /**
     * 调试功能：强制更新训练状态
     */
    debugUpdateTrainingStatus() {
        console.log('=== 调试：强制更新训练状态 ===');
        this.fetchTrainingData().then(() => {
            console.log('调试：训练状态更新完成');
        }).catch(error => {
            console.error('调试：训练状态更新失败:', error);
        });
    }

    /**
     * 调试功能：检查API健康状态
     */
    async debugApiHealth() {
        console.log('=== 调试：API健康检查 ===');
        try {
            const healthStatus = await this.apiClient.checkAllHealth();
            console.log('健康状态:', healthStatus);
        } catch (error) {
            console.error('健康检查失败:', error);
        }
    }
}