/**
 * 训练历史记录管理器
 * 负责管理历史训练数据的查看和可视化
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {HistoryApiClient} from './history-api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';

export class HistoryManager {
    constructor() {
        // 初始化各个模块
        this.apiClient = new HistoryApiClient();
        this.canvasRenderer = null; // 延迟初始化

        // 状态管理
        this.trainingList = [];
        this.currentTrainingId = null;
        this.currentTrainingInfo = null;
        this.currentEpisodeIndex = null;
        this.currentEpisodeData = null;

        // 防止重复请求的状态管理
        this.isLoadingHistory = false;
        this.isLoadingTraining = false;
        this.isLoadingEpisode = false;

        // DOM元素引用
        this.elements = this.initializeElements();

        this.init();
    }

    /**
     * 初始化DOM元素引用
     */
    initializeElements() {
        const elementIds = [
            'training-list', 'current-training-info', 'episode-navigation',
            'training-id-display', 'detail-length-display', 'best-episode-display',
            'episode-index-input', 'current-episode-display', 'episode-meta-info',
            'episode-reward-display', 'episode-length-display', 'episode-status-display',
            'actual-episode-number-display', 'boundary-vertices-count', 'mesh-vertices-count', 'ref-point-display',
            'click-coordinates-display', 'episode-data-container',
            'history-log-container', 'history-loading-overlay',
            // 按钮
            'refresh-history-btn', 'health-check-btn', 'goto-episode-btn',
            'goto-best-episode-btn', 'goto-last-episode-btn', 'prev-episode-btn',
            'next-episode-btn', 'clear-history-log-btn'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = document.getElementById(id);
        });

        return elements;
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
                await this.loadTrainingHistory();
            } else {
                this.logMessage('Cannot connect to history API server', LOG_TYPES.ERROR);
            }

            this.logMessage('History viewer initialized', LOG_TYPES.INFO);
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('System initialization failed: ' + error.message);
        }
    }

    /**
     * 设置Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('history-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
        } else {
            console.error('Canvas element not found');
        }
    }

    /**
     * 绑定事件监听器
     */
    bindEvents() {
        // 刷新历史记录按钮
        const refreshBtn = this.elements['refresh-history-btn'];
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshTrainingHistory());
        }

        // 健康检查按钮
        const healthBtn = this.elements['health-check-btn'];
        if (healthBtn) {
            healthBtn.addEventListener('click', () => this.checkHealthStatus());
        }

        // Episode导航按钮
        const gotoBtn = this.elements['goto-episode-btn'];
        if (gotoBtn) {
            gotoBtn.addEventListener('click', () => this.gotoEpisode());
        }

        const gotoBestBtn = this.elements['goto-best-episode-btn'];
        if (gotoBestBtn) {
            gotoBestBtn.addEventListener('click', () => this.gotoBestEpisode());
        }

        const gotoLastBtn = this.elements['goto-last-episode-btn'];
        if (gotoLastBtn) {
            gotoLastBtn.addEventListener('click', () => this.gotoLastEpisode());
        }

        const prevBtn = this.elements['prev-episode-btn'];
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.gotoPreviousEpisode());
        }

        const nextBtn = this.elements['next-episode-btn'];
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.gotoNextEpisode());
        }

        // 清除日志按钮
        const clearLogBtn = this.elements['clear-history-log-btn'];
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.clearLogs());
        }

        // Episode输入框回车事件
        const episodeInput = this.elements['episode-index-input'];
        if (episodeInput) {
            episodeInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.gotoEpisode();
                }
            });
        }

        // Canvas点击事件
        const canvas = document.getElementById('history-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * 检查后端连接状态
     */
    async checkBackendConnection() {
        try {
            const response = await this.apiClient.checkHistoryHealth();
            if (response.success && response.status === 'healthy') {
                this.logMessage('History API connection successful', LOG_TYPES.SUCCESS);
                return true;
            } else {
                this.logMessage('History API status abnormal: ' + (response.error || 'Unknown error'), LOG_TYPES.WARNING);
                return false;
            }
        } catch (error) {
            this.logMessage('History API connection failed: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * 加载训练历史记录列表
     */
    async loadTrainingHistory() {
        if (this.isLoadingHistory) {
            this.logMessage('Loading training history, please wait...', LOG_TYPES.WARNING);
            return;
        }
        try {
            this.isLoadingHistory = true;
            this.showLoading(true);

            const response = await this.apiClient.getTrainingHistoryList();

            if (response.success && response.training_ids) {
                this.trainingList = response.training_ids;
                this.updateTrainingList();
                this.logMessage(`Loaded ${response.count} training history entries`, LOG_TYPES.SUCCESS);
            } else {
                this.logMessage('No training history found: ' + (response.error || 'Unknown error'), LOG_TYPES.WARNING);
                this.trainingList = [];
                this.updateTrainingList();
            }

        } catch (error) {
            console.error('Failed to load training history:', error);
            this.showError('Failed to load training history: ' + error.message);
        } finally {
            this.showLoading(false);
            this.isLoadingHistory = false;
        }
    }

    /**
     * 刷新训练历史记录
     */
    async refreshTrainingHistory() {
        if (this.isLoadingHistory) {
            this.logMessage('Refreshing, please wait...', LOG_TYPES.WARNING);
            return;
        }
        this.logMessage('Refreshing training history...', LOG_TYPES.INFO);
        await this.loadTrainingHistory();
    }

    /**
     * 检查健康状态
     */
    async checkHealthStatus() {
        try {
            this.showLoading(true);
            const response = await this.apiClient.checkHistoryHealth();

            if (response.success && response.status === 'healthy') {
                this.logMessage(`Health check passed - available trainings: ${response.available_trainings}`, LOG_TYPES.SUCCESS);
                if (response.current_focus) {
                    this.logMessage(`Current focus: ${response.current_focus}`, LOG_TYPES.INFO);
                }
            } else {
                this.logMessage(`Service status abnormal: ${response.error || 'Unknown error'}`, LOG_TYPES.ERROR);
            }
        } catch (error) {
            this.logMessage(`健康检查失败: ${error.message}`, LOG_TYPES.ERROR);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 更新训练列表显示
     */
    updateTrainingList() {
        const container = this.elements['training-list'];
        if (!container) return;

        if (!this.trainingList || this.trainingList.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <div class="text-sm">暂无训练历史记录</div>
                    <button class="mt-2 text-primary hover:text-blue-600 text-xs" onclick="window.historyManager.refreshTrainingHistory()">
                        点击刷新
                    </button>
                </div>
            `;
            return;
        }

        // 生成训练项目HTML
        const itemsHTML = this.trainingList.map(trainingId => {
            const isActive = trainingId === this.currentTrainingId;
            const displayName = this.formatTrainingDisplayName(trainingId);

            return `
                <div class="training-item ${isActive ? 'active' : ''}" data-training-id="${trainingId}">
                    <div class="training-name">${displayName.name}</div>
                    <div class="training-meta">
                        <div>${displayName.timestamp}</div>
                        <div>Mesh: ${displayName.mesh}</div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = itemsHTML;

        // 绑定点击事件
        container.addEventListener('click', (e) => {
            const trainingItem = e.target.closest('.training-item');
            if (trainingItem) {
                const trainingId = trainingItem.dataset.trainingId;
                this.selectTraining(trainingId);
            }
        });
    }

    /**
     * 格式化训练显示名称
     */
    formatTrainingDisplayName(trainingId) {
        return this.apiClient.formatTrainingDisplayName(trainingId);
    }

    /**
     * 选择训练会话
     */
    async selectTraining(trainingId) {
        if (trainingId === this.currentTrainingId) return;

        try {
            this.showLoading(true);
            this.logMessage(`正在加载训练: ${trainingId}`, LOG_TYPES.INFO);

            // 获取训练信息
            const response = await this.apiClient.getTrainingInfo(trainingId);

            if (response.success) {
                this.currentTrainingId = trainingId;
                this.currentTrainingInfo = {
                    training_id: response.training_id,
                    detail_length: response.detail_length,
                    best_episode: response.best_episode
                };

                // 更新UI
                this.updateTrainingList(); // 刷新列表以显示选中状态
                this.updateTrainingInfo();
                this.showTrainingControls(true);

                // 默认加载最佳Episode
                await this.loadEpisode(this.currentTrainingInfo.best_episode);

                this.logMessage(`Training loaded successfully: ${response.detail_length} episodes`, LOG_TYPES.SUCCESS);
            } else {
                this.showError('Failed to load training info: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to select training:', error);
            this.showError('Failed to select training: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 更新训练信息显示
     */
    updateTrainingInfo() {
        if (!this.currentTrainingInfo) return;

        const displayName = this.formatTrainingDisplayName(this.currentTrainingInfo.training_id);

        this.updateElement('training-id-display', displayName.name);
        this.updateElement('detail-length-display', this.currentTrainingInfo.detail_length);
        this.updateElement('best-episode-display', this.currentTrainingInfo.best_episode);

        // 更新Episode输入框的最大值
        const episodeInput = this.elements['episode-index-input'];
        if (episodeInput) {
            episodeInput.max = this.currentTrainingInfo.detail_length - 1;
        }
    }

    /**
     * 显示/隐藏训练控制界面
     */
    showTrainingControls(show) {
        const infoDiv = this.elements['current-training-info'];
        const navDiv = this.elements['episode-navigation'];

        if (infoDiv) {
            if (show) {
                infoDiv.classList.remove('hidden');
            } else {
                infoDiv.classList.add('hidden');
            }
        }

        if (navDiv) {
            if (show) {
                navDiv.classList.remove('hidden');
            } else {
                navDiv.classList.add('hidden');
            }
        }
    }

    /**
     * 加载指定Episode数据
     */
    async loadEpisode(episodeIndex) {
        if (!this.currentTrainingId || !this.currentTrainingInfo) {
            this.showError('Please select a training session first');
            return;
        }

        if (episodeIndex < 0 || episodeIndex >= this.currentTrainingInfo.detail_length) {
            this.showError(`Episode index out of range: ${episodeIndex}`);
            return;
        }

        try {
            this.showLoading(true);
            this.logMessage(`正在加载Episode ${episodeIndex}...`, LOG_TYPES.INFO);

            const response = await this.apiClient.getEpisodeData(this.currentTrainingId, episodeIndex);

            if (response.success) {
                this.currentEpisodeIndex = episodeIndex;
                this.currentEpisodeData = response.episode_data;

                // 更新UI
                this.updateEpisodeInfo();
                this.updateEpisodeData();
                this.updateVisualization();

                // 更新Episode输入框值
                const episodeInput = this.elements['episode-index-input'];
                if (episodeInput) {
                    episodeInput.value = episodeIndex;
                }

                this.logMessage(`Episode ${episodeIndex} loaded successfully`, LOG_TYPES.SUCCESS);
            } else {
                this.showError('Failed to load episode: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to load episode:', error);
            this.showError('Failed to load episode: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 更新Episode信息显示
     */
    updateEpisodeInfo() {
        if (!this.currentEpisodeData) return;

        const {r: reward, l: length, is_completed, episode_number} = this.currentEpisodeData;

        this.updateElement('current-episode-display', `Episode ${this.currentEpisodeIndex}`);
        this.updateElement('actual-episode-number-display', episode_number || 'N/A');
        this.updateElement('episode-reward-display', formatNumber(reward));
        this.updateElement('episode-length-display', length);
        this.updateElement('episode-status-display', is_completed ? 'Completed' : 'Incomplete');

        // 显示元信息
        const metaInfo = this.elements['episode-meta-info'];
        if (metaInfo) {
            metaInfo.classList.remove('hidden');
        }

        // 更新边界和网格顶点数
        const boundaryVertices = this.currentEpisodeData.boundary_vertices_data || [];
        const meshData = this.currentEpisodeData.mesh_data || {};

        this.updateElement('boundary-vertices-count', boundaryVertices.length);
        this.updateElement('mesh-vertices-count', Object.keys(meshData).length);

        // 更新参考点信息
        const refPointInfo = this.currentEpisodeData.last_ref_point;
        if (refPointInfo && refPointInfo.ref_vertex) {
            const [rx, ry] = refPointInfo.ref_vertex;
            this.updateElement('ref-point-display', `(${formatNumber(rx)}, ${formatNumber(ry)})`);
        } else {
            this.updateElement('ref-point-display', 'N/A');
        }
    }

    /**
     * 更新Episode详细数据显示
     */
    updateEpisodeData() {
        if (!this.currentEpisodeData) return;

        // 更新Episode数据容器
        const episodeContainer = this.elements['episode-data-container'];
        if (episodeContainer) {
            const {r: reward, l: length, is_completed} = this.currentEpisodeData;

            const episode_number = this.currentEpisodeData.episode_number;

            episodeContainer.innerHTML = `
                <div class="episode-data-item">
                    <span class="episode-data-key">索引:</span>
                    <span class="episode-data-value">${this.currentEpisodeIndex}</span>
                </div>
                <div class="episode-data-item">
                    <span class="episode-data-key">实际Episode:</span>
                    <span class="episode-data-value">${episode_number || 'N/A'}</span>
                </div>
                <div class="episode-data-item">
                    <span class="episode-data-key">奖励值:</span>
                    <span class="episode-data-value">${formatNumber(reward)}</span>
                </div>
                <div class="episode-data-item">
                    <span class="episode-data-key">步数:</span>
                    <span class="episode-data-value">${length}</span>
                </div>
                <div class="episode-data-item">
                    <span class="episode-data-key">Status:</span>
                    <span class="episode-data-value ${is_completed ? 'text-green-600' : 'text-orange-600'}">
                        ${is_completed ? 'Completed' : 'Incomplete'}
                    </span>
                </div>
            `;
        }

    }

    /**
     * 更新可视化
     */
    updateVisualization() {
        if (!this.canvasRenderer || !this.currentEpisodeData) return;

        const meshData = this.currentEpisodeData.mesh_data;
        const boundaryVertices = this.currentEpisodeData.boundary_vertices_data;
        const refPointInfo = this.currentEpisodeData.last_ref_point;

        this.canvasRenderer.renderScene(meshData, boundaryVertices, refPointInfo);
    }

    /**
     * Episode导航方法
     */
    async gotoEpisode() {
        const episodeInput = this.elements['episode-index-input'];
        if (!episodeInput) return;

        const episodeIndex = parseInt(episodeInput.value);
        if (isNaN(episodeIndex)) {
            this.showError('Please enter a valid episode index');
            return;
        }

        await this.loadEpisode(episodeIndex);
    }

    async gotoBestEpisode() {
        if (!this.currentTrainingInfo) return;
        await this.loadEpisode(this.currentTrainingInfo.best_episode);
    }

    async gotoLastEpisode() {
        if (!this.currentTrainingInfo) return;
        await this.loadEpisode(this.currentTrainingInfo.detail_length - 1);
    }

    async gotoPreviousEpisode() {
        if (this.currentEpisodeIndex === null || this.currentEpisodeIndex <= 0) return;
        await this.loadEpisode(this.currentEpisodeIndex - 1);
    }

    async gotoNextEpisode() {
        if (!this.currentTrainingInfo || this.currentEpisodeIndex === null) return;
        if (this.currentEpisodeIndex >= this.currentTrainingInfo.detail_length - 1) return;
        await this.loadEpisode(this.currentEpisodeIndex + 1);
    }

    /**
     * Canvas点击事件处理
     */
    handleCanvasClick(event) {
        if (!this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) {
            this.updateElement('click-coordinates-display', 'No transform data');
            return;
        }

        const rect = event.target.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;

        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);
        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;

        this.updateElement('click-coordinates-display', coordText);
        this.logMessage(`点击坐标: ${coordText}`, LOG_TYPES.INFO);
    }

    /**
     * 工具方法
     */
    showLoading(show) {
        const overlay = this.elements['history-loading-overlay'];
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
    }

    updateElement(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.textContent = value;
        }
    }

    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['history-log-container'];
        if (!container) return;

        const timestamp = new Date().toLocaleTimeString();
        const colors = {
            [LOG_TYPES.SUCCESS]: '#059669',
            [LOG_TYPES.ERROR]: '#DC2626',
            [LOG_TYPES.WARNING]: '#D97706',
            [LOG_TYPES.INFO]: '#6B7280'
        };
        const icons = {
            [LOG_TYPES.SUCCESS]: '✓',
            [LOG_TYPES.ERROR]: '✗',
            [LOG_TYPES.WARNING]: '⚠',
            [LOG_TYPES.INFO]: 'ℹ'
        };

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.style.color = colors[type];
        logEntry.innerHTML = `<span style="color: #9CA3AF;">[${timestamp}]</span> ${icons[type]} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // 限制日志条数
        while (container.children.length > CONSTANTS.MAX_LOGS) {
            container.removeChild(container.firstChild);
        }
    }

    clearLogs() {
        const container = this.elements['history-log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">Logs cleared</div>';
        }
    }

    /**
     * 处理窗口大小变化
     */
    handleResize() {
        if (this.canvasRenderer) {
            this.canvasRenderer.onResize();
        }
        this.logMessage('Window resized', LOG_TYPES.INFO);
    }

    /**
     * Canvas点击事件的节流版本
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);

    /**
     * 销毁管理器，清理资源
     */
    destroy() {
        if (this.canvasRenderer) {
            this.canvasRenderer.destroy();
        }

        this.trainingList = [];
        this.currentTrainingId = null;
        this.currentTrainingInfo = null;
        this.currentEpisodeIndex = null;
        this.currentEpisodeData = null;

        console.log('HistoryManager destroyed');
    }
}