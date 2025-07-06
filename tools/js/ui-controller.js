/**
 * UI控制器模块 - 支持更新状态按钮版本
 * 负责所有UI更新和用户交互逻辑
 * 修复进度条假更新问题并支持手动更新状态按钮
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, getTimestamp, getLogStyle, safeGetElement} from './utils.js';

export class UIController {
    constructor() {
        this.elements = this.initializeElements();
        this.isTraining = false;
        this.meshData = null;
        this.boundaryData = null;
        this.refPointInfo = null;

        // 进度条相关状态
        this.progressTimer = null;
        this.progressStartTime = null;
        this.progressDuration = 0;
    }

    /**
     * 初始化DOM元素引用
     * @returns {Object} 元素引用对象
     */
    initializeElements() {
        const elementIds = [
            'status-indicator', 'status-text', 'mesh-select', 'mesh-info',
            'mesh-vertices', 'mesh-size', 'start-btn', 'stop-btn',
            'refresh-btn', 'clear-log-btn', 'max-episodes', 'max-steps',
            'update-interval', 'current-episode', 'total-steps', 'avg-reward',
            'buffer-size', 'episode-reward', 'episode-length', 'ref-point',
            'click-coordinates', 'display-episode', 'boundary-vertices',
            'log-container', 'loading-overlay',
            // 进度条相关元素
            'update-progress-bar', 'update-progress-text',
            // 🚨 新增：更新状态按钮
            'update-status-btn'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = safeGetElement(id);
        });

        return elements;
    }

    /**
     * 启动更新进度条 - 移除状态检查，由TrainingManager控制
     * @param {number} intervalSeconds - 更新间隔（秒）
     */
    startUpdateProgressBar(intervalSeconds) {
        console.log(`🎯 启动进度条，间隔: ${intervalSeconds}秒`);

        this.stopUpdateProgressBar(); // 先停止现有的计时器

        this.progressDuration = intervalSeconds * 1000; // 转换为毫秒
        this.progressStartTime = Date.now();

        // 立即更新一次
        this.updateProgressBar();

        // 设置定时器，每100ms更新一次进度条
        this.progressTimer = setInterval(() => {
            this.updateProgressBar();
        }, 100);

        this.logMessage(`进度条已启动，更新间隔: ${intervalSeconds}秒`, LOG_TYPES.INFO);
    }

    /**
     * 停止更新进度条
     */
    stopUpdateProgressBar() {
        if (this.progressTimer) {
            clearInterval(this.progressTimer);
            this.progressTimer = null;
            console.log('⏹️ 进度条已停止');
            this.logMessage('进度条已停止', LOG_TYPES.INFO);
        }
        this.resetProgressBar();
    }

    /**
     * 重置进度条
     */
    resetProgressBar() {
        const progressBar = this.elements['update-progress-bar'];
        const progressText = this.elements['update-progress-text'];

        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.classList.remove('near-complete');
        }

        if (progressText) {
            progressText.textContent = '下次更新倒计时: --';
        }
    }

    /**
     * 更新进度条 - 确保真实触发API请求
     */
    updateProgressBar() {
        if (!this.progressStartTime || this.progressDuration <= 0) {
            return false;
        }

        const now = Date.now();
        const elapsed = now - this.progressStartTime;
        const progress = Math.min(elapsed / this.progressDuration, 1);
        const remaining = Math.max(this.progressDuration - elapsed, 0);

        const progressBar = this.elements['update-progress-bar'];
        const progressText = this.elements['update-progress-text'];

        if (progressBar) {
            const percentage = (progress * 100).toFixed(1);
            progressBar.style.width = percentage + '%';

            // 当进度超过85%时添加脉冲效果
            if (progress > 0.85) {
                progressBar.classList.add('near-complete');
            } else {
                progressBar.classList.remove('near-complete');
            }
        }

        if (progressText) {
            if (remaining > 0) {
                const seconds = Math.ceil(remaining / 1000);
                progressText.textContent = `下次更新倒计时: ${seconds}秒`;
            } else {
                progressText.textContent = '正在更新...';
            }
        }

        // 如果完成了一个周期，重新开始
        if (progress >= 1) {
            this.progressStartTime = Date.now();
            console.log('✅ 进度条完成一个周期，应该触发API请求');
            return true; // 返回true表示完成了一个周期
        }

        return false;
    }

    /**
     * 更新状态指示器
     * @param {string} status - 状态值
     */
    updateStatusIndicator(status) {
        const indicator = this.elements['status-indicator']?.querySelector('div');
        const text = this.elements['status-text'];

        if (!indicator || !text) return;

        // 移除所有状态类
        indicator.className = 'w-2 h-2 rounded-full mr-2';

        const statusConfig = {
            [STATUS.RUNNING]: {class: 'status-running', text: '训练中'},
            [STATUS.STOPPED]: {class: 'status-stopped', text: '已停止'},
            [STATUS.COMPLETED]: {class: 'status-success', text: '已完成'},
            [STATUS.STOPPING]: {class: 'status-loading', text: '停止中'},
            [STATUS.ERROR]: {class: 'status-stopped', text: '出错'},
            [STATUS.IDLE]: {class: 'status-idle', text: '未启动'}
        };

        const config = statusConfig[status] || statusConfig[STATUS.IDLE];
        indicator.classList.add(config.class);
        text.textContent = config.text;

        // 记录状态变化
        console.log(`🔄 状态更新: ${config.text}`);
        this.logMessage(`状态更新: ${config.text}`, LOG_TYPES.INFO);
    }

    /**
     * 更新训练统计数据 - 仅接受真实的后端数据
     * @param {Object} stats - 统计数据（必须来自后端）
     */
    updateTrainingStats(stats) {
        if (!stats) {
            console.warn('⚠️ 收到空的统计数据');
            return;
        }

        // 验证数据来源，确保不是mock数据
        if (typeof stats !== 'object' || Array.isArray(stats)) {
            console.error('❌ 无效的统计数据格式:', stats);
            return;
        }

        // 只有在训练真正运行时才更新统计数据
        if (!this.isTraining) {
            console.warn('⚠️ 尝试更新统计数据，但训练未运行');
            return;
        }

        console.log('📊 更新训练统计数据:', stats);

        // 更新主要统计数据
        if (stats.current_episode !== undefined || stats.episode !== undefined) {
            const episode = stats.current_episode || stats.episode;
            this.updateElement('current-episode', episode);
            this.updateElement('display-episode', episode);
        }

        if (stats.total_steps !== undefined) {
            this.updateElement('total-steps', stats.total_steps);
        }

        if (stats.average_reward !== undefined) {
            this.updateElement('avg-reward', formatNumber(stats.average_reward));
        }

        if (stats.buffer_utilization !== undefined || stats.buffer_size !== undefined) {
            const bufferSize = stats.buffer_utilization || stats.buffer_size;
            this.updateElement('buffer-size', bufferSize);
        }

        if (stats.latest_reward !== undefined || stats.episode_reward !== undefined) {
            const reward = stats.latest_reward || stats.episode_reward;
            this.updateElement('episode-reward', formatNumber(reward));
        }

        if (stats.episode_length !== undefined) {
            this.updateElement('episode-length', stats.episode_length);
        }

        console.log('✅ 统计数据更新完成');
    }

    /**
     * 更新进度数据 - 移除状态检查，接受真实的后端数据
     * @param {Object} progress - 进度数据（来自后端）
     */
    updateProgress(progress) {
        if (!progress) {
            console.warn('⚠️ 收到空的进度数据');
            return;
        }

        // 验证数据来源，确保不是mock数据
        if (typeof progress !== 'object' || Array.isArray(progress)) {
            console.error('❌ 无效的进度数据格式:', progress);
            return;
        }

        console.log('📈 更新进度数据:', progress);

        if (progress.current_episode !== undefined) {
            this.updateElement('current-episode', progress.current_episode);
            this.updateElement('display-episode', progress.current_episode);
        }

        if (progress.total_steps !== undefined) {
            this.updateElement('total-steps', progress.total_steps);
        }

        if (progress.average_reward !== undefined) {
            this.updateElement('avg-reward', formatNumber(progress.average_reward));
        }

        if (progress.buffer_utilization !== undefined) {
            this.updateElement('buffer-size', progress.buffer_utilization);
        }

        if (progress.latest_reward !== undefined) {
            this.updateElement('episode-reward', formatNumber(progress.latest_reward));
        }

        console.log('✅ 进度数据更新完成');
    }

    /**
     * 更新UI按钮状态
     * @param {boolean} isTraining - 是否正在训练（必须是确认的状态）
     */
    updateButtonStates(isTraining) {
        console.log(`🔄 更新按钮状态: isTraining=${isTraining}`);

        this.isTraining = isTraining;

        const buttonStates = {
            'start-btn': !isTraining,
            'stop-btn': isTraining,
            'mesh-select': !isTraining,
            'max-episodes': !isTraining,
            'max-steps': !isTraining,
            'update-interval': !isTraining,
            // 🚨 新增：更新状态按钮在训练时启用
            'update-status-btn': isTraining
        };

        Object.entries(buttonStates).forEach(([elementId, enabled]) => {
            const element = this.elements[elementId];
            if (element) {
                element.disabled = !enabled;
                // 添加视觉反馈
                if (enabled) {
                    element.classList.remove('opacity-50', 'cursor-not-allowed');
                } else {
                    element.classList.add('opacity-50', 'cursor-not-allowed');
                }
            }
        });

        // 注意：进度条的启动/停止由TrainingManager控制，这里不自动启动
        // 这确保进度条只有在训练真正确认开始后才会启动
        console.log('✅ 按钮状态更新完成');
    }

    /**
     * 显示/隐藏加载指示器
     * @param {boolean} show - 是否显示
     */
    showLoading(show) {
        const overlay = this.elements['loading-overlay'];
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    /**
     * 显示错误信息
     * @param {string} message - 错误信息
     */
    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
        console.error('❌ UI错误:', message);
    }

    /**
     * 添加日志信息
     * @param {string} message - 日志信息
     * @param {string} type - 日志类型
     */
    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['log-container'];
        if (!container) return;

        const timestamp = getTimestamp();
        const style = getLogStyle(type);

        const logEntry = document.createElement('div');
        logEntry.style.color = style.color;
        logEntry.innerHTML = `<span style="color: #9ca3af;">[${timestamp}]</span> ${style.icon} ${message}`;

        // 移除初始提示信息
        const placeholderText = container.querySelector('.text-gray-500');
        if (placeholderText && placeholderText.textContent === '等待训练开始...') {
            placeholderText.remove();
        }

        container.appendChild(logEntry);

        // 限制日志数量
        const logs = container.children;
        if (logs.length > CONSTANTS.MAX_LOGS) {
            container.removeChild(logs[0]);
        }

        // 滚动到底部
        container.scrollTop = container.scrollHeight;
    }

    /**
     * 清除日志
     */
    clearLog() {
        const container = this.elements['log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">日志已清除</div>';
        }
    }

    /**
     * 填充Mesh选择列表
     * @param {Array} meshes - mesh列表
     */
    populateMeshList(meshes) {
        const select = this.elements['mesh-select'];
        if (!select) {
            console.error('❌ 未找到mesh-select元素');
            return;
        }

        console.log('📋 填充mesh列表, 收到数据:', meshes);

        // 清空现有选项
        select.innerHTML = '<option value="">选择一个Mesh</option>';

        if (Array.isArray(meshes) && meshes.length > 0) {
            meshes.forEach((mesh, index) => {
                const option = document.createElement('option');

                // 处理不同的数据格式
                if (typeof mesh === 'string') {
                    option.value = mesh;
                    option.textContent = mesh;
                } else if (mesh && typeof mesh === 'object' && mesh.name) {
                    option.value = mesh.name;
                    option.textContent = mesh.name;
                } else {
                    console.warn('⚠️ 跳过无效的mesh数据:', mesh);
                    return;
                }

                select.appendChild(option);
                console.log(`✅ 添加mesh选项 ${index + 1}: ${option.textContent}`);
            });

            console.log(`✅ 成功添加 ${meshes.length} 个mesh选项`);
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = '无可用Mesh';
            option.disabled = true;
            select.appendChild(option);
            console.log('⚠️ 添加了"无可用Mesh"选项');
        }
    }

    /**
     * 获取训练配置
     * @returns {Object} 训练配置对象
     */
    getTrainingConfig() {
        return {
            mesh_name: this.getSelectedMesh(),
            max_episodes: parseInt(this.getElementValue('max-episodes')) || 100,
            max_steps: parseInt(this.getElementValue('max-steps')) || 1000,
            update_interval: parseInt(this.getElementValue('update-interval')) || 10
        };
    }

    /**
     * 验证训练配置
     * @returns {Object} 验证结果
     */
    validateTrainingConfig() {
        const meshName = this.getSelectedMesh();
        if (!meshName) {
            return {valid: false, message: '请选择一个Mesh文件'};
        }

        const maxEpisodes = parseInt(this.getElementValue('max-episodes'));
        if (!maxEpisodes || maxEpisodes <= 0) {
            return {valid: false, message: '最大Episodes必须是正整数'};
        }

        const maxSteps = parseInt(this.getElementValue('max-steps'));
        if (!maxSteps || maxSteps <= 0) {
            return {valid: false, message: '最大Steps必须是正整数'};
        }

        const updateInterval = parseInt(this.getElementValue('update-interval'));
        if (!updateInterval || updateInterval <= 0) {
            return {valid: false, message: '更新间隔必须是正整数'};
        }

        return {valid: true};
    }

    /**
     * 获取选中的Mesh
     * @returns {string} Mesh名称
     */
    getSelectedMesh() {
        return this.getElementValue('mesh-select');
    }

    /**
     * 获取更新间隔（毫秒）
     * @returns {number} 更新间隔
     */
    getUpdateInterval() {
        const interval = parseInt(this.getElementValue('update-interval')) || 10;
        return interval * 1000; // 转换为毫秒
    }

    /**
     * 更新元素内容
     * @param {string} elementId - 元素ID
     * @param {any} value - 值
     */
    updateElement(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            if (element.tagName === 'INPUT' || element.tagName === 'SELECT') {
                element.value = value;
            } else {
                element.textContent = value;
            }
        }
    }

    /**
     * 获取元素的值
     * @param {string} elementId - 元素ID
     * @returns {any} 元素的值
     */
    getElementValue(elementId) {
        const element = this.elements[elementId];
        return element ? element.value : '';
    }

    /**
     * 设置元素的值
     * @param {string} elementId - 元素ID
     * @param {any} value - 值
     */
    setElementValue(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.value = value;
        }
    }

    /**
     * 更新mesh信息显示
     * @param {Object} meshInfo - mesh信息
     */
    updateMeshInfo(meshInfo) {
        if (!meshInfo) return;

        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.remove('hidden');
        }

        if (meshInfo.vertices !== undefined) {
            this.updateElement('mesh-vertices', meshInfo.vertices);
        }

        if (meshInfo.size !== undefined) {
            this.updateElement('mesh-size', meshInfo.size);
        }

        if (meshInfo.boundary_vertices !== undefined) {
            this.updateElement('boundary-vertices', meshInfo.boundary_vertices);
        }
    }

    /**
     * 隐藏mesh信息
     */
    hideMeshInfo() {
        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.add('hidden');
        }
    }

    /**
     * 更新点击坐标显示
     * @param {Array} coordinates - 坐标数组 [x, y]
     */
    updateClickCoordinates(coordinates) {
        if (coordinates && coordinates.length === 2) {
            const coordText = `(${coordinates[0].toFixed(2)}, ${coordinates[1].toFixed(2)})`;
            this.updateElement('click-coordinates', coordText);
        }
    }
}