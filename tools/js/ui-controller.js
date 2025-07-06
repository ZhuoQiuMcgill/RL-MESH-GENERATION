/**
 * UI控制器模块
 * 负责所有UI更新和用户交互逻辑
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
            // 新增进度条相关元素
            'update-progress-bar', 'update-progress-text'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = safeGetElement(id);
        });

        return elements;
    }

    /**
     * 启动更新进度条
     * @param {number} intervalSeconds - 更新间隔（秒）
     */
    startUpdateProgressBar(intervalSeconds) {
        this.stopUpdateProgressBar(); // 先停止现有的计时器

        this.progressDuration = intervalSeconds * 1000; // 转换为毫秒
        this.progressStartTime = Date.now();

        // 立即更新一次
        this.updateProgressBar();

        // 设置定时器，每100ms更新一次进度条
        this.progressTimer = setInterval(() => {
            this.updateProgressBar();
        }, 100);
    }

    /**
     * 停止更新进度条
     */
    stopUpdateProgressBar() {
        if (this.progressTimer) {
            clearInterval(this.progressTimer);
            this.progressTimer = null;
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
     * 更新进度条
     */
    updateProgressBar() {
        if (!this.progressStartTime || this.progressDuration <= 0) {
            return;
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
    }

    /**
     * 更新训练统计数据
     * @param {Object} stats - 统计数据
     */
    updateTrainingStats(stats) {
        if (!stats) return;

        // 更新主要统计数据
        if (stats.current_episode !== undefined) {
            this.updateElement('current-episode', stats.current_episode);
            this.updateElement('display-episode', stats.current_episode);
        }

        if (stats.total_steps !== undefined) {
            this.updateElement('total-steps', stats.total_steps);
        }

        if (stats.average_reward !== undefined) {
            this.updateElement('avg-reward', formatNumber(stats.average_reward));
        }

        if (stats.buffer_utilization !== undefined) {
            this.updateElement('buffer-size', stats.buffer_utilization);
        }

        if (stats.latest_reward !== undefined) {
            this.updateElement('episode-reward', formatNumber(stats.latest_reward));
        }

        if (stats.episode_length !== undefined) {
            this.updateElement('episode-length', stats.episode_length);
        }
    }

    /**
     * 更新进度数据
     * @param {Object} progress - 进度数据
     */
    updateProgress(progress) {
        if (!progress) return;

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
    }

    /**
     * 更新UI按钮状态
     * @param {boolean} isTraining - 是否正在训练
     */
    updateButtonStates(isTraining) {
        this.isTraining = isTraining;

        const buttonStates = {
            'start-btn': !isTraining,
            'stop-btn': isTraining,
            'mesh-select': !isTraining,
            'max-episodes': !isTraining,
            'max-steps': !isTraining,
            'update-interval': !isTraining
        };

        Object.entries(buttonStates).forEach(([elementId, enabled]) => {
            const element = this.elements[elementId];
            if (element) {
                element.disabled = !enabled;
            }
        });

        // 根据训练状态管理进度条
        if (isTraining) {
            const interval = this.getUpdateInterval() / 1000; // 转换为秒
            this.startUpdateProgressBar(interval);
        } else {
            this.stopUpdateProgressBar();
        }
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
     * 填充Mesh选择列表
     * @param {Array} meshes - mesh列表
     */
    populateMeshList(meshes) {
        const select = this.elements['mesh-select'];
        if (!select) {
            console.error('未找到mesh-select元素');
            return;
        }

        console.log('开始填充mesh列表, 收到数据:', meshes);

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
                    console.warn('跳过无效的mesh数据:', mesh);
                    return;
                }

                select.appendChild(option);
                console.log(`添加mesh选项 ${index + 1}: ${option.textContent}`);
            });

            console.log(`成功添加 ${meshes.length} 个mesh选项`);
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = '无可用Mesh';
            option.disabled = true;
            select.appendChild(option);
            console.log('添加了"无可用Mesh"选项');
        }

        // 验证选项是否正确添加
        console.log('当前select元素的选项数量:', select.options.length);
        for (let i = 0; i < select.options.length; i++) {
            console.log(`选项 ${i}: ${select.options[i].textContent} (value: ${select.options[i].value})`);
        }
    }

    /**
     * 显示错误消息
     * @param {string} message - 错误消息
     */
    showError(message) {
        console.error('UI错误:', message);
        this.logMessage(message, LOG_TYPES.ERROR);

        // 可选：显示一个更明显的错误提示
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        // 3秒后自动移除
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 3000);
    }

    /**
     * 记录消息到日志
     * @param {string} message - 消息内容
     * @param {string} type - 消息类型
     */
    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['log-container'];
        if (!container) {
            console.warn('未找到log-container元素');
            return;
        }

        const timestamp = getTimestamp();
        const logStyle = getLogStyle(type);

        const logEntry = document.createElement('div');
        logEntry.style.color = logStyle.color;
        logEntry.style.fontWeight = logStyle.fontWeight || 'normal';
        logEntry.style.marginBottom = '2px';
        logEntry.innerHTML = `[${timestamp}] ${logStyle.icon || ''} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // 限制日志条目数量
        const maxEntries = 50;
        while (container.children.length > maxEntries) {
            container.removeChild(container.firstChild);
        }

        // 同时输出到控制台
        console.log(`[${type}] ${message}`);
    }

    /**
     * 清除日志
     */
    clearLog() {
        const container = this.elements['log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">日志已清除</div>';
        }
        console.log('日志已清除');
    }

    /**
     * 更新参考点坐标
     * @param {Array} coords - 坐标数组 [x, y]
     */
    updateReferencePointCoordinates(coords) {
        if (!coords || coords.length !== 2) {
            this.updateElement('ref-point', 'N/A');
            return;
        }

        const coordText = `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`;
        this.updateElement('ref-point', coordText);
    }

    /**
     * 更新点击坐标
     * @param {Array} coords - 坐标数组 [x, y]
     */
    updateClickCoordinates(coords) {
        if (!coords || coords.length !== 2) {
            this.updateElement('click-coordinates', '无变换数据');
            return;
        }

        const coordText = `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`;
        this.updateElement('click-coordinates', coordText);
    }

    /**
     * 获取训练配置
     * @returns {Object} 训练配置
     */
    getTrainingConfig() {
        return {
            mesh_name: this.getElementValue('mesh-select'),
            max_episodes: parseInt(this.getElementValue('max-episodes')) || null,
            max_steps: parseInt(this.getElementValue('max-steps')) || null
        };
    }

    /**
     * 获取更新间隔
     * @returns {number} 更新间隔（毫秒）
     */
    getUpdateInterval() {
        const interval = parseInt(this.getElementValue('update-interval')) || 10;
        return interval * 1000; // 转换为毫秒
    }

    /**
     * 验证训练配置
     * @returns {Object} 验证结果 {valid: boolean, message: string}
     */
    validateTrainingConfig() {
        const config = this.getTrainingConfig();

        if (!config.mesh_name) {
            return {
                valid: false,
                message: '请先选择一个Mesh文件'
            };
        }

        if (config.max_episodes && config.max_episodes < 1) {
            return {
                valid: false,
                message: '最大Episodes数必须大于0'
            };
        }

        if (config.max_steps && config.max_steps < 1) {
            return {
                valid: false,
                message: '每Episode最大步数必须大于0'
            };
        }

        return {valid: true};
    }

    /**
     * 获取mesh和boundary数据
     * @returns {Object} 包含mesh和boundary数据的对象
     */
    getRenderData() {
        return {
            meshData: this.meshData,
            boundaryData: this.boundaryData,
            refPointInfo: this.refPointInfo
        };
    }

    /**
     * 更新单个元素的文本内容
     * @param {string} elementId - 元素ID
     * @param {any} value - 值
     */
    updateElement(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * 获取元素的值
     * @param {string} elementId - 元素ID
     * @returns {string} 元素值
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
}