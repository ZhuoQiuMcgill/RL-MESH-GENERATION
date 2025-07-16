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
        this.checkpoints = []; // 存储checkpoint列表
    }

    /**
     * 初始化DOM元素引用
     * @returns {Object} 元素引用对象
     */
    initializeElements() {
        const elementIds = [
            'status-indicator', 'status-text', 'mesh-select', 'mesh-info',
            'mesh-vertices', 'mesh-size', 'start-btn', 'stop-btn',
            'refresh-btn', 'clear-log-btn', 'max-timesteps', 'max-steps',
            'update-interval', 'description', 'current-episode', 'total-steps', 'avg-reward',
            'buffer-size', 'episode-reward', 'episode-length', 'ref-point',
            'click-coordinates', 'display-episode', 'display-total-steps', 'boundary-vertices',
            'log-container', 'loading-overlay',
            // 新增的checkpoint相关元素
            'checkpoint-mode', 'checkpoint-select', 'checkpoint-info', 'checkpoint-details'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = safeGetElement(id);
        });

        return elements;
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

        // 更新基础统计信息
        this.updateElement('current-episode', stats.episode || 0);
        this.updateElement('display-episode', stats.episode || 0);
        this.updateElement('total-steps', stats.total_steps || 0);
        this.updateElement('display-total-steps', stats.total_steps || 0);
        this.updateElement('avg-reward', formatNumber(stats.average_reward));
        this.updateElement('buffer-size', stats.buffer_size || 0);
        this.updateElement('episode-reward', formatNumber(stats.episode_reward));
        this.updateElement('episode-length', stats.episode_length || 0);
        this.updateElement('boundary-vertices', stats.boundary_vertices || 0);

        // 更新参考点信息
        if (stats.reference_point_info && stats.reference_point_info.ref_vertex) {
            const [rx, ry] = stats.reference_point_info.ref_vertex;
            this.updateElement('ref-point', `(${formatNumber(rx)}, ${formatNumber(ry)})`);
            this.refPointInfo = stats.reference_point_info;
        } else {
            this.updateElement('ref-point', 'N/A');
        }

        // 更新详细统计信息（如果statsContainer存在）
        const statsContainer = document.getElementById('stats-container');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <span>Episode: ${stats.episode || 'N/A'}</span>
                <span>总步数: ${stats.total_steps || 'N/A'}</span>
                <span>Episode奖励: ${formatNumber(stats.episode_reward)}</span>
                <span>平均奖励: ${formatNumber(stats.average_reward)}</span>
                <span>Episode长度: ${stats.episode_length || 'N/A'}</span>
                <span>边界顶点: ${stats.boundary_vertices || 'N/A'}</span>
                <span>Buffer大小: ${stats.buffer_size || 'N/A'}</span>
                <span>Actor Loss: ${formatNumber(stats.recent_actor_loss)}</span>
                <span>Critic Loss: ${formatNumber(stats.recent_critic_loss)}</span>
                <span>Alpha: ${formatNumber(stats.current_alpha)}</span>
            `;
        }

        // 更新mesh和boundary数据
        if (stats.mesh_data) {
            this.meshData = stats.mesh_data;
        }
        if (stats.boundary_vertices_data) {
            this.boundaryData = stats.boundary_vertices_data;
        }
    }

    /**
     * 更新进度信息
     * @param {Object} progress - 进度数据
     */
    updateProgressInfo(progress) {
        if (!progress) return;

        if (progress.current_episode !== undefined) {
            this.updateElement('current-episode', progress.current_episode);
            this.updateElement('display-episode', progress.current_episode);
        }

        if (progress.total_steps !== undefined) {
            this.updateElement('total-steps', progress.total_steps);
            this.updateElement('display-total-steps', progress.total_steps);
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
            'max-timesteps': !isTraining,
            'max-steps': !isTraining,
            'update-interval': !isTraining,
            'description': !isTraining,
            // 新增的checkpoint相关控件
            'checkpoint-mode': !isTraining,
            'checkpoint-select': !isTraining
        };

        Object.entries(buttonStates).forEach(([elementId, enabled]) => {
            const element = this.elements[elementId];
            if (element) {
                element.disabled = !enabled;
            }
        });
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
        if (!select) return;

        // 清空现有选项
        select.innerHTML = '<option value="">选择一个Mesh</option>';

        if (Array.isArray(meshes) && meshes.length > 0) {
            meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh;
                option.textContent = mesh;
                select.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = '未找到可用的Mesh文件';
            select.appendChild(option);
        }
    }

    /**
     * 填充Checkpoint选择列表
     * @param {Array} checkpoints - checkpoint列表
     */
    populateCheckpointList(checkpoints) {
        const select = this.elements['checkpoint-select'];
        if (!select) return;

        this.checkpoints = checkpoints || [];

        // 清空现有选项
        select.innerHTML = '<option value="">选择一个Checkpoint</option>';

        if (Array.isArray(checkpoints) && checkpoints.length > 0) {
            checkpoints.forEach(checkpoint => {
                const option = document.createElement('option');
                option.value = checkpoint.name;

                // 显示checkpoint名称和相关信息
                const displayText = `${checkpoint.name} (${checkpoint.modified_datetime}, ${checkpoint.file_size_mb}MB)`;
                option.textContent = displayText;

                // 如果checkpoint无效，禁用该选项
                if (!checkpoint.is_valid) {
                    option.disabled = true;
                    option.textContent += ' [无效]';
                }

                select.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = '未找到可用的Checkpoint文件';
            select.appendChild(option);
        }
    }

    /**
     * 显示Mesh信息
     * @param {Object} info - mesh信息
     */
    showMeshInfo(info) {
        if (!info) return;

        this.updateElement('mesh-vertices', info.vertex_count || 0);
        this.updateElement('mesh-size', info.file_size || 0);

        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.remove('hidden');
        }
    }

    /**
     * 隐藏Mesh信息
     */
    hideMeshInfo() {
        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.add('hidden');
        }
    }

    /**
     * 显示Checkpoint信息
     * @param {Object} info - checkpoint信息
     */
    showCheckpointInfo(info) {
        if (!info) return;

        const infoDiv = this.elements['checkpoint-info'];
        const detailsDiv = this.elements['checkpoint-details'];

        if (infoDiv) {
            infoDiv.classList.remove('hidden');
        }

        if (detailsDiv) {
            detailsDiv.innerHTML = `
                <div class="text-xs text-gray-600 space-y-1">
                    <div>训练步数: ${info.training_timesteps.toLocaleString()}</div>
                    <div>学习率: ${info.learning_rate}</div>
                    <div>文件大小: ${info.file_size_mb} MB</div>
                    <div>修改时间: ${info.modified_datetime}</div>
                    <div>有效性: ${info.is_valid ? '✓ 有效' : '✗ 无效'}</div>
                    ${info.has_replay_buffer ? '<div>包含经验回放缓冲区</div>' : ''}
                </div>
            `;
        }
    }

    /**
     * 隐藏Checkpoint信息
     */
    hideCheckpointInfo() {
        const infoDiv = this.elements['checkpoint-info'];
        if (infoDiv) {
            infoDiv.classList.add('hidden');
        }
    }

    /**
     * 控制Checkpoint选择区域的显示/隐藏
     * @param {boolean} show - 是否显示
     */
    showCheckpointSelection(show) {
        const checkpointSelect = this.elements['checkpoint-select'];
        const checkpointInfo = this.elements['checkpoint-info'];

        if (checkpointSelect) {
            checkpointSelect.style.display = show ? 'block' : 'none';
        }

        if (!show && checkpointInfo) {
            checkpointInfo.classList.add('hidden');
        }
    }

    /**
     * 记录日志消息
     * @param {string} message - 消息内容
     * @param {string} type - 消息类型
     */
    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['log-container'];
        if (!container) return;

        const timestamp = getTimestamp();
        const style = getLogStyle(type);

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.style.color = style.color;
        logEntry.innerHTML = `<span style="color: #9CA3AF;">[${timestamp}]</span> ${style.icon} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // 限制日志条数
        while (container.children.length > CONSTANTS.MAX_LOGS) {
            container.removeChild(container.firstChild);
        }
    }

    /**
     * 清除日志
     */
    clearLogs() {
        const container = this.elements['log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">日志已清除</div>';
        }
    }

    /**
     * 更新点击坐标显示
     * @param {Array} coords - 世界坐标 [x, y]
     */
    updateClickCoordinates(coords) {
        if (!coords || !Array.isArray(coords) || coords.length !== 2) {
            this.updateElement('click-coordinates', '无变换数据');
            return;
        }

        const coordText = `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`;
        this.updateElement('click-coordinates', coordText);
    }

    /**
     * 获取训练配置 - 基于timestep控制，支持checkpoint
     * @returns {Object} 训练配置
     */
    getTrainingConfig() {
        // 获取所有输入值
        const maxTimestepsValue = this.getElementValue('max-timesteps');
        const maxStepsValue = this.getElementValue('max-steps');
        const descriptionValue = this.getElementValue('description');

        // 获取checkpoint相关配置 - 修复版本
        const checkpointModeElement = document.getElementById('checkpoint-mode');
        const useCheckpoint = checkpointModeElement ? checkpointModeElement.checked : false;

        const rawName = this.getElementValue('checkpoint-select').trim();
        const checkpointName = rawName !== '' ? rawName : null;

        // 添加调试日志
        console.log('Checkpoint mode element:', checkpointModeElement);
        console.log('Use checkpoint:', useCheckpoint);
        console.log('Selected checkpoint:', checkpointName);

        let maxTimesteps = null;
        let maxSteps = null;

        // 安全地解析max_timesteps（主要控制参数）
        if (maxTimestepsValue && maxTimestepsValue.trim() !== '') {
            const parsed = parseInt(maxTimestepsValue.trim());
            if (!isNaN(parsed) && parsed > 0) {
                maxTimesteps = parsed;
            }
        }

        // 安全地解析max_steps
        if (maxStepsValue && maxStepsValue.trim() !== '') {
            const parsed = parseInt(maxStepsValue.trim());
            if (!isNaN(parsed) && parsed > 0) {
                maxSteps = parsed;
            }
        }

        const config = {
            mesh_name: this.getElementValue('mesh-select'),
            max_timesteps: maxTimesteps,
            max_steps: maxSteps,
            description: descriptionValue && descriptionValue.trim() !== '' ? descriptionValue.trim() : null
        };

        // 如果使用checkpoint，添加checkpoint配置
        if (useCheckpoint && checkpointName) {
            config.checkpoint_name = checkpointName;
            config.from_checkpoint = !!useCheckpoint;
        }

        return config;
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
     * 验证训练配置 - 基于timestep控制，支持checkpoint
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

        // 验证checkpoint（如果选择了使用checkpoint）
        const checkpointModeElement = document.getElementById('checkpoint-mode');
        const useCheckpoint = checkpointModeElement && checkpointModeElement.checked;

        if (useCheckpoint) {
            if (!config.checkpoint_name) {
                return {
                    valid: false,
                    message: '启用checkpoint模式时必须选择一个checkpoint'
                };
            }

            // 检查选中的checkpoint是否有效
            const selectedCheckpoint = this.checkpoints.find(cp => cp.name === config.checkpoint_name);
            if (!selectedCheckpoint || !selectedCheckpoint.is_valid) {
                return {
                    valid: false,
                    message: '选择的checkpoint无效'
                };
            }
        }

        // 主要验证：max_timesteps
        if (!config.max_timesteps) {
            return {
                valid: false,
                message: '请指定最大训练步数'
            };
        }

        if (config.max_timesteps && config.max_timesteps < 1000) {
            return {
                valid: false,
                message: '最大训练步数应至少为1000'
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
        // 如果初始化时没拿到元素，再即时查一次并写回缓存
        let element = this.elements[elementId];
        if (!element) {
            element = document.getElementById(elementId);
            if (element) this.elements[elementId] = element;   // 补进缓存
        }
        const value = element ? element.value : '';

        // 添加调试日志
        if (elementId === 'checkpoint-select') {
            console.log(`getElementValue(${elementId}):`, {
                element: element,
                value: value,
                directValue: document.getElementById(elementId)?.value
            });
        }

        return value;
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
     * 显示错误状态
     * @param {string} message - 错误消息
     */
    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
        this.updateStatusIndicator(STATUS.ERROR);
    }

    /**
     * 显示成功状态
     * @param {string} message - 成功消息
     */
    showSuccess(message) {
        this.logMessage(message, LOG_TYPES.SUCCESS);
    }

    /**
     * 显示警告状态
     * @param {string} message - 警告消息
     */
    showWarning(message) {
        this.logMessage(message, LOG_TYPES.WARNING);
    }

    /**
     * 重置UI到初始状态
     */
    reset() {
        this.isTraining = false;
        this.meshData = null;
        this.boundaryData = null;
        this.refPointInfo = null;
        this.checkpoints = [];

        this.updateStatusIndicator(STATUS.IDLE);
        this.updateButtonStates(false);
        this.showLoading(false);
        this.clearLogs();
        this.updateClickCoordinates(null);
        this.hideCheckpointInfo();
    }
}