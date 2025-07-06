/**
 * 强化学习网格生成训练管理系统
 * 前端控制脚本
 */

class TrainingManager {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:5000';
        this.isTraining = false;
        this.updateInterval = null;
        this.logContainer = document.getElementById('log-container');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statsContainer = document.getElementById('stats-container');
        this.progressContainer = document.getElementById('progress-container');
        this.canvas = document.getElementById('mesh-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.meshData = null;
        this.boundaryData = null;
        this.refPointInfo = null; // 新增属性

        this.init();
    }

    /**
     * 初始化应用程序
     */
    async init() {
        this.setupCanvas();
        this.bindEvents();

        // 检查后端连接
        const isConnected = await this.checkBackendConnection();
        if (isConnected) {
            await this.loadMeshList();
        } else {
            this.logMessage('无法连接到后端服务器，请确保Flask应用正在运行在 http://localhost:5000', 'error');
        }

        this.updateUI();
        this.logMessage('系统初始化完成', 'info');
    }

    /**
     * 检查后端连接状态
     */
    async checkBackendConnection() {
        try {
            // 创建超时控制器（兼容性处理）
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${this.apiBaseUrl}/training/status`, {
                method: 'GET',
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (response.ok) {
                this.logMessage('后端连接正常', 'success');
                return true;
            } else {
                this.logMessage(`后端响应异常: ${response.status}`, 'warning');
                return false;
            }
        } catch (error) {
            console.error('后端连接失败:', error);
            if (error.name === 'AbortError') {
                this.logMessage('连接超时，请检查后端服务器', 'error');
            } else {
                this.logMessage('后端连接失败，请检查服务器状态', 'error');
            }
            return false;
        }
    }

    /**
     * 设置Canvas
     */
    setupCanvas() {
        this.canvas = document.getElementById('mesh-canvas');
        this.ctx = this.canvas.getContext('2d');

        // 设置canvas大小
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // 清空canvas
        this.clearCanvas();
    }

    /**
     * 调整Canvas大小
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();

        // 考虑padding
        const padding = 32; // 对应p-4 = 16px * 2
        this.canvas.width = rect.width - padding;
        this.canvas.height = rect.height - padding;

        // 重新绘制，使用统一的渲染函数
        if (this.meshData || this.boundaryData) {
            this.renderMeshAndBoundary(this.meshData, this.boundaryData);
        } else {
            this.clearCanvas();
        }
    }

    /**
     * 清空Canvas
     */
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 绘制网格背景
        this.drawGrid();

        // 绘制提示文本
        this.ctx.fillStyle = '#9CA3AF';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            '等待训练开始...',
            this.canvas.width / 2,
            this.canvas.height / 2
        );

        // 清除缓存的数据
        this.meshData = null;
        this.boundaryData = null;
    }

    /**
     * 绘制网格背景
     */
    drawGrid() {
        const gridSize = 20;
        this.ctx.strokeStyle = '#F3F4F6';
        this.ctx.lineWidth = 0.5;

        // 垂直线
        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        // 水平线
        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    /**
     * 绑定事件监听器
     */
    bindEvents() {
        // 开始训练按钮
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startTraining();
        });

        // 停止训练按钮
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // 刷新状态按钮
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshStatus();
        });

        // 清除日志按钮
        document.getElementById('clear-log-btn').addEventListener('click', () => {
            this.clearLogs();
        });

        // Mesh选择变化
        document.getElementById('mesh-select').addEventListener('change', (e) => {
            this.onMeshSelectionChange(e.target.value);
        });
    }

    /**
     * 加载可用的Mesh列表
     */
    async loadMeshList() {
        try {
            this.showLoading(true);
            const response = await fetch(`${this.apiBaseUrl}/mesh/list`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const select = document.getElementById('mesh-select');

            // 清空现有选项
            select.innerHTML = '<option value="">选择一个Mesh</option>';

            // 添加mesh选项
            if (data.meshes && data.meshes.length > 0) {
                data.meshes.forEach(mesh => {
                    const option = document.createElement('option');
                    option.value = mesh;
                    option.textContent = mesh;
                    select.appendChild(option);
                });
                this.logMessage(`成功加载 ${data.meshes.length} 个Mesh文件`, 'success');
            } else {
                this.logMessage('未找到可用的Mesh文件', 'warning');
            }

        } catch (error) {
            console.error('加载Mesh列表失败:', error);
            this.logMessage('加载Mesh列表失败: ' + error.message, 'error');

            // 设置错误状态
            const select = document.getElementById('mesh-select');
            select.innerHTML = '<option value="">加载失败</option>';
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Mesh选择变化事件处理
     */
    async onMeshSelectionChange(meshName) {
        if (!meshName) {
            this.hideMeshInfo();
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/mesh/info/${meshName}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const info = await response.json();
            this.showMeshInfo(info);
            this.logMessage(`选择了Mesh: ${meshName}`, 'info');

        } catch (error) {
            console.error('获取Mesh信息失败:', error);
            this.logMessage('获取Mesh信息失败: ' + error.message, 'error');
            this.hideMeshInfo();
        }
    }

    /**
     * 显示Mesh信息
     */
    showMeshInfo(info) {
        const infoDiv = document.getElementById('mesh-info');
        const verticesSpan = document.getElementById('mesh-vertices');
        const sizeSpan = document.getElementById('mesh-size');

        verticesSpan.textContent = info.vertex_count || 0;
        sizeSpan.textContent = info.file_size || 0;

        infoDiv.classList.remove('hidden');
    }

    /**
     * 隐藏Mesh信息
     */
    hideMeshInfo() {
        document.getElementById('mesh-info').classList.add('hidden');
    }

    /**
     * 开始训练
     */
    async startTraining() {
        const meshName = document.getElementById('mesh-select').value;
        const maxEpisodes = parseInt(document.getElementById('max-episodes').value);
        const maxSteps = parseInt(document.getElementById('max-steps').value);

        if (!meshName) {
            this.logMessage('请先选择一个Mesh文件', 'error');
            return;
        }

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/training/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mesh_name: meshName,
                    max_episodes: maxEpisodes,
                    max_steps: maxSteps
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.logMessage('训练已启动: ' + result.message, 'success');

            this.isTraining = true;
            this.updateUI();
            this.startPeriodicUpdate();

        } catch (error) {
            console.error('启动训练失败:', error);
            this.logMessage('启动训练失败: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 停止训练
     */
    async stopTraining() {
        // 在发送请求前就立即停止轮询
        this.stopPeriodicUpdate();

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/training/stop`, {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.logMessage('训练停止请求已发送: ' + result.message, 'info');

            // 手动更新状态为 "stopping"
            this.isTraining = false;
            this.updateStatusIndicator('stopping');
            this.updateUI();

        } catch (error) {
            console.error('停止训练失败:', error);
            this.logMessage('停止训练失败: ' + error.message, 'error');
            // 如果出错，确保UI状态正确
            this.isTraining = false;
            this.updateUI();
        } finally {
            this.showLoading(false);
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

        const interval = parseInt(document.getElementById('update-interval').value) * 1000; // 转换为毫秒

        this.updateInterval = setInterval(async () => {
            await this.updateTrainingStatus();
        }, interval);
    }

    /**
     * 停止定期更新
     */
    stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * 更新训练状态
     */
    async updateTrainingStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/training/status`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const status = await response.json();
            this.handleStatusUpdate(status);

        } catch (error) {
            console.error('获取训练状态失败:', error);
            this.logMessage('获取训练状态失败: ' + error.message, 'error');
        }
    }

    /**
     * 处理状态更新
     */
    handleStatusUpdate(status) {
        // 更新运行状态
        this.isTraining = status.running;

        // 更新状态指示器
        this.updateStatusIndicator(status.status);

        // 更新统计数据
        if (status.stats) {
            this.updateTrainingStats(status.stats);
        }

        // 更新进度信息
        if (status.progress) {
            this.updateProgressInfo(status.progress);
        }

        // 如果训练状态明确表示已停止、完成或出错，则确保停止定期更新
        const isFinished = !status.running || ['stopped', 'completed', 'error'].includes(status.status);
        if (isFinished && this.updateInterval) {
            this.stopPeriodicUpdate();
        }

        this.updateUI();
    }

    /**
     * 更新状态指示器
     */
    updateStatusIndicator(status) {
        const indicator = document.getElementById('status-indicator').querySelector('div');
        const text = document.getElementById('status-text');

        // 移除所有状态类
        indicator.className = 'w-2 h-2 rounded-full mr-2';

        switch (status) {
            case 'running':
                indicator.classList.add('status-running');
                text.textContent = '训练中';
                break;
            case 'stopped':
                indicator.classList.add('status-stopped');
                text.textContent = '已停止';
                break;
            case 'completed':
                indicator.classList.add('status-success');
                text.textContent = '已完成';
                break;
            case 'stopping':
                indicator.classList.add('status-loading');
                text.textContent = '停止中';
                break;
            case 'error':
                indicator.classList.add('status-stopped');
                text.textContent = '出错';
                break;
            default:
                indicator.classList.add('status-idle');
                text.textContent = '未启动';
        }
    }

    /**
     * 更新进度信息
     */
    updateProgressInfo(progress) {
        if (progress.current_episode !== undefined) {
            document.getElementById('current-episode').textContent = progress.current_episode;
            document.getElementById('display-episode').textContent = progress.current_episode;
        }

        if (progress.total_steps !== undefined) {
            document.getElementById('total-steps').textContent = progress.total_steps;
        }

        if (progress.average_reward !== undefined) {
            document.getElementById('avg-reward').textContent = progress.average_reward.toFixed(3);
        }

        if (progress.buffer_utilization !== undefined) {
            document.getElementById('buffer-size').textContent = progress.buffer_utilization;
        }

        if (progress.latest_reward !== undefined) {
            document.getElementById('episode-reward').textContent = progress.latest_reward.toFixed(3);
        }
    }

    /**
     * 更新训练统计数据
     */
    updateTrainingStats(stats) {
        const formatNumber = (num) => (num !== undefined && num !== null) ? num.toFixed(3) : 'N/A';

        // 如果页面中存在statsContainer，则更新其内容
        if (this.statsContainer) {
            this.statsContainer.innerHTML = `
                <span>Episode: ${stats.episode || 'N/A'}</span>
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

        // 新增：处理参考点信息
        if (stats.reference_point_info) {
            this.refPointInfo = stats.reference_point_info;
        }

        // 统一渲染mesh和boundary数据，避免坐标变换不一致导致的错位问题
        let meshData = stats.mesh_data || this.meshData;
        let boundaryData = stats.boundary_vertices_data || this.boundaryData;

        // 如果后端返回的mesh或boundary数据是字符串，尝试解析
        try {
            if (typeof meshData === 'string') {
                meshData = JSON.parse(meshData);
            }
        } catch (e) {
            console.error('解析mesh数据失败:', e);
            meshData = null;
        }

        try {
            if (typeof boundaryData === 'string') {
                boundaryData = JSON.parse(boundaryData);
            }
        } catch (e) {
            console.error('解析boundary数据失败:', e);
            boundaryData = null;
        }

        if (meshData || boundaryData) {
            this.meshData = meshData;
            this.boundaryData = boundaryData;
            this.renderMeshAndBoundary(meshData, boundaryData);
        }
    }

    /**
     * 统一渲染Mesh和Boundary，避免坐标变换不一致导致的错位问题
     */
    renderMeshAndBoundary(meshData, boundaryVertices) {
        // 兼容后端可能返回字符串的情况
        try {
            if (typeof meshData === 'string') {
                meshData = JSON.parse(meshData);
            }
        } catch (e) {
            console.error('解析mesh数据失败:', e);
            meshData = null;
        }

        try {
            if (typeof boundaryVertices === 'string') {
                boundaryVertices = JSON.parse(boundaryVertices);
            }
        } catch (e) {
            console.error('解析boundary数据失败:', e);
            boundaryVertices = null;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // 重新绘制背景网格
        this.drawGrid();

        const allVertices = [];
        if (boundaryVertices) {
            allVertices.push(...boundaryVertices);
        }

        if (meshData) {
            Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
                try {
                    const vertex = JSON.parse(vertexStr);
                    allVertices.push(vertex);
                    allVertices.push(...adjacentVertices);
                } catch (e) {
                }
            });
        }

        if (allVertices.length === 0) return;

        const xCoords = allVertices.map(v => v[0]);
        const yCoords = allVertices.map(v => v[1]);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        const dataWidth = maxX - minX;
        const dataHeight = maxY - minY;

        const padding = 50;
        const scaleX = (this.canvas.width - 2 * padding) / (dataWidth || 1);
        const scaleY = (this.canvas.height - 2 * padding) / (dataHeight || 1);
        const scale = Math.min(scaleX, scaleY);

        const offsetX = (this.canvas.width - dataWidth * scale) / 2 - minX * scale;
        const offsetY = (this.canvas.height - dataHeight * scale) / 2 - minY * scale;

        const transform = {
            scale,
            offsetX,
            offsetY
        };

        // 首先渲染mesh（如果存在）
        if (meshData && Object.keys(meshData).length > 0) {
            this.renderMeshWithTransform(meshData, transform);
        }

        // 然后渲染boundary（如果存在）
        if (boundaryVertices && boundaryVertices.length > 0) {
            this.renderBoundaryWithTransform(boundaryVertices, transform);
        }

        // 最后在顶层渲染参考点信息
        if (this.refPointInfo) {
            this.renderReferencePointInfo(this.refPointInfo, transform);
        }
    }

    renderReferencePointInfo(refInfo, transform) {
        if (!refInfo || !refInfo.local_env_vertices || !refInfo.ref_vertex) return;

        const {local_env_vertices, ref_vertex} = refInfo;

        // 1. 绘制局部环境的边 (用醒目的颜色)
        if (local_env_vertices.length > 1) {
            this.ctx.strokeStyle = '#F59E0B'; // 黄色
            this.ctx.lineWidth = 4; // 更粗的线条
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();

            const firstPoint = this.worldToScreen(local_env_vertices[0], transform);
            this.ctx.moveTo(firstPoint[0], firstPoint[1]);
            for (let i = 1; i < local_env_vertices.length; i++) {
                const point = this.worldToScreen(local_env_vertices[i], transform);
                this.ctx.lineTo(point[0], point[1]);
            }
            this.ctx.stroke();
        }

        // 2. 突出显示参考点本身 (用另一种醒目的颜色)
        const refScreenPos = this.worldToScreen(ref_vertex, transform);
        this.ctx.fillStyle = '#10B981'; // 绿色
        this.ctx.strokeStyle = '#FFFFFF'; // 白色描边，使其在任何背景下都清晰可见
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(refScreenPos[0], refScreenPos[1], 8, 0, 2 * Math.PI); // 更大的半径
        this.ctx.fill();
        this.ctx.stroke();
    }

    /**
     * 使用指定变换参数渲染Mesh
     */
    renderMeshWithTransform(meshData, transform) {
        // 绘制边
        this.ctx.strokeStyle = '#6366F1';
        this.ctx.lineWidth = 2;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                // 后端现在发送的是JSON字符串，所以可以直接解析
                const [x1, y1] = JSON.parse(vertexStr);
                const screenPos1 = this.worldToScreen([x1, y1], transform);

                adjacentVertices.forEach(([x2, y2]) => {
                    const screenPos2 = this.worldToScreen([x2, y2], transform);

                    this.ctx.beginPath();
                    this.ctx.moveTo(screenPos1[0], screenPos1[1]);
                    this.ctx.lineTo(screenPos2[0], screenPos2[1]);
                    this.ctx.stroke();
                });
            } catch (e) {
                // 如果解析失败，在控制台打印错误，避免整个应用崩溃
                console.error('无法解析顶点数据:', vertexStr, e);
            }
        });

        // 绘制mesh顶点... (这部分代码无需修改)
    }

    /**
     * 使用指定变换参数渲染Boundary
     */
    renderBoundaryWithTransform(boundaryVertices, transform) {
        if (!boundaryVertices || boundaryVertices.length === 0) return;

        // 绘制边界线
        this.ctx.strokeStyle = '#EF4444';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();

        const firstPoint = this.worldToScreen(boundaryVertices[0], transform);
        this.ctx.moveTo(firstPoint[0], firstPoint[1]);

        for (let i = 1; i < boundaryVertices.length; i++) {
            const point = this.worldToScreen(boundaryVertices[i], transform);
            this.ctx.lineTo(point[0], point[1]);
        }

        // 闭合边界
        this.ctx.lineTo(firstPoint[0], firstPoint[1]);
        this.ctx.stroke();

        // 绘制边界顶点
        this.ctx.fillStyle = '#DC2626';
        boundaryVertices.forEach(vertex => {
            const screenPos = this.worldToScreen(vertex, transform);
            this.ctx.beginPath();
            this.ctx.arc(screenPos[0], screenPos[1], 4, 0, 2 * Math.PI);
            this.ctx.fill();
        });
    }

    /**
     * 渲染Mesh（保持向后兼容性）
     */
    renderMesh(meshData) {
        this.renderMeshAndBoundary(meshData, null);
    }

    /**
     * 渲染边界（保持向后兼容性）
     */
    renderBoundary(boundaryVertices) {
        this.boundaryData = boundaryVertices;
        this.renderMeshAndBoundary(this.meshData, boundaryVertices);
    }

    /**
     * 计算顶点边界框
     */
    calculateBounds(vertices) {
        if (vertices.length === 0) return {minX: 0, minY: 0, maxX: 1, maxY: 1};

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

        vertices.forEach(([x, y]) => {
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        });

        return {minX, minY, maxX, maxY};
    }

    /**
     * 计算坐标变换参数
     */
    calculateTransform(bounds) {
        const margin = 40;
        const canvasWidth = this.canvas.width - 2 * margin;
        const canvasHeight = this.canvas.height - 2 * margin;

        const dataWidth = bounds.maxX - bounds.minX || 1;
        const dataHeight = bounds.maxY - bounds.minY || 1;

        const scaleX = canvasWidth / dataWidth;
        const scaleY = canvasHeight / dataHeight;
        const scale = Math.min(scaleX, scaleY);

        const offsetX = margin + (canvasWidth - dataWidth * scale) / 2 - bounds.minX * scale;
        const offsetY = margin + (canvasHeight - dataHeight * scale) / 2 - bounds.minY * scale;

        return {scale, offsetX, offsetY};
    }

    /**
     * 世界坐标转屏幕坐标
     */
    worldToScreen([x, y], transform) {
        return [
            x * transform.scale + transform.offsetX,
            y * transform.scale + transform.offsetY
        ];
    }

    /**
     * 更新UI状态
     */
    updateUI() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const meshSelect = document.getElementById('mesh-select');
        const maxEpisodes = document.getElementById('max-episodes');
        const maxSteps = document.getElementById('max-steps');
        const updateInterval = document.getElementById('update-interval');

        if (this.isTraining) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            meshSelect.disabled = true;
            maxEpisodes.disabled = true;
            maxSteps.disabled = true;
            updateInterval.disabled = true;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            meshSelect.disabled = false;
            maxEpisodes.disabled = false;
            maxSteps.disabled = false;
            updateInterval.disabled = false;
        }
    }

    /**
     * 显示/隐藏加载指示器
     */
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    /**
     * 记录日志消息
     */
    logMessage(message, type = 'info') {
        const container = document.getElementById('log-container');
        const timestamp = new Date().toLocaleTimeString();

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;

        let icon = '';
        switch (type) {
            case 'success':
                icon = '✓';
                logEntry.style.color = '#059669';
                break;
            case 'error':
                icon = '✗';
                logEntry.style.color = '#DC2626';
                break;
            case 'warning':
                icon = '⚠';
                logEntry.style.color = '#D97706';
                break;
            default:
                icon = 'ℹ';
                logEntry.style.color = '#6B7280';
        }

        logEntry.innerHTML = `<span style="color: #9CA3AF;">[${timestamp}]</span> ${icon} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // 限制日志条数
        const maxLogs = 100;
        while (container.children.length > maxLogs) {
            container.removeChild(container.firstChild);
        }
    }

    /**
     * 清除日志
     */
    clearLogs() {
        const container = document.getElementById('log-container');
        container.innerHTML = '<div class="text-gray-500">日志已清除</div>';
    }
}

// 当页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.trainingManager = new TrainingManager();
});