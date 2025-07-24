/**
 * Canvas Rendering Module - Fixed responsive and zoom issues
 * Handles all Canvas-related drawing functions
 */

import {CONSTANTS, isValidCoordinate, parseBackendData} from './utils.js';

export class CanvasRenderer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.currentTransform = null;
        this.isResizing = false;

        // Add debounce mechanism
        this.resizeDebounceTimer = null;
        this.lastRenderData = null; // Cache last render data

        this.setupCanvas();
        this.bindResizeEvent();
    }

    /**
     * Setup Canvas basic configuration
     */
    setupCanvas() {
        this.resizeCanvas();
        this.clearCanvas();
    }

    /**
     * 绑定窗口大小改变事件 - 修复版本
     */
    bindResizeEvent() {
        // 使用防抖机制优化性能
        const debouncedResize = () => {
            clearTimeout(this.resizeDebounceTimer);
            this.resizeDebounceTimer = setTimeout(() => {
                this.handleResize();
            }, 150);
        };

        window.addEventListener('resize', debouncedResize);

        // 监听浏览器缩放变化
        let lastDevicePixelRatio = window.devicePixelRatio;
        const checkPixelRatio = () => {
            if (window.devicePixelRatio !== lastDevicePixelRatio) {
                lastDevicePixelRatio = window.devicePixelRatio;
                this.handleResize();
            }
            requestAnimationFrame(checkPixelRatio);
        };
        checkPixelRatio();
    }

    /**
     * 处理窗口大小变化 - 新增方法
     */
    handleResize() {
        if (this.isResizing) return;

        this.isResizing = true;

        try {
            this.resizeCanvas();

            // 如果有缓存的渲染数据，重新渲染
            if (this.lastRenderData) {
                this.renderScene(
                    this.lastRenderData.meshData,
                    this.lastRenderData.boundaryVertices,
                    this.lastRenderData.refPointInfo
                );
            } else {
                this.clearCanvas();
            }
        } catch (error) {
            console.error('Canvas resize error:', error);
        } finally {
            this.isResizing = false;
        }
    }

    /**
     * 调整Canvas大小 - 修复版本，确保居中
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        const rect = container.getBoundingClientRect();

        // 确保容器有有效的尺寸
        if (rect.width === 0 || rect.height === 0) {
            // 延迟重试
            setTimeout(() => this.resizeCanvas(), 100);
            return;
        }

        // 考虑容器的padding和边框
        const computedStyle = window.getComputedStyle(container);
        const paddingX = parseFloat(computedStyle.paddingLeft) + parseFloat(computedStyle.paddingRight);
        const paddingY = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
        const borderX = parseFloat(computedStyle.borderLeftWidth) + parseFloat(computedStyle.borderRightWidth);
        const borderY = parseFloat(computedStyle.borderTopWidth) + parseFloat(computedStyle.borderBottomWidth);

        // 计算可用空间
        const availableWidth = rect.width - paddingX - borderX;
        const availableHeight = rect.height - paddingY - borderY;

        // 设置Canvas的显示大小，保持一定的边距
        const margin = 8; // 8px的边距
        const displayWidth = Math.max(100, availableWidth - margin * 2);
        const displayHeight = Math.max(100, availableHeight - margin * 2);

        // 获取当前设备像素比，处理高DPI显示器
        const devicePixelRatio = window.devicePixelRatio || 1;

        // 设置Canvas的实际像素大小
        this.canvas.width = displayWidth * devicePixelRatio;
        this.canvas.height = displayHeight * devicePixelRatio;

        // 设置Canvas的显示大小
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';

        // 清除之前的变换并重新设置
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);

        // 缩放Canvas上下文以匹配设备像素比
        this.ctx.scale(devicePixelRatio, devicePixelRatio);

        // 重新设置默认样式
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';

        // 确保Canvas在容器中居中（通过CSS flexbox已经处理，这里不需要额外设置）
    }

    /**
     * 清空Canvas
     */
    clearCanvas() {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.clearRect(0, 0, displayWidth, displayHeight);
        this.drawGrid();
        this.drawWaitingText();
        this.currentTransform = null;
        this.lastRenderData = null;
    }

    /**
     * 绘制网格背景 - 修复版本
     */
    drawGrid() {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        const gridSize = CONSTANTS.GRID_SIZE;
        this.ctx.strokeStyle = '#F3F4F6';
        this.ctx.lineWidth = 0.5;

        // 垂直线
        for (let x = 0; x <= displayWidth; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, displayHeight);
            this.ctx.stroke();
        }

        // 水平线
        for (let y = 0; y <= displayHeight; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(displayWidth, y);
            this.ctx.stroke();
        }
    }

    /**
     * 绘制等待文本 - 修复版本
     */
    drawWaitingText() {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.fillStyle = '#9CA3AF';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Select a mesh file to preview boundary...',
            displayWidth / 2,
            displayHeight / 2
        );
    }

    /**
     * 渲染边界预览（新增方法）
     * @param {Array} boundaryVertices - 边界顶点数据
     * @param {string} meshName - mesh名称
     */
    renderBoundaryPreview(boundaryVertices, meshName = '') {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.clearRect(0, 0, displayWidth, displayHeight);
        this.drawGrid();

        if (!boundaryVertices || !Array.isArray(boundaryVertices) || boundaryVertices.length === 0) {
            this.drawWaitingText();
            return;
        }

        // 缓存预览数据
        this.lastRenderData = {
            meshData: null,
            boundaryVertices: boundaryVertices,
            refPointInfo: null,
            isPreview: true,
            meshName: meshName
        };

        // 计算变换参数
        const transform = this.calculateTransform(boundaryVertices);
        this.currentTransform = transform;

        // 绘制边界
        this.renderBoundaryWithTransform(boundaryVertices, transform);

        // 绘制标题
        this.drawPreviewTitle(meshName, boundaryVertices.length);
    }

    /**
     * 绘制预览标题
     * @param {string} meshName - mesh名称
     * @param {number} vertexCount - 顶点数量
     */
    drawPreviewTitle(meshName, vertexCount) {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);

        this.ctx.fillStyle = '#374151';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            `${meshName} (${vertexCount} vertices)`,
            displayWidth / 2,
            30
        );
    }

    /**
     * 统一渲染Mesh和Boundary - 修复版本
     * @param {Object} meshData - 网格数据
     * @param {Array} boundaryVertices - 边界顶点数据
     * @param {Object} refPointInfo - 参考点信息
     */
    renderScene(meshData, boundaryVertices, refPointInfo = null) {
        // 缓存渲染数据
        this.lastRenderData = {
            meshData: meshData,
            boundaryVertices: boundaryVertices,
            refPointInfo: refPointInfo,
            isPreview: false
        };

        // 解析数据
        meshData = parseBackendData(meshData);
        boundaryVertices = parseBackendData(boundaryVertices);

        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.clearRect(0, 0, displayWidth, displayHeight);
        this.drawGrid();

        // 收集所有顶点用于计算变换
        const allVertices = this.collectAllVertices(meshData, boundaryVertices);

        if (allVertices.length === 0) {
            this.currentTransform = null;
            return;
        }

        // 计算变换参数
        const transform = this.calculateTransform(allVertices);
        this.currentTransform = transform;

        // 按层次渲染
        if (meshData && Object.keys(meshData).length > 0) {
            this.renderMeshWithTransform(meshData, transform);
        }

        if (boundaryVertices && boundaryVertices.length > 0) {
            this.renderBoundaryWithTransform(boundaryVertices, transform);
        }

        if (refPointInfo) {
            this.renderReferencePointInfo(refPointInfo, transform);
        }
    }

    /**
     * 收集所有顶点用于计算边界
     * @param {Object} meshData - 网格数据
     * @param {Array} boundaryVertices - 边界顶点
     * @returns {Array} 所有顶点数组
     */
    collectAllVertices(meshData, boundaryVertices) {
        const allVertices = [];

        if (boundaryVertices && Array.isArray(boundaryVertices)) {
            allVertices.push(...boundaryVertices.filter(isValidCoordinate));
        }

        if (meshData && typeof meshData === 'object') {
            Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
                try {
                    const vertex = JSON.parse(vertexStr);
                    if (isValidCoordinate(vertex)) {
                        allVertices.push(vertex);
                    }

                    if (Array.isArray(adjacentVertices)) {
                        allVertices.push(...adjacentVertices.filter(isValidCoordinate));
                    }
                } catch (e) {
                    console.warn('Failed to parse vertex data:', vertexStr);
                }
            });
        }

        return allVertices;
    }

    /**
     * 计算坐标变换参数 - 修复版本
     * @param {Array} vertices - 顶点数组
     * @returns {Object} 变换参数
     */
    calculateTransform(vertices) {
        const xCoords = vertices.map(v => v[0]);
        const yCoords = vertices.map(v => v[1]);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        const dataWidth = maxX - minX;
        const dataHeight = maxY - minY;

        // 使用逻辑像素计算
        const logicalWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const logicalHeight = this.canvas.height / (window.devicePixelRatio || 1);

        const padding = CONSTANTS.DEFAULT_PADDING;
        const scaleX = (logicalWidth - 2 * padding) / (dataWidth || 1);
        const scaleY = (logicalHeight - 2 * padding) / (dataHeight || 1);
        const scale = Math.min(scaleX, scaleY);

        const offsetX = (logicalWidth - dataWidth * scale) / 2 - minX * scale;
        const offsetY = (logicalHeight - dataHeight * scale) / 2 - minY * scale;

        return {scale, offsetX, offsetY};
    }

    /**
     * 使用指定变换参数渲染Mesh
     * @param {Object} meshData - 网格数据
     * @param {Object} transform - 变换参数
     */
    renderMeshWithTransform(meshData, transform) {
        // 绘制网格边
        this.ctx.strokeStyle = '#6366F1';
        this.ctx.lineWidth = 2;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                const [x1, y1] = JSON.parse(vertexStr);
                const p1 = this.worldToScreen([x1, y1], transform);

                if (Array.isArray(adjacentVertices)) {
                    adjacentVertices.forEach(vertex => {
                        if (isValidCoordinate(vertex)) {
                            const p2 = this.worldToScreen(vertex, transform);
                            this.drawLine(p1, p2);
                        }
                    });
                }
            } catch (e) {
                console.warn('Failed to render mesh edge:', vertexStr);
            }
        });

        // 绘制网格顶点
        this.drawMeshVertices(meshData, transform);
    }

    /**
     * 绘制网格顶点
     * @param {Object} meshData - 网格数据
     * @param {Object} transform - 变换参数
     */
    drawMeshVertices(meshData, transform) {
        const drawn = new Set();

        this.ctx.fillStyle = '#3B82F6';
        this.ctx.strokeStyle = '#1E40AF';
        this.ctx.lineWidth = 1.5;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                // 绘制中心顶点
                if (!drawn.has(vertexStr)) {
                    const center = JSON.parse(vertexStr);
                    if (isValidCoordinate(center)) {
                        const pos = this.worldToScreen(center, transform);
                        this.drawVertex(pos, CONSTANTS.VERTEX_RADIUS);
                        drawn.add(vertexStr);
                    }
                }

                // 绘制邻接顶点
                if (Array.isArray(adjacentVertices)) {
                    adjacentVertices.forEach(vertex => {
                        if (isValidCoordinate(vertex)) {
                            const key = JSON.stringify(vertex);
                            if (!drawn.has(key)) {
                                const pos = this.worldToScreen(vertex, transform);
                                this.drawVertex(pos, CONSTANTS.VERTEX_RADIUS);
                                drawn.add(key);
                            }
                        }
                    });
                }
            } catch (e) {
                console.warn('Failed to draw mesh vertex:', vertexStr);
            }
        });
    }

    /**
     * 使用指定变换参数渲染边界
     * @param {Array} boundaryVertices - 边界顶点
     * @param {Object} transform - 变换参数
     */
    renderBoundaryWithTransform(boundaryVertices, transform) {
        if (!Array.isArray(boundaryVertices) || boundaryVertices.length === 0) {
            return;
        }

        // 绘制边界线
        this.ctx.strokeStyle = '#EF4444';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();

        const firstPoint = this.worldToScreen(boundaryVertices[0], transform);
        this.ctx.moveTo(firstPoint[0], firstPoint[1]);

        for (let i = 1; i < boundaryVertices.length; i++) {
            if (isValidCoordinate(boundaryVertices[i])) {
                const point = this.worldToScreen(boundaryVertices[i], transform);
                this.ctx.lineTo(point[0], point[1]);
            }
        }

        // 闭合边界
        this.ctx.lineTo(firstPoint[0], firstPoint[1]);
        this.ctx.stroke();

        // 绘制边界顶点
        this.ctx.fillStyle = '#DC2626';
        boundaryVertices.forEach(vertex => {
            if (isValidCoordinate(vertex)) {
                const screenPos = this.worldToScreen(vertex, transform);
                this.drawVertex(screenPos, 4);
            }
        });
    }

    /**
     * 渲染参考点信息
     * @param {Object} refInfo - 参考点信息
     * @param {Object} transform - 变换参数
     */
    renderReferencePointInfo(refInfo, transform) {
        if (!refInfo || !refInfo.local_env_vertices || !refInfo.ref_vertex) {
            return;
        }

        const {local_env_vertices, ref_vertex} = refInfo;

        // 绘制局部环境的边
        if (Array.isArray(local_env_vertices) && local_env_vertices.length > 1) {
            this.ctx.strokeStyle = '#F59E0B';
            this.ctx.lineWidth = 4;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();

            const firstPoint = this.worldToScreen(local_env_vertices[0], transform);
            this.ctx.moveTo(firstPoint[0], firstPoint[1]);

            for (let i = 1; i < local_env_vertices.length; i++) {
                if (isValidCoordinate(local_env_vertices[i])) {
                    const point = this.worldToScreen(local_env_vertices[i], transform);
                    this.ctx.lineTo(point[0], point[1]);
                }
            }
            this.ctx.stroke();
        }

        // 突出显示参考点
        if (isValidCoordinate(ref_vertex)) {
            const refScreenPos = this.worldToScreen(ref_vertex, transform);
            this.ctx.fillStyle = '#10B981';
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            this.drawVertex(refScreenPos, 8);
        }
    }

    /**
     * 绘制顶点
     * @param {Array} position - 屏幕坐标 [x, y]
     * @param {number} radius - 半径
     */
    drawVertex(position, radius) {
        this.ctx.beginPath();
        this.ctx.arc(position[0], position[1], radius, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.stroke();
    }

    /**
     * 绘制线段
     * @param {Array} start - 起点 [x, y]
     * @param {Array} end - 终点 [x, y]
     */
    drawLine(start, end) {
        this.ctx.beginPath();
        this.ctx.moveTo(start[0], start[1]);
        this.ctx.lineTo(end[0], end[1]);
        this.ctx.stroke();
    }

    /**
     * 世界坐标转屏幕坐标
     * @param {Array} worldCoords - 世界坐标 [x, y]
     * @param {Object} transform - 变换参数
     * @returns {Array} 屏幕坐标 [x, y]
     */
    worldToScreen(worldCoords, transform) {
        const [x, y] = worldCoords;
        return [
            x * transform.scale + transform.offsetX,
            y * transform.scale + transform.offsetY
        ];
    }

    /**
     * 屏幕坐标转世界坐标 - 修复版本
     * @param {number} screenX - 屏幕X坐标
     * @param {number} screenY - 屏幕Y坐标
     * @param {Object} transform - 变换参数
     * @returns {Array} 世界坐标 [x, y]
     */
    screenToWorld(screenX, screenY, transform) {
        if (!transform) {
            return [0, 0];
        }

        const worldX = (screenX - transform.offsetX) / transform.scale;
        const worldY = (screenY - transform.offsetY) / transform.scale;
        return [worldX, worldY];
    }

    /**
     * 获取当前变换参数
     * @returns {Object|null} 变换参数
     */
    getCurrentTransform() {
        return this.currentTransform;
    }

    /**
     * 公共方法：处理窗口大小变化
     */
    onResize() {
        this.handleResize();
    }

    /**
     * 销毁Canvas渲染器
     */
    destroy() {
        if (this.resizeDebounceTimer) {
            clearTimeout(this.resizeDebounceTimer);
        }
        this.lastRenderData = null;
        this.currentTransform = null;
    }
}