/**
 * Canvas渲染模块 - 修复版本
 * 负责所有Canvas相关的绘制功能
 * 修复了DOM元素获取和初始化的问题
 */

import {CONSTANTS, isValidCoordinate, parseBackendData} from './utils.js';

export class CanvasRenderer {
    constructor(canvasElement = null) {
        this.canvas = null;
        this.ctx = null;
        this.currentTransform = null;
        this.isResizing = false;
        this.lastRenderData = null;
        this.currentPixelRatio = 1;

        // 如果传入了canvas元素，立即初始化
        if (canvasElement) {
            this.init(canvasElement);
        }
    }

    /**
     * 初始化Canvas渲染器
     * @param {HTMLCanvasElement} canvasElement - Canvas元素
     */
    init(canvasElement = null) {
        if (canvasElement) {
            this.canvas = canvasElement;
        } else {
            // 尝试从DOM获取canvas元素
            this.canvas = document.getElementById('mesh-canvas');
        }

        if (!this.canvas) {
            console.error('未找到Canvas元素，无法初始化CanvasRenderer');
            return false;
        }

        try {
            this.ctx = this.canvas.getContext('2d');
            if (!this.ctx) {
                console.error('无法获取Canvas 2D上下文');
                return false;
            }

            this.setupCanvas();
            this.bindEvents();
            console.log('CanvasRenderer初始化成功');
            return true;
        } catch (error) {
            console.error('CanvasRenderer初始化失败:', error);
            return false;
        }
    }

    /**
     * 检查是否已初始化
     * @returns {boolean} 是否已初始化
     */
    isInitialized() {
        return this.canvas && this.ctx;
    }

    /**
     * 设置Canvas基本配置
     */
    setupCanvas() {
        if (!this.isInitialized()) {
            console.warn('Canvas未初始化，无法设置');
            return;
        }

        this.resizeCanvas();
        this.clearCanvas();
    }

    /**
     * 绑定事件监听器
     */
    bindEvents() {
        if (!this.isInitialized()) {
            console.warn('Canvas未初始化，无法绑定事件');
            return;
        }

        // 防抖的resize处理
        let resizeTimeout;
        const debouncedResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (!this.isResizing) {
                    this.resizeCanvas();
                }
            }, 150);
        };

        window.addEventListener('resize', debouncedResize);

        // 监听缩放变化
        let currentZoom = window.devicePixelRatio;
        const checkZoom = () => {
            if (Math.abs(window.devicePixelRatio - currentZoom) > 0.1) {
                currentZoom = window.devicePixelRatio;
                debouncedResize();
            }
        };

        setInterval(checkZoom, 500);
    }

    /**
     * 调整Canvas大小 - 优化缩放支持
     */
    resizeCanvas() {
        if (!this.isInitialized()) {
            console.warn('Canvas未初始化，无法调整大小');
            return;
        }

        if (this.isResizing) return;
        this.isResizing = true;

        try {
            const container = this.canvas.parentElement;
            if (!container) {
                console.warn('Canvas容器不存在');
                return;
            }

            const rect = container.getBoundingClientRect();

            // 计算可用空间，考虑padding和border
            const computedStyle = window.getComputedStyle(container);
            const paddingX = parseFloat(computedStyle.paddingLeft) + parseFloat(computedStyle.paddingRight);
            const paddingY = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
            const borderX = parseFloat(computedStyle.borderLeftWidth) + parseFloat(computedStyle.borderRightWidth);
            const borderY = parseFloat(computedStyle.borderTopWidth) + parseFloat(computedStyle.borderBottomWidth);

            const availableWidth = rect.width - paddingX - borderX;
            const availableHeight = rect.height - paddingY - borderY;

            // 确保最小尺寸
            const displayWidth = Math.max(availableWidth, 200);
            const displayHeight = Math.max(availableHeight, 150);

            // 获取设备像素比，但限制范围避免极端情况
            const pixelRatio = Math.min(Math.max(window.devicePixelRatio || 1, 0.5), 3);

            // 设置Canvas的实际像素大小
            this.canvas.width = Math.floor(displayWidth * pixelRatio);
            this.canvas.height = Math.floor(displayHeight * pixelRatio);

            // 设置Canvas的显示大小
            this.canvas.style.width = displayWidth + 'px';
            this.canvas.style.height = displayHeight + 'px';

            // 缩放Canvas上下文以匹配设备像素比
            this.ctx.scale(pixelRatio, pixelRatio);

            // 保存当前的pixelRatio用于后续计算
            this.currentPixelRatio = pixelRatio;

            // 重新绘制
            this.redrawCurrentContent();

        } catch (error) {
            console.error('Canvas resize error:', error);
        } finally {
            this.isResizing = false;
        }
    }

    /**
     * 重新绘制当前内容
     */
    redrawCurrentContent() {
        if (!this.isInitialized()) {
            return;
        }

        if (this.lastRenderData) {
            this.renderScene(
                this.lastRenderData.meshData,
                this.lastRenderData.boundaryVertices,
                this.lastRenderData.refPointInfo
            );
        } else {
            this.clearCanvas();
        }
    }

    /**
     * 清空Canvas
     */
    clearCanvas() {
        if (!this.isInitialized()) {
            return;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawGrid();
        this.drawWaitingText();
        this.currentTransform = null;
        this.lastRenderData = null;
    }

    /**
     * 绘制网格背景
     */
    drawGrid() {
        if (!this.isInitialized()) {
            return;
        }

        const gridSize = 20; // 固定网格大小
        this.ctx.strokeStyle = '#F3F4F6';
        this.ctx.lineWidth = 0.5;

        const displayWidth = this.canvas.width / (this.currentPixelRatio || 1);
        const displayHeight = this.canvas.height / (this.currentPixelRatio || 1);

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
     * 绘制等待文本
     */
    drawWaitingText() {
        if (!this.isInitialized()) {
            return;
        }

        this.ctx.fillStyle = '#9CA3AF';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';

        const displayWidth = this.canvas.width / (this.currentPixelRatio || 1);
        const displayHeight = this.canvas.height / (this.currentPixelRatio || 1);

        this.ctx.fillText(
            '等待训练开始...',
            displayWidth / 2,
            displayHeight / 2
        );
    }

    /**
     * 统一渲染Mesh和Boundary
     * @param {Object} meshData - 网格数据
     * @param {Array} boundaryVertices - 边界顶点数据
     * @param {Object} refPointInfo - 参考点信息
     */
    renderScene(meshData, boundaryVertices, refPointInfo = null) {
        if (!this.isInitialized()) {
            console.warn('Canvas未初始化，无法渲染场景');
            return;
        }

        // 保存渲染数据用于重绘
        this.lastRenderData = {meshData, boundaryVertices, refPointInfo};

        // 解析数据
        meshData = parseBackendData(meshData);
        boundaryVertices = parseBackendData(boundaryVertices);

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawGrid();

        // 收集所有顶点用于计算变换
        const allVertices = this.collectAllVertices(meshData, boundaryVertices);

        if (allVertices.length === 0) {
            this.currentTransform = null;
            this.drawWaitingText();
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
                    console.warn('解析顶点数据失败:', vertexStr);
                }
            });
        }

        return allVertices;
    }

    /**
     * 计算坐标变换参数 - 优化缩放适配
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

        // 使用逻辑像素计算，适配不同缩放级别
        const logicalWidth = this.canvas.width / (this.currentPixelRatio || 1);
        const logicalHeight = this.canvas.height / (this.currentPixelRatio || 1);

        // 自适应padding，确保在不同尺寸下都有合适的边距
        const paddingRatio = 0.1; // 10%的边距
        const padding = Math.min(logicalWidth, logicalHeight) * paddingRatio;

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
        if (!this.isInitialized()) {
            return;
        }

        // 绘制网格边
        this.ctx.strokeStyle = '#6366F1';
        this.ctx.lineWidth = 2;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                const vertex = JSON.parse(vertexStr);
                if (isValidCoordinate(vertex) && Array.isArray(adjacentVertices)) {
                    adjacentVertices.forEach(adjVertex => {
                        if (isValidCoordinate(adjVertex)) {
                            const [x1, y1] = this.transformPoint(vertex, transform);
                            const [x2, y2] = this.transformPoint(adjVertex, transform);

                            this.ctx.beginPath();
                            this.ctx.moveTo(x1, y1);
                            this.ctx.lineTo(x2, y2);
                            this.ctx.stroke();
                        }
                    });
                }
            } catch (e) {
                console.warn('渲染网格边失败:', e);
            }
        });

        // 绘制网格顶点
        this.ctx.fillStyle = '#4F46E5';
        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                const vertex = JSON.parse(vertexStr);
                if (isValidCoordinate(vertex)) {
                    const [x, y] = this.transformPoint(vertex, transform);
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                }
            } catch (e) {
                console.warn('渲染网格顶点失败:', e);
            }
        });
    }

    /**
     * 使用指定变换参数渲染边界
     * @param {Array} boundaryVertices - 边界顶点数组
     * @param {Object} transform - 变换参数
     */
    renderBoundaryWithTransform(boundaryVertices, transform) {
        if (!this.isInitialized() || !Array.isArray(boundaryVertices) || boundaryVertices.length === 0) {
            return;
        }

        // 绘制边界线
        this.ctx.strokeStyle = '#059669';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();

        const validVertices = boundaryVertices.filter(isValidCoordinate);
        if (validVertices.length === 0) return;

        const firstVertex = this.transformPoint(validVertices[0], transform);
        this.ctx.moveTo(firstVertex[0], firstVertex[1]);

        validVertices.slice(1).forEach(vertex => {
            const [x, y] = this.transformPoint(vertex, transform);
            this.ctx.lineTo(x, y);
        });

        this.ctx.closePath();
        this.ctx.stroke();

        // 绘制边界顶点
        this.ctx.fillStyle = '#DC2626';
        validVertices.forEach(vertex => {
            const [x, y] = this.transformPoint(vertex, transform);
            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
            this.ctx.fill();
        });
    }

    /**
     * 渲染参考点信息
     * @param {Object} refPointInfo - 参考点信息
     * @param {Object} transform - 变换参数
     */
    renderReferencePointInfo(refPointInfo, transform) {
        if (!this.isInitialized() || !refPointInfo || !refPointInfo.coordinates) {
            return;
        }

        const coords = refPointInfo.coordinates;
        if (isValidCoordinate(coords)) {
            const [x, y] = this.transformPoint(coords, transform);

            // 绘制参考点
            this.ctx.fillStyle = '#F59E0B';
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
            this.ctx.fill();

            // 添加边框
            this.ctx.strokeStyle = '#D97706';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
    }

    /**
     * 变换点坐标
     * @param {Array} point - 原始点坐标 [x, y]
     * @param {Object} transform - 变换参数
     * @returns {Array} 变换后的坐标 [x, y]
     */
    transformPoint(point, transform) {
        return [
            point[0] * transform.scale + transform.offsetX,
            point[1] * transform.scale + transform.offsetY
        ];
    }

    /**
     * 获取点击坐标对应的世界坐标
     * @param {Event} event - 点击事件
     * @returns {Array} 世界坐标 [x, y] 或 null
     */
    getClickCoordinates(event) {
        if (!this.isInitialized() || !this.currentTransform) {
            return null;
        }

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // 考虑设备像素比
        const worldX = (x - this.currentTransform.offsetX) / this.currentTransform.scale;
        const worldY = (y - this.currentTransform.offsetY) / this.currentTransform.scale;

        return [worldX, worldY];
    }

    /**
     * 屏幕坐标转世界坐标
     * @param {number} screenX - 屏幕X坐标
     * @param {number} screenY - 屏幕Y坐标
     * @returns {Object|null} 世界坐标 {x, y} 或 null
     */
    screenToWorld(screenX, screenY) {
        if (!this.isInitialized() || !this.currentTransform) {
            return null;
        }

        const rect = this.canvas.getBoundingClientRect();
        const x = screenX - rect.left;
        const y = screenY - rect.top;

        // 考虑设备像素比
        const worldX = (x - this.currentTransform.offsetX) / this.currentTransform.scale;
        const worldY = (y - this.currentTransform.offsetY) / this.currentTransform.scale;

        return {x: worldX, y: worldY};
    }

    /**
     * 销毁渲染器
     */
    destroy() {
        if (this.isInitialized()) {
            this.clearCanvas();
        }
        this.canvas = null;
        this.ctx = null;
        this.currentTransform = null;
        this.lastRenderData = null;
    }
}