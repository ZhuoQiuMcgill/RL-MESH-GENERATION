/**
 * Canvas渲染模块
 * 负责所有Canvas相关的绘制功能
 */

import {CONSTANTS, isValidCoordinate, parseBackendData} from './utils.js';

export class CanvasRenderer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.currentTransform = null;

        this.setupCanvas();
        this.bindResizeEvent();
    }

    /**
     * 设置Canvas基本配置
     */
    setupCanvas() {
        this.resizeCanvas();
        this.clearCanvas();
    }

    /**
     * 绑定窗口大小改变事件
     */
    bindResizeEvent() {
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    /**
     * 调整Canvas大小
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();

        // 考虑padding
        const displayWidth = rect.width - CONSTANTS.CANVAS_PADDING;
        const displayHeight = rect.height - CONSTANTS.CANVAS_PADDING;

        // 设置Canvas的实际像素大小
        this.canvas.width = displayWidth * CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO;
        this.canvas.height = displayHeight * CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO;

        // 设置Canvas的显示大小
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';

        // 缩放Canvas上下文以匹配设备像素比
        this.ctx.scale(CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO, CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO);
    }

    /**
     * 清空Canvas
     */
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawGrid();
        this.drawWaitingText();
        this.currentTransform = null;
    }

    /**
     * 绘制网格背景
     */
    drawGrid() {
        const gridSize = CONSTANTS.GRID_SIZE;
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
     * 绘制等待文本
     */
    drawWaitingText() {
        this.ctx.fillStyle = '#9CA3AF';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            '等待训练开始...',
            this.canvas.width / (2 * CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO),
            this.canvas.height / (2 * CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO)
        );
    }

    /**
     * 统一渲染Mesh和Boundary
     * @param {Object} meshData - 网格数据
     * @param {Array} boundaryVertices - 边界顶点数据
     * @param {Object} refPointInfo - 参考点信息
     */
    renderScene(meshData, boundaryVertices, refPointInfo = null) {
        // 解析数据
        meshData = parseBackendData(meshData);
        boundaryVertices = parseBackendData(boundaryVertices);

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
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
                    console.warn('解析顶点数据失败:', vertexStr);
                }
            });
        }

        return allVertices;
    }

    /**
     * 计算坐标变换参数
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
        const logicalWidth = this.canvas.width / CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO;
        const logicalHeight = this.canvas.height / CONSTANTS.CANVAS_DEVICE_PIXEL_RATIO;

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
                console.warn('渲染网格边失败:', vertexStr);
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
                console.warn('绘制网格顶点失败:', vertexStr);
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
     * 屏幕坐标转世界坐标
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
}
