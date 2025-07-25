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
     * Bind window resize event - Fixed version
     */
    bindResizeEvent() {
        // Use debounce mechanism to optimize performance
        const debouncedResize = () => {
            clearTimeout(this.resizeDebounceTimer);
            this.resizeDebounceTimer = setTimeout(() => {
                this.handleResize();
            }, 150);
        };

        window.addEventListener('resize', debouncedResize);

        // Monitor browser zoom changes
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
     * Handle window size changes - New method
     */
    handleResize() {
        if (this.isResizing) return;

        this.isResizing = true;

        try {
            this.resizeCanvas();

            // If there is cached render data, re-render
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
     * Resize Canvas - Fixed version, ensuring centering
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        const rect = container.getBoundingClientRect();

        // Ensure container has valid dimensions
        if (rect.width === 0 || rect.height === 0) {
            // Delayed retry
            setTimeout(() => this.resizeCanvas(), 100);
            return;
        }

        // Consider container padding and borders
        const computedStyle = window.getComputedStyle(container);
        const paddingX = parseFloat(computedStyle.paddingLeft) + parseFloat(computedStyle.paddingRight);
        const paddingY = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
        const borderX = parseFloat(computedStyle.borderLeftWidth) + parseFloat(computedStyle.borderRightWidth);
        const borderY = parseFloat(computedStyle.borderTopWidth) + parseFloat(computedStyle.borderBottomWidth);

        // Calculate available space
        const availableWidth = rect.width - paddingX - borderX;
        const availableHeight = rect.height - paddingY - borderY;

        // Set Canvas display size, maintaining certain margins
        const margin = 8; // 8px margin
        const displayWidth = Math.max(100, availableWidth - margin * 2);
        const displayHeight = Math.max(100, availableHeight - margin * 2);

        // Get current device pixel ratio, handle high DPI displays
        const devicePixelRatio = window.devicePixelRatio || 1;

        // Set Canvas actual pixel size
        this.canvas.width = displayWidth * devicePixelRatio;
        this.canvas.height = displayHeight * devicePixelRatio;

        // Set Canvas display size
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';

        // Clear previous transforms and reset
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);

        // Scale Canvas context to match device pixel ratio
        this.ctx.scale(devicePixelRatio, devicePixelRatio);

        // Reset default styles
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';

        // Ensure Canvas is centered in container (already handled by CSS flexbox, no additional setup needed here)
    }

    /**
     * Clear Canvas
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
     * Draw grid background - Fixed version
     */
    drawGrid() {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        const gridSize = CONSTANTS.GRID_SIZE;
        this.ctx.strokeStyle = '#F3F4F6';
        this.ctx.lineWidth = 0.5;

        // Vertical lines
        for (let x = 0; x <= displayWidth; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, displayHeight);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= displayHeight; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(displayWidth, y);
            this.ctx.stroke();
        }
    }

    /**
     * Draw waiting text - Fixed version
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
     * Render boundary preview (New method)
     * @param {Array} boundaryVertices - Boundary vertex data
     * @param {string} meshName - Mesh name
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

        // Cache preview data
        this.lastRenderData = {
            meshData: null,
            boundaryVertices: boundaryVertices,
            refPointInfo: null,
            isPreview: true,
            meshName: meshName
        };

        // Calculate transformation parameters
        const transform = this.calculateTransform(boundaryVertices);
        this.currentTransform = transform;

        // Draw boundary
        this.renderBoundaryWithTransform(boundaryVertices, transform);

        // Draw title
        this.drawPreviewTitle(meshName, boundaryVertices.length);
    }

    /**
     * Draw preview title
     * @param {string} meshName - Mesh name
     * @param {number} vertexCount - Vertex count
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
     * Unified rendering of Mesh and Boundary - Fixed version
     * @param {Object} meshData - Mesh data
     * @param {Array} boundaryVertices - Boundary vertex data
     * @param {Object} refPointInfo - Reference point information
     */
    renderScene(meshData, boundaryVertices, refPointInfo = null) {
        // Cache render data
        this.lastRenderData = {
            meshData: meshData,
            boundaryVertices: boundaryVertices,
            refPointInfo: refPointInfo,
            isPreview: false
        };

        // Parse data
        meshData = parseBackendData(meshData);
        boundaryVertices = parseBackendData(boundaryVertices);

        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.clearRect(0, 0, displayWidth, displayHeight);
        this.drawGrid();

        // Collect all vertices for transform calculation
        const allVertices = this.collectAllVertices(meshData, boundaryVertices);

        if (allVertices.length === 0) {
            this.currentTransform = null;
            return;
        }

        // Calculate transformation parameters
        const transform = this.calculateTransform(allVertices);
        this.currentTransform = transform;

        // Render in layers
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
     * Collect all vertices for boundary calculation
     * @param {Object} meshData - Mesh data
     * @param {Array} boundaryVertices - Boundary vertices
     * @returns {Array} Array of all vertices
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
     * Calculate coordinate transformation parameters - Fixed version
     * @param {Array} vertices - Vertex array
     * @returns {Object} Transformation parameters
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

        // Use logical pixels for calculation
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
     * Render Mesh using specified transformation parameters
     * @param {Object} meshData - Mesh data
     * @param {Object} transform - Transformation parameters
     */
    renderMeshWithTransform(meshData, transform) {
        // Draw mesh edges
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

        // Draw mesh vertices
        this.drawMeshVertices(meshData, transform);
    }

    /**
     * Draw mesh vertices
     * @param {Object} meshData - Mesh data
     * @param {Object} transform - Transformation parameters
     */
    drawMeshVertices(meshData, transform) {
        const drawn = new Set();

        this.ctx.fillStyle = '#3B82F6';
        this.ctx.strokeStyle = '#1E40AF';
        this.ctx.lineWidth = 1.5;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                // Draw center vertex
                if (!drawn.has(vertexStr)) {
                    const center = JSON.parse(vertexStr);
                    if (isValidCoordinate(center)) {
                        const pos = this.worldToScreen(center, transform);
                        this.drawVertex(pos, CONSTANTS.VERTEX_RADIUS);
                        drawn.add(vertexStr);
                    }
                }

                // Draw adjacent vertices
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
     * Render boundary using specified transformation parameters
     * @param {Array} boundaryVertices - Boundary vertices
     * @param {Object} transform - Transformation parameters
     */
    renderBoundaryWithTransform(boundaryVertices, transform) {
        if (!Array.isArray(boundaryVertices) || boundaryVertices.length === 0) {
            return;
        }

        // Draw boundary lines
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

        // Close boundary
        this.ctx.lineTo(firstPoint[0], firstPoint[1]);
        this.ctx.stroke();

        // Draw boundary vertices
        this.ctx.fillStyle = '#DC2626';
        boundaryVertices.forEach(vertex => {
            if (isValidCoordinate(vertex)) {
                const screenPos = this.worldToScreen(vertex, transform);
                this.drawVertex(screenPos, 4);
            }
        });
    }

    /**
     * Render reference point information
     * @param {Object} refInfo - Reference point information
     * @param {Object} transform - Transformation parameters
     */
    renderReferencePointInfo(refInfo, transform) {
        if (!refInfo || !refInfo.local_env_vertices || !refInfo.ref_vertex) {
            return;
        }

        const {local_env_vertices, ref_vertex} = refInfo;

        // Draw local environment edges
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

        // Highlight reference point
        if (isValidCoordinate(ref_vertex)) {
            const refScreenPos = this.worldToScreen(ref_vertex, transform);
            this.ctx.fillStyle = '#10B981';
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            this.drawVertex(refScreenPos, 8);
        }
    }

    /**
     * Draw vertex
     * @param {Array} position - Screen coordinates [x, y]
     * @param {number} radius - Radius
     */
    drawVertex(position, radius) {
        this.ctx.beginPath();
        this.ctx.arc(position[0], position[1], radius, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.stroke();
    }

    /**
     * Draw line segment
     * @param {Array} start - Start point [x, y]
     * @param {Array} end - End point [x, y]
     */
    drawLine(start, end) {
        this.ctx.beginPath();
        this.ctx.moveTo(start[0], start[1]);
        this.ctx.lineTo(end[0], end[1]);
        this.ctx.stroke();
    }

    /**
     * Convert world coordinates to screen coordinates
     * @param {Array} worldCoords - World coordinates [x, y]
     * @param {Object} transform - Transformation parameters
     * @returns {Array} Screen coordinates [x, y]
     */
    worldToScreen(worldCoords, transform) {
        const [x, y] = worldCoords;
        return [
            x * transform.scale + transform.offsetX,
            y * transform.scale + transform.offsetY
        ];
    }

    /**
     * Convert screen coordinates to world coordinates - Fixed version
     * @param {number} screenX - Screen X coordinate
     * @param {number} screenY - Screen Y coordinate
     * @param {Object} transform - Transformation parameters
     * @returns {Array} World coordinates [x, y]
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
     * Get current transformation parameters
     * @returns {Object|null} Transformation parameters
     */
    getCurrentTransform() {
        return this.currentTransform;
    }

    /**
     * Public method: Handle window size changes
     */
    onResize() {
        this.handleResize();
    }

    /**
     * Destroy Canvas renderer
     */
    destroy() {
        if (this.resizeDebounceTimer) {
            clearTimeout(this.resizeDebounceTimer);
        }
        this.lastRenderData = null;
        this.currentTransform = null;
    }
}