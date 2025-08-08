/**
 * Canvas Rendering Module - React adapted version
 * Handles all Canvas-related drawing functions
 */

import { CONSTANTS, isValidCoordinate, parseBackendData } from './constants.js';

export class CanvasRenderer {
    constructor(canvasElement, options = {}) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.currentTransform = null;
        this.isResizing = false;

        // Enhanced options with defaults
        this.options = {
            backgroundColor: 'transparent',
            showGrid: true,
            enableZoom: true,
            enablePan: true,
            devicePixelRatio: null,
            minZoom: 0.1,
            maxZoom: 5.0,
            ...options
        };

        // Add debounce mechanism
        this.resizeDebounceTimer = null;
        this.lastRenderData = null; // Cache last render data

        // Track current device pixel ratio for high-DPI handling
        this.currentDevicePixelRatio = this.options.devicePixelRatio || window.devicePixelRatio || 1;

        // Dynamic sizing state
        this.adaptiveSizes = {
            vertexRadius: CONSTANTS.VERTEX_RADIUS,
            boundaryVertexRadius: 4,
            boundaryLineWidth: 3,
            meshVertexLineWidth: 1.5,
            referencePointRadius: 8
        };

        this.setupCanvas();
    }

    /**
     * Setup Canvas basic configuration
     */
    setupCanvas() {
        this.resizeCanvas();
        this.clearCanvas();
    }

    /**
     * Bind window resize event with debouncing
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

        // Return cleanup function
        return () => {
            window.removeEventListener('resize', debouncedResize);
        };
    }

    /**
     * Handle window size changes
     */
    handleResize() {
        if (this.isResizing) return;

        this.isResizing = true;

        try {
            this.resizeCanvas();

            // If there is cached render data, re-render  
            if (this.lastRenderData) {
                if (this.lastRenderData.isPreview) {
                    this.renderBoundaryPreview(
                        this.lastRenderData.boundaryVertices,
                        this.lastRenderData.meshName || ''
                    );
                } else {
                    this.renderScene(
                        this.lastRenderData.meshData,
                        this.lastRenderData.boundaryVertices,
                        this.lastRenderData.refPointInfo
                    );
                }
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
     * Enhanced resize canvas with improved high-DPI handling
     * @param {Object} dimensions - Optional dimension overrides
     */
    resizeCanvas(dimensions = null) {
        const container = this.canvas.parentElement;
        if (!container) return;

        let width, height, devicePixelRatio;

        if (dimensions) {
            // Use provided dimensions (from ResizeObserver)
            width = dimensions.width || 0;
            height = dimensions.height || 0;
            devicePixelRatio = dimensions.devicePixelRatio || this.currentDevicePixelRatio;
        } else {
            // Calculate from container
            const rect = container.getBoundingClientRect();

            // Ensure container has valid dimensions
            if (rect.width === 0 || rect.height === 0) {
                setTimeout(() => this.resizeCanvas(), 100);
                return;
            }

            // Consider container padding and borders
            const computedStyle = window.getComputedStyle(container);
            const paddingX = parseFloat(computedStyle.paddingLeft) + parseFloat(computedStyle.paddingRight);
            const paddingY = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
            const borderX = parseFloat(computedStyle.borderLeftWidth) + parseFloat(computedStyle.borderRightWidth);
            const borderY = parseFloat(computedStyle.borderTopWidth) + parseFloat(computedStyle.borderBottomWidth);

            width = rect.width - paddingX - borderX;
            height = rect.height - paddingY - borderY;
            devicePixelRatio = this.options.devicePixelRatio || window.devicePixelRatio || 1;
        }

        // Set minimum canvas dimensions
        const displayWidth = Math.max(100, width - 16); // 8px margin on each side
        const displayHeight = Math.max(100, height - 16);

        // Update current device pixel ratio
        this.currentDevicePixelRatio = devicePixelRatio;

        // Set Canvas actual pixel size with high-DPI scaling
        this.canvas.width = Math.round(displayWidth * devicePixelRatio);
        this.canvas.height = Math.round(displayHeight * devicePixelRatio);

        // Set Canvas display size (CSS pixels)
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';

        // Clear previous transforms and reset context
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);

        // Scale Canvas context to match device pixel ratio
        this.ctx.scale(devicePixelRatio, devicePixelRatio);

        // Enhanced rendering settings for high quality
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.textBaseline = 'middle';
        this.ctx.textAlign = 'center';
    }

    /**
     * Enhanced clear canvas with optional background
     */
    clearCanvas() {
        const displayWidth = this.canvas.width / this.currentDevicePixelRatio;
        const displayHeight = this.canvas.height / this.currentDevicePixelRatio;

        // Clear the entire canvas
        this.ctx.clearRect(0, 0, displayWidth, displayHeight);

        // Apply background color if specified
        if (this.options.backgroundColor !== 'transparent') {
            this.ctx.fillStyle = this.options.backgroundColor;
            this.ctx.fillRect(0, 0, displayWidth, displayHeight);
        }

        // Draw grid if enabled
        if (this.options.showGrid) {
            this.drawGrid();
        }
        
        this.drawWaitingText();
        this.currentTransform = null;
        this.lastRenderData = null;
    }

    /**
     * Enhanced grid drawing with high-DPI support
     */
    drawGrid() {
        if (!this.options.showGrid) return;

        const displayWidth = this.canvas.width / this.currentDevicePixelRatio;
        const displayHeight = this.canvas.height / this.currentDevicePixelRatio;

        const gridSize = CONSTANTS.GRID_SIZE;
        
        // Adjust grid opacity based on background
        const gridOpacity = this.options.backgroundColor === 'transparent' || 
                           this.options.backgroundColor.includes('dark') ? 0.08 : 0.15;
        
        this.ctx.strokeStyle = `rgba(128, 128, 128, ${gridOpacity})`;
        this.ctx.lineWidth = 0.5 / this.currentDevicePixelRatio; // Scale line width for high-DPI

        this.ctx.beginPath();
        
        // Vertical lines
        for (let x = 0; x <= displayWidth; x += gridSize) {
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, displayHeight);
        }

        // Horizontal lines
        for (let y = 0; y <= displayHeight; y += gridSize) {
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(displayWidth, y);
        }
        
        this.ctx.stroke();
    }

    /**
     * Draw waiting text
     */
    drawWaitingText() {
        const displayWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const displayHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.fillStyle = '#a0aec0';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Select a mesh file to preview boundary...',
            displayWidth / 2,
            displayHeight / 2
        );
    }

    /**
     * Render boundary preview
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

        // Calculate adaptive sizes for boundary preview
        this.calculateAdaptiveSizes(boundaryVertices, transform, {
            meshData: null,
            boundaryVertices: boundaryVertices,
            context: 'boundary_preview'
        });

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

        this.ctx.fillStyle = '#cbd5e0';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            `${meshName} (${vertexCount} vertices)`,
            displayWidth / 2,
            30
        );
    }

    /**
     * Unified rendering of Mesh and Boundary
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

        // Calculate adaptive sizes based on data density and context
        this.calculateAdaptiveSizes(allVertices, transform, {
            meshData: meshData,
            boundaryVertices: boundaryVertices,
            context: 'training'
        });

        // Render in layers
        if (meshData && Object.keys(meshData).length > 0) {
            this.renderMeshWithTransform(meshData, transform);
        }

        if (boundaryVertices && boundaryVertices.length > 0) {
            this.renderBoundaryWithTransform(boundaryVertices, transform);
        }

        if (refPointInfo) {
            this.renderReferencePointInfo(refPointInfo, transform, boundaryVertices);
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
     * Calculate coordinate transformation parameters
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

        return { scale, offsetX, offsetY };
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
        this.ctx.lineWidth = this.adaptiveSizes.meshVertexLineWidth;

        Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
            try {
                // Draw center vertex
                if (!drawn.has(vertexStr)) {
                    const center = JSON.parse(vertexStr);
                    if (isValidCoordinate(center)) {
                        const pos = this.worldToScreen(center, transform);
                        this.drawVertex(pos, this.adaptiveSizes.vertexRadius);
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
                                this.drawVertex(pos, this.adaptiveSizes.vertexRadius);
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
        this.ctx.lineWidth = this.adaptiveSizes.boundaryLineWidth;
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
                this.drawVertex(screenPos, this.adaptiveSizes.boundaryVertexRadius);
            }
        });
    }

    /**
     * Render reference point information (simplified version)
     * @param {Object} refInfo - Reference point information
     * @param {Object} transform - Transformation parameters
     */
    renderReferencePointInfo(refInfo, transform, boundaryVertices) {
        if (!refInfo || !transform) return;

        // Handle basic reference point rendering
        if (refInfo.ref_vertex && isValidCoordinate(refInfo.ref_vertex)) {
            const refScreenPos = this.worldToScreen(refInfo.ref_vertex, transform);
            this.ctx.fillStyle = '#10B981';
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 1;
            this.drawVertex(refScreenPos, this.adaptiveSizes.referencePointRadius);
        }

        // Handle clicked point for Type1 actions
        if (refInfo.clicked_point && isValidCoordinate(refInfo.clicked_point)) {
            const clickedScreenPos = this.worldToScreen(refInfo.clicked_point, transform);
            this.ctx.fillStyle = '#FF6B6B';
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 1.5;
            this.drawVertex(clickedScreenPos, Math.max(3, this.adaptiveSizes.referencePointRadius * 0.75));
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
     * Calculate adaptive sizes (simplified version)
     */
    calculateAdaptiveSizes(allVertices, transform, context = {}) {
        if (!allVertices || allVertices.length === 0 || !transform) {
            this.adaptiveSizes = {
                vertexRadius: CONSTANTS.VERTEX_RADIUS,
                boundaryVertexRadius: 4,
                boundaryLineWidth: 3,
                meshVertexLineWidth: 1.5,
                referencePointRadius: 8
            };
            return;
        }

        const totalVertexCount = allVertices.length;
        let sizeMultiplier = 1.0;
        
        if (totalVertexCount > 500) {
            sizeMultiplier = 0.4;
        } else if (totalVertexCount > 200) {
            sizeMultiplier = 0.6;
        } else if (totalVertexCount > 100) {
            sizeMultiplier = 0.8;
        } else if (totalVertexCount < 20) {
            sizeMultiplier = 1.3;
        }

        // Scale factor adjustment
        const scale = transform.scale;
        const scaleFactor = Math.max(0.5, Math.min(2.0, scale / 100));
        sizeMultiplier *= scaleFactor;

        // Apply sizing with bounds
        this.adaptiveSizes = {
            vertexRadius: Math.max(1.0, Math.min(12, CONSTANTS.VERTEX_RADIUS * sizeMultiplier)),
            boundaryVertexRadius: Math.max(1.0, Math.min(8, 4 * sizeMultiplier)),
            boundaryLineWidth: Math.max(0.5, Math.min(8, 3 * sizeMultiplier)),
            meshVertexLineWidth: Math.max(0.3, Math.min(4, 1.5 * sizeMultiplier)),
            referencePointRadius: Math.max(2, Math.min(16, 8 * sizeMultiplier))
        };
    }

    /**
     * Enhanced world to screen coordinate conversion with zoom/pan support
     * @param {Array} worldCoords - World coordinates [x, y]
     * @param {Object} transform - Transformation parameters
     * @param {Object} viewState - Optional zoom/pan state
     * @returns {Array} Screen coordinates [x, y]
     */
    worldToScreen(worldCoords, transform, viewState = null) {
        if (!transform) {
            return [0, 0];
        }

        const [x, y] = worldCoords;
        let adjustedScale = transform.scale;
        let adjustedOffsetX = transform.offsetX;
        let adjustedOffsetY = transform.offsetY;

        // Apply zoom/pan adjustments if provided
        if (viewState) {
            if (viewState.zoom && viewState.zoom !== 1.0) {
                adjustedScale *= viewState.zoom;
            }
            if (viewState.pan) {
                adjustedOffsetX += viewState.pan.x;
                adjustedOffsetY += viewState.pan.y;
            }
        }

        return [
            x * adjustedScale + adjustedOffsetX,
            y * adjustedScale + adjustedOffsetY
        ];
    }

    /**
     * Enhanced screen to world coordinate conversion with zoom/pan support
     * @param {number} screenX - Screen X coordinate
     * @param {number} screenY - Screen Y coordinate
     * @param {Object} transform - Transformation parameters
     * @param {Object} viewState - Optional zoom/pan state
     * @returns {Array} World coordinates [x, y]
     */
    screenToWorld(screenX, screenY, transform, viewState = null) {
        if (!transform) {
            return [0, 0];
        }

        let adjustedX = screenX;
        let adjustedY = screenY;
        let adjustedScale = transform.scale;
        let adjustedOffsetX = transform.offsetX;
        let adjustedOffsetY = transform.offsetY;

        // Apply zoom/pan adjustments if provided
        if (viewState) {
            if (viewState.zoom && viewState.zoom !== 1.0) {
                adjustedScale *= viewState.zoom;
            }
            if (viewState.pan) {
                adjustedOffsetX += viewState.pan.x;
                adjustedOffsetY += viewState.pan.y;
            }
        }

        const worldX = (adjustedX - adjustedOffsetX) / adjustedScale;
        const worldY = (adjustedY - adjustedOffsetY) / adjustedScale;
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
     * Enhanced public resize method
     * @param {Object} dimensions - Optional dimensions from ResizeObserver
     */
    onResize(dimensions = null) {
        if (dimensions) {
            this.resizeCanvas(dimensions);
            // Re-render with cached data if available
            if (this.lastRenderData) {
                if (this.lastRenderData.isPreview) {
                    this.renderBoundaryPreview(
                        this.lastRenderData.boundaryVertices,
                        this.lastRenderData.meshName || ''
                    );
                } else {
                    this.renderScene(
                        this.lastRenderData.meshData,
                        this.lastRenderData.boundaryVertices,
                        this.lastRenderData.refPointInfo
                    );
                }
            } else {
                this.clearCanvas();
            }
        } else {
            this.handleResize();
        }
    }

    /**
     * Destroy Canvas renderer and cleanup
     */
    destroy() {
        if (this.resizeDebounceTimer) {
            clearTimeout(this.resizeDebounceTimer);
        }
        this.lastRenderData = null;
        this.currentTransform = null;
    }
}
