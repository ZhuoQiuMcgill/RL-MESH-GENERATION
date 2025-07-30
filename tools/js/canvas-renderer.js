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

        // Dynamic sizing state
        this.adaptiveSizes = {
            vertexRadius: CONSTANTS.VERTEX_RADIUS,
            boundaryVertexRadius: 4,
            boundaryLineWidth: 3,
            meshVertexLineWidth: 1.5,
            referencePointRadius: 8
        };

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
        this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-grid-color').trim() || 'rgba(255, 255, 255, 0.08)';
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

        this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim() || '#a0aec0';
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

        this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-text-secondary').trim() || '#cbd5e0';
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
        this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-mesh-edge-color').trim() || '#6366F1';
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

        this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-mesh-vertex-color').trim() || '#3B82F6';
        this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-mesh-vertex-stroke').trim() || '#1E40AF';
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
        this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-boundary-color').trim() || '#EF4444';
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
        this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-boundary-vertex-color').trim() || '#DC2626';
        boundaryVertices.forEach(vertex => {
            if (isValidCoordinate(vertex)) {
                const screenPos = this.worldToScreen(vertex, transform);
                this.drawVertex(screenPos, this.adaptiveSizes.boundaryVertexRadius);
            }
        });
    }

    /**
     * Render reference point information
     * @param {Object} refInfo - Reference point information
     * @param {Object} transform - Transformation parameters
     */
    renderReferencePointInfo(refInfo, transform, boundaryVertices) {
        if (!refInfo || !transform) return;

        // --- Unified Rendering Logic --- //
        // This function now handles multiple data structures for backward compatibility
        // across predict, training, and history pages.

        // --- Mode 1: New structure from Predict Page (reference_vertex_idx) --- //
        if (refInfo.reference_vertex_idx !== undefined && boundaryVertices && boundaryVertices.length > 0) {
            const refVertexIdx = refInfo.reference_vertex_idx;
            const refVertexCoords = refInfo.reference_vertex_coords;
            if (!isValidCoordinate(refVertexCoords)) return;

            // Render N neighboring edges
            const n = refInfo.selector_info?.config?.n || 1;
            const boundarySize = boundaryVertices.length;

            if (n > 0 && boundarySize > 1) {
                this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-local-env-color').trim() || '#F59E0B';
                this.ctx.lineWidth = Math.max(2.5, this.adaptiveSizes.boundaryLineWidth * 1.5);
                this.ctx.lineCap = 'round';

                for (let i = 0; i < n; i++) {
                    // Left edge
                    const p1_idx_left = (refVertexIdx - i + boundarySize) % boundarySize;
                    const p2_idx_left = (refVertexIdx - i - 1 + boundarySize) % boundarySize;
                    if (boundaryVertices[p1_idx_left] && boundaryVertices[p2_idx_left]) {
                        this.drawLine(this.worldToScreen(boundaryVertices[p1_idx_left], transform), this.worldToScreen(boundaryVertices[p2_idx_left], transform));
                    }

                    // Right edge
                    const p1_idx_right = (refVertexIdx + i + boundarySize) % boundarySize;
                    const p2_idx_right = (refVertexIdx + i + 1 + boundarySize) % boundarySize;
                    if (boundaryVertices[p1_idx_right] && boundaryVertices[p2_idx_right]) {
                        this.drawLine(this.worldToScreen(boundaryVertices[p1_idx_right], transform), this.worldToScreen(boundaryVertices[p2_idx_right], transform));
                    }
                }
            }

            // Highlight the reference point itself (drawn on top)
            const refScreenPos = this.worldToScreen(refVertexCoords, transform);
            this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-color').trim() || '#10B981';
            this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-white').trim() || '#FFFFFF';
            this.ctx.lineWidth = Math.max(1.5, this.adaptiveSizes.meshVertexLineWidth);
            this.drawVertex(refScreenPos, this.adaptiveSizes.referencePointRadius);

        // --- Mode 2: Old structure from History/Training Pages (ref_vertex) --- //
        } else if (refInfo.ref_vertex) {
            const { local_env_vertices, ref_vertex, clicked_point, new_element } = refInfo;

            // Draw local environment edges (original logic)
            if (Array.isArray(local_env_vertices) && local_env_vertices.length > 1) {
                this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-local-env-color').trim() || '#F59E0B';
                this.ctx.lineWidth = Math.max(2, this.adaptiveSizes.boundaryLineWidth * 1.33);
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                const firstPoint = this.worldToScreen(local_env_vertices[0], transform);
                this.ctx.moveTo(firstPoint[0], firstPoint[1]);
                for (let i = 1; i < local_env_vertices.length; i++) {
                    if (isValidCoordinate(local_env_vertices[i])) {
                        this.ctx.lineTo(...this.worldToScreen(local_env_vertices[i], transform));
                    }
                }
                this.ctx.stroke();
            }

            // Highlight reference point (original logic)
            if (isValidCoordinate(ref_vertex)) {
                const refScreenPos = this.worldToScreen(ref_vertex, transform);
                this.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-color').trim() || '#10B981';
                this.ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-white').trim() || '#FFFFFF';
                this.ctx.lineWidth = Math.max(1, this.adaptiveSizes.meshVertexLineWidth);
                this.drawVertex(refScreenPos, this.adaptiveSizes.referencePointRadius);
            }

            // Draw clicked point for Type1 actions (original logic)
            if (clicked_point && isValidCoordinate(clicked_point)) {
                const clickedScreenPos = this.worldToScreen(clicked_point, transform);
                this.ctx.fillStyle = '#FF6B6B';
                this.ctx.strokeStyle = '#FFFFFF';
                this.ctx.lineWidth = 1.5;
                this.drawVertex(clickedScreenPos, Math.max(3, this.adaptiveSizes.referencePointRadius * 0.75));
                if (isValidCoordinate(ref_vertex)) {
                    this.ctx.strokeStyle = '#FF6B6B';
                    this.ctx.lineWidth = 1;
                    this.ctx.setLineDash([5, 5]);
                    this.drawLine(this.worldToScreen(ref_vertex, transform), clickedScreenPos);
                    this.ctx.setLineDash([]);
                }
            }

            // Draw new element if it was generated (original logic)
            if (new_element && Array.isArray(new_element) && new_element.length >= 3) {
                this.ctx.strokeStyle = '#00D2FF';
                this.ctx.fillStyle = 'rgba(0, 210, 255, 0.1)';
                this.ctx.lineWidth = 1.5;
                this.ctx.beginPath();
                this.ctx.moveTo(...this.worldToScreen(new_element[0], transform));
                for (let i = 1; i < new_element.length; i++) {
                    if (isValidCoordinate(new_element[i])) {
                        this.ctx.lineTo(...this.worldToScreen(new_element[i], transform));
                    }
                }
                this.ctx.closePath();
                this.ctx.fill();
                this.ctx.stroke();
            }
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
     * Calculate adaptive sizes based on data density and canvas scale
     * @param {Array} allVertices - All vertices to analyze  
     * @param {Object} transform - Transformation parameters
     * @param {Object} context - Context information {meshData, boundaryVertices, context}
     */
    calculateAdaptiveSizes(allVertices, transform, context = {}) {
        if (!allVertices || allVertices.length === 0 || !transform) {
            // Use default sizes
            this.adaptiveSizes = {
                vertexRadius: CONSTANTS.VERTEX_RADIUS,
                boundaryVertexRadius: 4,
                boundaryLineWidth: 3,
                meshVertexLineWidth: 1.5,
                referencePointRadius: 8
            };
            return;
        }

        const {meshData, boundaryVertices, context: scenarioType} = context;
        
        // Calculate mesh vertex count
        let meshVertexCount = 0;
        if (meshData && typeof meshData === 'object') {
            // Count unique vertices in mesh data
            const uniqueMeshVertices = new Set();
            Object.entries(meshData).forEach(([vertexStr, adjacentVertices]) => {
                uniqueMeshVertices.add(vertexStr);
                if (Array.isArray(adjacentVertices)) {
                    adjacentVertices.forEach(vertex => {
                        uniqueMeshVertices.add(JSON.stringify(vertex));
                    });
                }
            });
            meshVertexCount = uniqueMeshVertices.size;
        }

        const boundaryVertexCount = boundaryVertices ? boundaryVertices.length : 0;
        const totalVertexCount = allVertices.length;

        // Determine scenario and primary vertex count for sizing
        let primaryVertexCount = totalVertexCount;
        let scenario = 'mixed';
        
        if (meshVertexCount > 50 && meshVertexCount > boundaryVertexCount * 3) {
            // Mesh-heavy scenario (like history page with lots of mesh data)
            scenario = 'mesh_heavy';
            primaryVertexCount = meshVertexCount;
        } else if (boundaryVertexCount > 0 && meshVertexCount < boundaryVertexCount) {
            // Boundary-heavy scenario (like boundary preview)
            scenario = 'boundary_heavy';
            primaryVertexCount = boundaryVertexCount;
        }

        // Base sizes
        const baseSizes = {
            vertexRadius: CONSTANTS.VERTEX_RADIUS,
            boundaryVertexRadius: 4,
            boundaryLineWidth: 3,
            meshVertexLineWidth: 1.5,
            referencePointRadius: 8
        };

        // Point-count-based sizing (simpler and more reliable)
        let sizeMultiplier = 1.0;
        
        if (primaryVertexCount > 500) {
            sizeMultiplier = 0.4;  // Very dense
        } else if (primaryVertexCount > 200) {
            sizeMultiplier = 0.6;  // Dense
        } else if (primaryVertexCount > 100) {
            sizeMultiplier = 0.8;  // Medium dense
        } else if (primaryVertexCount > 50) {
            sizeMultiplier = 0.9;  // Slightly dense
        } else if (primaryVertexCount < 20) {
            sizeMultiplier = 1.3;  // Sparse, make larger
        }

        // Scale factor adjustment
        const scale = transform.scale;
        const scaleFactor = Math.max(0.5, Math.min(2.0, scale / 100));
        sizeMultiplier *= scaleFactor;

        // Apply sizing with bounds
        this.adaptiveSizes = {
            vertexRadius: Math.max(1.0, Math.min(12, baseSizes.vertexRadius * sizeMultiplier)),
            boundaryVertexRadius: Math.max(1.0, Math.min(8, baseSizes.boundaryVertexRadius * sizeMultiplier)),
            boundaryLineWidth: Math.max(0.5, Math.min(8, baseSizes.boundaryLineWidth * sizeMultiplier)),
            meshVertexLineWidth: Math.max(0.3, Math.min(4, baseSizes.meshVertexLineWidth * sizeMultiplier)),
            referencePointRadius: Math.max(2, Math.min(16, baseSizes.referencePointRadius * sizeMultiplier))
        };

        // Debug logging
        console.debug('Adaptive sizing:', {
            scenario,
            scenarioType,
            totalVertexCount,
            meshVertexCount,
            boundaryVertexCount,
            primaryVertexCount,
            scale: scale.toFixed(2),
            scaleFactor: scaleFactor.toFixed(2),
            sizeMultiplier: sizeMultiplier.toFixed(2),
            sizes: this.adaptiveSizes
        });
    }

    /**
     * Calculate average distance between adjacent vertices
     * @param {Array} vertices - Array of vertices
     * @returns {number} Average distance
     */
    calculateAverageVertexDistance(vertices) {
        if (vertices.length < 2) return 50; // Default distance

        let totalDistance = 0;
        let count = 0;

        // Calculate distances between consecutive vertices
        for (let i = 0; i < vertices.length - 1; i++) {
            const v1 = vertices[i];
            const v2 = vertices[i + 1];
            
            if (isValidCoordinate(v1) && isValidCoordinate(v2)) {
                const dx = v2[0] - v1[0];
                const dy = v2[1] - v1[1];
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 0) {
                    totalDistance += distance;
                    count++;
                }
            }
        }

        // Also check distance from last to first vertex for closed shapes
        if (count > 0) {
            const first = vertices[0];
            const last = vertices[vertices.length - 1];
            if (isValidCoordinate(first) && isValidCoordinate(last)) {
                const dx = last[0] - first[0];
                const dy = last[1] - first[1];
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance > 0 && distance < totalDistance / count * 2) { // Only if it's reasonable
                    totalDistance += distance;
                    count++;
                }
            }
        }

        return count > 0 ? totalDistance / count : 50;
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
     * Get current adaptive sizes for debugging/inspection
     * @returns {Object} Current adaptive sizes
     */
    getCurrentSizes() {
        return {
            ...this.adaptiveSizes,
            isAdaptive: true,
            timestamp: Date.now()
        };
    }

    /**
     * Set manual size overrides (useful for testing)
     * @param {Object} sizeOverrides - Size overrides
     */
    setSizeOverrides(sizeOverrides) {
        this.adaptiveSizes = {
            ...this.adaptiveSizes,
            ...sizeOverrides
        };
        
        // Re-render if we have cached data
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
        }
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