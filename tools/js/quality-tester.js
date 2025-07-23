/**
 * Quality Tester Module
 * Handles quadrilateral drawing and quality calculation functionality
 */

import { CONSTANTS, LOG_TYPES, formatNumber, getTimestamp, safeGetElement, calculateDistance } from './utils.js';
import { ApiClient, withErrorHandling } from './api-client.js';

export class QualityTester {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.vertices = [];
        this.maxVertices = 4;
        this.qualityMethods = [];
        this.currentMethod = null;
        this.isMouseTracking = false;
        this.apiClient = new ApiClient();
        
        this.init();
    }
    
    async init() {
        this.setupCanvas();
        this.setupEventListeners();
        await this.loadQualityMethods();
        this.log('Quality tester initialized', LOG_TYPES.SUCCESS);
    }
    
    setupCanvas() {
        this.canvas = safeGetElement('quality-canvas');
        if (!this.canvas) {
            this.log('Canvas element not found', LOG_TYPES.ERROR);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        
        this.canvas.width = Math.max(800, rect.width - 20);
        this.canvas.height = Math.max(600, rect.height - 20);
        
        // Update canvas size display
        const canvasSize = safeGetElement('canvas-size');
        if (canvasSize) {
            canvasSize.textContent = `${this.canvas.width}×${this.canvas.height}`;
        }
        
        this.drawGrid();
        this.log(`Canvas initialized: ${this.canvas.width}×${this.canvas.height}`, LOG_TYPES.INFO);
    }
    
    setupEventListeners() {
        if (!this.canvas) return;
        
        // Mouse events for drawing
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleRightClick();
        });
        
        // Mouse tracking for coordinates display
        this.canvas.addEventListener('mouseenter', () => {
            this.isMouseTracking = true;
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.isMouseTracking = false;
            const mouseCoords = safeGetElement('mouse-coordinates');
            if (mouseCoords) mouseCoords.textContent = '(0, 0)';
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isMouseTracking) {
                const rect = this.canvas.getBoundingClientRect();
                const x = Math.round(e.clientX - rect.left);
                const y = Math.round(e.clientY - rect.top);
                
                const mouseCoords = safeGetElement('mouse-coordinates');
                if (mouseCoords) mouseCoords.textContent = `(${x}, ${y})`;
            }
        });
        
        // Control buttons
        const clearBtn = safeGetElement('clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearAll());
        }
        
        const calculateBtn = safeGetElement('calculate-btn');
        if (calculateBtn) {
            calculateBtn.addEventListener('click', () => this.calculateQuality());
        }
        
        // Quality method selection
        const methodSelect = safeGetElement('quality-method-select');
        if (methodSelect) {
            methodSelect.addEventListener('change', (e) => this.selectQualityMethod(e.target.value));
        }
        
        // Example shapes
        const exampleButtons = [
            { id: 'square-example', handler: () => this.drawExampleSquare() },
            { id: 'rectangle-example', handler: () => this.drawExampleRectangle() },
            { id: 'rhombus-example', handler: () => this.drawExampleRhombus() }
        ];
        
        exampleButtons.forEach(({ id, handler }) => {
            const btn = safeGetElement(id);
            if (btn) btn.addEventListener('click', handler);
        });
        
        // Clear log
        const clearLogBtn = safeGetElement('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.clearLog());
        }
        
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    async loadQualityMethods() {
        const wrappedRequest = withErrorHandling(async () => {
            return await this.apiClient.request('/quality/methods');
        });
        
        try {
            const data = await wrappedRequest();
            
            if (data.success) {
                this.qualityMethods = data.methods;
                this.populateMethodSelect(data.method_info);
                this.log(`Loaded ${data.count} quality methods`, LOG_TYPES.SUCCESS);
            } else {
                throw new Error(data.error || 'Failed to load quality methods');
            }
        } catch (error) {
            this.log(`Error loading quality methods: ${error.message}`, LOG_TYPES.ERROR);
        }
    }
    
    populateMethodSelect(methodInfo) {
        const select = safeGetElement('quality-method-select');
        if (!select) return;
        
        select.innerHTML = '<option value="">Select a method...</option>';
        
        this.qualityMethods.forEach(method => {
            const option = document.createElement('option');
            option.value = method;
            option.textContent = methodInfo[method]?.full_name || `quality_${method}`;
            select.appendChild(option);
        });
    }
    
    selectQualityMethod(method) {
        this.currentMethod = method;
        
        const methodInfo = safeGetElement('method-info');
        const methodDescription = safeGetElement('method-description');
        
        if (method && methodInfo && methodDescription) {
            methodInfo.classList.remove('hidden');
            methodDescription.textContent = `Selected: quality_${method}`;
        } else if (methodInfo) {
            methodInfo.classList.add('hidden');
        }
        
        this.updateCalculateButton();
        
        if (method) {
            this.log(`Selected quality method: ${method}`, LOG_TYPES.INFO);
            if (this.vertices.length === 4) {
                this.calculateQuality();
            }
        }
    }
    
    handleCanvasClick(e) {
        if (this.vertices.length >= this.maxVertices) {
            this.log('Maximum 4 vertices allowed', LOG_TYPES.WARNING);
            return;
        }
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.vertices.push([x, y]);
        this.log(`Added vertex ${this.vertices.length}: (${Math.round(x)}, ${Math.round(y)})`, LOG_TYPES.SUCCESS);
        
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        
        if (this.vertices.length === 4 && this.currentMethod) {
            this.calculateQuality();
        }
    }
    
    handleRightClick() {
        if (this.vertices.length === 0) {
            this.log('No vertices to remove', LOG_TYPES.WARNING);
            return;
        }
        
        const removed = this.vertices.pop();
        this.log(`Removed vertex: (${Math.round(removed[0])}, ${Math.round(removed[1])})`, LOG_TYPES.INFO);
        
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        this.clearQualityDisplay();
    }
    
    async calculateQuality() {
        if (this.vertices.length !== 4 || !this.currentMethod) {
            this.log('Need 4 vertices and a quality method selected', LOG_TYPES.WARNING);
            return;
        }
        
        const loadingOverlay = safeGetElement('loading-overlay');
        if (loadingOverlay) loadingOverlay.classList.remove('hidden');
        
        const wrappedRequest = withErrorHandling(async () => {
            return await this.apiClient.request('/quality/calculate', {
                method: 'POST',
                body: JSON.stringify({
                    vertices: this.vertices,
                    method: this.currentMethod
                })
            });
        });
        
        try {
            const data = await wrappedRequest();
            
            if (data.success) {
                const score = data.quality_score;
                this.displayQualityScore(score);
                this.log(`Quality score (${this.currentMethod}): ${formatNumber(score)}`, LOG_TYPES.SUCCESS);
            } else {
                throw new Error(data.error || 'Quality calculation failed');
            }
            
        } catch (error) {
            this.log(`Error calculating quality: ${error.message}`, LOG_TYPES.ERROR);
        } finally {
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
        }
    }
    
    displayQualityScore(score) {
        const qualityScore = safeGetElement('quality-score');
        const qualityBar = safeGetElement('quality-bar');
        
        if (qualityScore) {
            qualityScore.textContent = formatNumber(score, 4);
        }
        
        if (qualityBar) {
            const percentage = Math.max(0, Math.min(100, score * 100));
            qualityBar.style.width = `${percentage}%`;
            
            // Color based on quality
            if (score >= 0.8) {
                qualityBar.className = 'bg-success h-2 rounded-full transition-all duration-300';
            } else if (score >= 0.5) {
                qualityBar.className = 'bg-warning h-2 rounded-full transition-all duration-300';
            } else {
                qualityBar.className = 'bg-danger h-2 rounded-full transition-all duration-300';
            }
        }
    }
    
    clearQualityDisplay() {
        const qualityScore = safeGetElement('quality-score');
        const qualityBar = safeGetElement('quality-bar');
        
        if (qualityScore) qualityScore.textContent = '-';
        if (qualityBar) {
            qualityBar.style.width = '0%';
            qualityBar.className = 'bg-primary h-2 rounded-full transition-all duration-300';
        }
    }
    
    updateVertexDisplay() {
        const vertexCount = safeGetElement('vertex-count');
        const vertexList = safeGetElement('vertex-list');
        
        if (vertexCount) {
            vertexCount.textContent = this.vertices.length;
        }
        
        if (vertexList) {
            if (this.vertices.length === 0) {
                vertexList.innerHTML = '<div class="text-gray-400">No vertices added</div>';
            } else {
                vertexList.innerHTML = this.vertices.map((vertex, index) => 
                    `<div class="vertex-item">
                        <span>V${index + 1}:</span>
                        <span class="vertex-coordinates">(${Math.round(vertex[0])}, ${Math.round(vertex[1])})</span>
                    </div>`
                ).join('');
                
                // Add quality-scrollbar class
                vertexList.classList.add('quality-scrollbar');
            }
        }
    }
    
    updateCalculateButton() {
        const calculateBtn = safeGetElement('calculate-btn');
        if (!calculateBtn) return;
        
        const canCalculate = this.vertices.length === 4 && this.currentMethod;
        calculateBtn.disabled = !canCalculate;
    }
    
    clearAll() {
        this.vertices = [];
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        this.clearQualityDisplay();
        this.log('Cleared all vertices', LOG_TYPES.INFO);
    }
    
    drawGrid() {
        if (!this.ctx) return;
        
        this.ctx.strokeStyle = '#f0f0f0';
        this.ctx.lineWidth = 1;
        
        const gridSize = CONSTANTS.GRID_SIZE;
        
        // Draw vertical lines
        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Draw horizontal lines
        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }
    
    redraw() {
        if (!this.ctx) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Redraw grid
        this.drawGrid();
        
        if (this.vertices.length === 0) return;
        
        // Draw lines connecting vertices
        this.ctx.strokeStyle = '#3B82F6';
        this.ctx.lineWidth = 2;
        
        if (this.vertices.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.vertices[0][0], this.vertices[0][1]);
            
            for (let i = 1; i < this.vertices.length; i++) {
                this.ctx.lineTo(this.vertices[i][0], this.vertices[i][1]);
            }
            
            // Close the shape if we have 4 vertices
            if (this.vertices.length === 4) {
                this.ctx.closePath();
                // Fill with transparent color
                this.ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
                this.ctx.fill();
            }
            
            this.ctx.stroke();
        }
        
        // Draw vertices
        this.ctx.fillStyle = '#3B82F6';
        this.vertices.forEach((vertex, index) => {
            this.ctx.beginPath();
            this.ctx.arc(vertex[0], vertex[1], 6, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Draw vertex label
            this.ctx.fillStyle = '#1F2937';
            this.ctx.font = '12px sans-serif';
            this.ctx.fillText(`${index + 1}`, vertex[0] + 10, vertex[1] - 10);
            this.ctx.fillStyle = '#3B82F6';
        });
    }
    
    drawExampleSquare() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const size = 100;
        
        this.vertices = [
            [centerX - size/2, centerY - size/2], // top-left
            [centerX + size/2, centerY - size/2], // top-right
            [centerX + size/2, centerY + size/2], // bottom-right
            [centerX - size/2, centerY + size/2]  // bottom-left
        ];
        
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        this.log('Drew example square', LOG_TYPES.INFO);
        
        if (this.currentMethod) {
            this.calculateQuality();
        }
    }
    
    drawExampleRectangle() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const width = 150;
        const height = 80;
        
        this.vertices = [
            [centerX - width/2, centerY - height/2], // top-left
            [centerX + width/2, centerY - height/2], // top-right
            [centerX + width/2, centerY + height/2], // bottom-right
            [centerX - width/2, centerY + height/2]  // bottom-left
        ];
        
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        this.log('Drew example rectangle', LOG_TYPES.INFO);
        
        if (this.currentMethod) {
            this.calculateQuality();
        }
    }
    
    drawExampleRhombus() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const size = 80;
        
        this.vertices = [
            [centerX, centerY - size],      // top
            [centerX + size, centerY],      // right
            [centerX, centerY + size],      // bottom
            [centerX - size, centerY]       // left
        ];
        
        this.redraw();
        this.updateVertexDisplay();
        this.updateCalculateButton();
        this.log('Drew example rhombus', LOG_TYPES.INFO);
        
        if (this.currentMethod) {
            this.calculateQuality();
        }
    }
    
    handleResize() {
        const container = this.canvas?.parentElement;
        if (!container) return;
        
        const rect = container.getBoundingClientRect();
        const newWidth = Math.max(800, rect.width - 20);
        const newHeight = Math.max(600, rect.height - 20);
        
        if (this.canvas.width !== newWidth || this.canvas.height !== newHeight) {
            this.canvas.width = newWidth;
            this.canvas.height = newHeight;
            
            const canvasSize = safeGetElement('canvas-size');
            if (canvasSize) {
                canvasSize.textContent = `${newWidth}×${newHeight}`;
            }
            
            this.redraw();
        }
    }
    
    log(message, type = LOG_TYPES.INFO) {
        const logContainer = safeGetElement('log-container');
        if (!logContainer) return;
        
        const timestamp = getTimestamp();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const style = this.getLogStyle(type);
        logEntry.innerHTML = `<span class="log-icon" style="color: ${style.color}">${style.icon}</span>[${timestamp}] ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Keep only last 50 log entries
        const entries = logContainer.children;
        if (entries.length > 50) {
            logContainer.removeChild(entries[0]);
        }
    }
    
    getLogStyle(type) {
        const styles = {
            [LOG_TYPES.SUCCESS]: {color: '#059669', icon: '✓'},
            [LOG_TYPES.ERROR]: {color: '#DC2626', icon: '✗'},
            [LOG_TYPES.WARNING]: {color: '#D97706', icon: '⚠'},
            [LOG_TYPES.INFO]: {color: '#6B7280', icon: 'ℹ'}
        };
        
        return styles[type] || styles[LOG_TYPES.INFO];
    }
    
    clearLog() {
        const logContainer = safeGetElement('log-container');
        if (logContainer) {
            logContainer.innerHTML = '<div class="text-gray-500">Ready to draw quadrilateral...</div>';
        }
    }
}