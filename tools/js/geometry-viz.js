/**
 * Geometric Coordinate Normalization Visualization Tool JavaScript
 */

class GeometryViz {
    constructor() {
        this.inputCanvas = document.getElementById('inputCanvas');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.inputCtx = this.inputCanvas.getContext('2d');
        this.outputCtx = this.outputCanvas.getContext('2d');
        
        // Coordinate storage
        this.points = [];
        this.normalizedData = null;
        
        // UI elements
        this.pointCountEl = document.getElementById('pointCount');
        this.coordinatesListEl = document.getElementById('coordinatesList');
        this.resultsListEl = document.getElementById('resultsList');
        this.statusMessageEl = document.getElementById('statusMessage');
        this.statusTextEl = document.getElementById('statusText');
        this.clearBtn = document.getElementById('clearBtn');
        this.processBtn = document.getElementById('processBtn');
        
        // Configuration
        this.pointRadius = 6;
        this.apiBaseUrl = 'http://localhost:5000';
        
        this.initEventListeners();
        this.updateUI();
    }
    
    initEventListeners() {
        // Canvas click event
        this.inputCanvas.addEventListener('click', (e) => {
            this.addPoint(e);
        });
        
        // Button events
        this.clearBtn.addEventListener('click', () => {
            this.clearAll();
        });
        
        this.processBtn.addEventListener('click', () => {
            this.processCoordinates();
        });
    }
    
    addPoint(event) {
        const rect = this.inputCanvas.getBoundingClientRect();
        const scaleX = this.inputCanvas.width / rect.width;
        const scaleY = this.inputCanvas.height / rect.height;
        
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;
        
        this.points.push({ x, y });
        this.updateUI();
        this.drawInputCanvas();
    }
    
    clearAll() {
        this.points = [];
        this.normalizedData = null;
        this.updateUI();
        this.drawInputCanvas();
        this.drawOutputCanvas();
        this.hideStatus();
    }
    
    updateUI() {
        // Update point count display
        this.pointCountEl.textContent = this.points.length;
        
        // Update coordinates list
        if (this.points.length === 0) {
            this.coordinatesListEl.textContent = 'No coordinate points';
        } else {
            this.coordinatesListEl.innerHTML = this.points
                .map((point, index) => {
                    const refIndex = Math.floor(this.points.length / 2);
                    const rightNeighborIndex = refIndex - 1;
                    let className = '';
                    let label = '';
                    
                    if (index === refIndex) {
                        className = 'text-green-600 font-semibold';
                        label = ' (Reference Point)';
                    } else if (index === rightNeighborIndex && rightNeighborIndex >= 0) {
                        className = 'text-yellow-600 font-semibold';
                        label = ' (Right Neighbor)';
                    }
                    
                    return `<div class="${className}">${index}: [${point.x.toFixed(1)}, ${point.y.toFixed(1)}]${label}</div>`;
                })
                .join('');
        }
        
        // Update process button state
        const isOdd = this.points.length > 0 && this.points.length % 2 === 1;
        this.processBtn.disabled = !isOdd;
        
        if (this.points.length === 0) {
            this.processBtn.textContent = 'Process Coordinates';
        } else if (this.points.length % 2 === 0) {
            this.processBtn.textContent = `Need odd number of points (Current: ${this.points.length})`;
        } else {
            this.processBtn.textContent = `Process ${this.points.length} coordinates`;
        }
    }
    
    drawInputCanvas() {
        const ctx = this.inputCtx;
        const canvas = this.inputCanvas;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.points.length === 0) return;
        
        const refIndex = Math.floor(this.points.length / 2);
        const rightNeighborIndex = refIndex - 1;
        
        // Draw connecting lines
        if (this.points.length > 1) {
            ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-connecting-line-color').trim() || '#718096';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(this.points[0].x, this.points[0].y);
            
            for (let i = 1; i < this.points.length; i++) {
                ctx.lineTo(this.points[i].x, this.points[i].y);
            }
            ctx.stroke();
        }
        
        // Draw points
        this.points.forEach((point, index) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, this.pointRadius, 0, 2 * Math.PI);
            
            // Set colors
            if (index === refIndex) {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-point-color').trim() || '#22c55e';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-point-stroke').trim() || '#16a34a';
            } else if (index === rightNeighborIndex && rightNeighborIndex >= 0) {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-neighbor-point-color').trim() || '#f59e0b';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-neighbor-point-stroke').trim() || '#d97706';
            } else {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-normal-point-color').trim() || '#ef4444';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-normal-point-stroke').trim() || '#dc2626';
            }
            
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            // Draw point indices
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-text-color').trim() || '#f7fafc';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(index.toString(), point.x, point.y);
        });
    }
    
    drawOutputCanvas() {
        const ctx = this.outputCtx;
        const canvas = this.outputCanvas;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!this.normalizedData || !this.normalizedData.normalized_coordinates) {
            // Draw waiting text
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim() || '#a0aec0';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Waiting for processing results...', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        const normalizedCoords = this.normalizedData.normalized_coordinates;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const maxRadius = Math.min(canvas.width, canvas.height) / 3;
        
        // Draw coordinate axes
        this.drawCoordinateAxes(ctx, centerX, centerY, maxRadius);
        
        // Find maximum radius for scaling
        const maxR = Math.max(...normalizedCoords.map(coord => coord[0]));
        const scale = maxR > 0 ? maxRadius / maxR : 1;
        
        const refIndex = this.normalizedData.ref_vertex_index;
        const rightNeighborIndex = this.normalizedData.right_neighbor_index;
        
        // Convert polar coordinates to Cartesian and draw connecting lines
        const cartesianPoints = normalizedCoords.map(([r, theta]) => ({
            x: centerX + r * scale * Math.cos(theta),
            y: centerY + r * scale * Math.sin(theta)
        }));
        
        if (cartesianPoints.length > 1) {
            ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-connecting-line-color').trim() || '#718096';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cartesianPoints[0].x, cartesianPoints[0].y);
            
            for (let i = 1; i < cartesianPoints.length; i++) {
                ctx.lineTo(cartesianPoints[i].x, cartesianPoints[i].y);
            }
            ctx.stroke();
        }
        
        // Draw points
        cartesianPoints.forEach((point, index) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, this.pointRadius, 0, 2 * Math.PI);
            
            // Set colors
            if (index === refIndex) {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-point-color').trim() || '#22c55e';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-reference-point-stroke').trim() || '#16a34a';
            } else if (index === rightNeighborIndex) {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-neighbor-point-color').trim() || '#f59e0b';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-neighbor-point-stroke').trim() || '#d97706';
            } else {
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-normal-point-color').trim() || '#ef4444';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-normal-point-stroke').trim() || '#dc2626';
            }
            
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            // Draw point indices
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-text-color').trim() || '#f7fafc';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(index.toString(), point.x, point.y);
        });
        
        // Draw origin point
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-origin-color').trim() || '#343a4a';
        ctx.fill();
    }
    
    drawCoordinateAxes(ctx, centerX, centerY, maxRadius) {
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-axis-color').trim() || '#4a5568';
        ctx.lineWidth = 1;
        
        // X axis
        ctx.beginPath();
        ctx.moveTo(centerX - maxRadius, centerY);
        ctx.lineTo(centerX + maxRadius, centerY);
        ctx.stroke();
        
        // Y axis
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - maxRadius);
        ctx.lineTo(centerX, centerY + maxRadius);
        ctx.stroke();
        
        // Draw circular grid
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--canvas-grid-light-color').trim() || 'rgba(255, 255, 255, 0.15)';
        for (let r = maxRadius / 4; r <= maxRadius; r += maxRadius / 4) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
            ctx.stroke();
        }
        
        // Axis labels
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim() || '#a0aec0';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('0', centerX - 10, centerY + 15);
        ctx.fillText('+X', centerX + maxRadius - 10, centerY - 10);
        ctx.fillText('+Y', centerX + 10, centerY - maxRadius + 15);
    }
    
    async processCoordinates() {
        if (this.points.length === 0 || this.points.length % 2 === 0) {
            this.showStatus('error', 'Please add an odd number of coordinate points');
            return;
        }
        
        this.showStatus('info', 'Processing coordinates...');
        this.processBtn.disabled = true;
        
        try {
            const coordinates = this.points.map(point => [point.x, point.y]);
            
            const response = await fetch(`${this.apiBaseUrl}/geometry/normalize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    coordinates: coordinates
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.normalizedData = data;
                this.updateResultsList();
                this.drawOutputCanvas();
                this.showStatus('success', 'Coordinates processed successfully!');
            } else {
                this.showStatus('error', `Processing failed: ${data.message}`);
            }
        } catch (error) {
            console.error('API request failed:', error);
            this.showStatus('error', `Network error: ${error.message}`);
        } finally {
            this.processBtn.disabled = false;
        }
    }
    
    updateResultsList() {
        if (!this.normalizedData) {
            this.resultsListEl.textContent = 'Waiting for processing...';
            return;
        }
        
        const coords = this.normalizedData.normalized_coordinates;
        const refIndex = this.normalizedData.ref_vertex_index;
        const rightNeighborIndex = this.normalizedData.right_neighbor_index;
        
        this.resultsListEl.innerHTML = coords
            .map(([r, theta], index) => {
                let className = '';
                let label = '';
                
                if (index === refIndex) {
                    className = 'text-green-600 font-semibold';
                    label = ' (Reference Point)';
                } else if (index === rightNeighborIndex) {
                    className = 'text-yellow-600 font-semibold';
                    label = ' (Right Neighbor)';
                }
                
                const degrees = (theta * 180 / Math.PI).toFixed(1);
                return `<div class="${className}">${index}: [${r.toFixed(3)}, ${theta.toFixed(3)} (${degrees}°)]${label}</div>`;
            })
            .join('');
        
        // Add processing information
        this.resultsListEl.innerHTML += `
            <div class="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                Scale Factor: ${this.normalizedData.scale_factor.toFixed(4)}<br>
                Average Edge Length: ${this.normalizedData.average_edge_length.toFixed(2)}<br>
                Edges Used for Scale: ${this.normalizedData.edges_used_for_scale}
            </div>
        `;
    }
    
    showStatus(type, message) {
        this.statusMessageEl.className = `p-4 rounded-lg border ${type === 'success' ? 'status-success' : type === 'error' ? 'status-error' : 'bg-blue-50 text-blue-700'}`;
        this.statusTextEl.textContent = message;
        this.statusMessageEl.classList.remove('hidden');
        
        // Auto-hide success messages
        if (type === 'success') {
            setTimeout(() => {
                this.hideStatus();
            }, 3000);
        }
    }
    
    hideStatus() {
        this.statusMessageEl.classList.add('hidden');
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    new GeometryViz();
});