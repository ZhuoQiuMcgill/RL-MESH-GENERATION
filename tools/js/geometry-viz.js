/**
 * 几何坐标归一化可视化工具 JavaScript
 */

class GeometryViz {
    constructor() {
        this.inputCanvas = document.getElementById('inputCanvas');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.inputCtx = this.inputCanvas.getContext('2d');
        this.outputCtx = this.outputCanvas.getContext('2d');
        
        // 坐标存储
        this.points = [];
        this.normalizedData = null;
        
        // UI 元素
        this.pointCountEl = document.getElementById('pointCount');
        this.coordinatesListEl = document.getElementById('coordinatesList');
        this.resultsListEl = document.getElementById('resultsList');
        this.statusMessageEl = document.getElementById('statusMessage');
        this.statusTextEl = document.getElementById('statusText');
        this.clearBtn = document.getElementById('clearBtn');
        this.processBtn = document.getElementById('processBtn');
        
        // 配置
        this.pointRadius = 6;
        this.apiBaseUrl = 'http://localhost:5000';
        
        this.initEventListeners();
        this.updateUI();
    }
    
    initEventListeners() {
        // 画布点击事件
        this.inputCanvas.addEventListener('click', (e) => {
            this.addPoint(e);
        });
        
        // 按钮事件
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
        // 更新点数显示
        this.pointCountEl.textContent = this.points.length;
        
        // 更新坐标列表
        if (this.points.length === 0) {
            this.coordinatesListEl.textContent = '暂无坐标点';
        } else {
            this.coordinatesListEl.innerHTML = this.points
                .map((point, index) => {
                    const refIndex = Math.floor(this.points.length / 2);
                    const rightNeighborIndex = refIndex - 1;
                    let className = '';
                    let label = '';
                    
                    if (index === refIndex) {
                        className = 'text-green-600 font-semibold';
                        label = ' (参考点)';
                    } else if (index === rightNeighborIndex && rightNeighborIndex >= 0) {
                        className = 'text-yellow-600 font-semibold';
                        label = ' (右邻居)';
                    }
                    
                    return `<div class="${className}">${index}: [${point.x.toFixed(1)}, ${point.y.toFixed(1)}]${label}</div>`;
                })
                .join('');
        }
        
        // 更新处理按钮状态
        const isOdd = this.points.length > 0 && this.points.length % 2 === 1;
        this.processBtn.disabled = !isOdd;
        
        if (this.points.length === 0) {
            this.processBtn.textContent = '处理坐标';
        } else if (this.points.length % 2 === 0) {
            this.processBtn.textContent = `需要奇数个点 (当前: ${this.points.length})`;
        } else {
            this.processBtn.textContent = `处理 ${this.points.length} 个坐标`;
        }
    }
    
    drawInputCanvas() {
        const ctx = this.inputCtx;
        const canvas = this.inputCanvas;
        
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.points.length === 0) return;
        
        const refIndex = Math.floor(this.points.length / 2);
        const rightNeighborIndex = refIndex - 1;
        
        // 绘制连接线
        if (this.points.length > 1) {
            ctx.strokeStyle = '#6b7280';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(this.points[0].x, this.points[0].y);
            
            for (let i = 1; i < this.points.length; i++) {
                ctx.lineTo(this.points[i].x, this.points[i].y);
            }
            ctx.stroke();
        }
        
        // 绘制点
        this.points.forEach((point, index) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, this.pointRadius, 0, 2 * Math.PI);
            
            // 设置颜色
            if (index === refIndex) {
                ctx.fillStyle = '#22c55e';
                ctx.strokeStyle = '#16a34a';
            } else if (index === rightNeighborIndex && rightNeighborIndex >= 0) {
                ctx.fillStyle = '#f59e0b';
                ctx.strokeStyle = '#d97706';
            } else {
                ctx.fillStyle = '#ef4444';
                ctx.strokeStyle = '#dc2626';
            }
            
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            // 绘制点的索引
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(index.toString(), point.x, point.y);
        });
    }
    
    drawOutputCanvas() {
        const ctx = this.outputCtx;
        const canvas = this.outputCanvas;
        
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!this.normalizedData || !this.normalizedData.normalized_coordinates) {
            // 绘制等待文本
            ctx.fillStyle = '#6b7280';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('等待处理结果...', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        const normalizedCoords = this.normalizedData.normalized_coordinates;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const maxRadius = Math.min(canvas.width, canvas.height) / 3;
        
        // 绘制坐标轴
        this.drawCoordinateAxes(ctx, centerX, centerY, maxRadius);
        
        // 找到最大半径用于缩放
        const maxR = Math.max(...normalizedCoords.map(coord => coord[0]));
        const scale = maxR > 0 ? maxRadius / maxR : 1;
        
        const refIndex = this.normalizedData.ref_vertex_index;
        const rightNeighborIndex = this.normalizedData.right_neighbor_index;
        
        // 转换极坐标到笛卡尔坐标并绘制连接线
        const cartesianPoints = normalizedCoords.map(([r, theta]) => ({
            x: centerX + r * scale * Math.cos(theta),
            y: centerY + r * scale * Math.sin(theta)
        }));
        
        if (cartesianPoints.length > 1) {
            ctx.strokeStyle = '#6b7280';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cartesianPoints[0].x, cartesianPoints[0].y);
            
            for (let i = 1; i < cartesianPoints.length; i++) {
                ctx.lineTo(cartesianPoints[i].x, cartesianPoints[i].y);
            }
            ctx.stroke();
        }
        
        // 绘制点
        cartesianPoints.forEach((point, index) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, this.pointRadius, 0, 2 * Math.PI);
            
            // 设置颜色
            if (index === refIndex) {
                ctx.fillStyle = '#22c55e';
                ctx.strokeStyle = '#16a34a';
            } else if (index === rightNeighborIndex) {
                ctx.fillStyle = '#f59e0b';
                ctx.strokeStyle = '#d97706';
            } else {
                ctx.fillStyle = '#ef4444';
                ctx.strokeStyle = '#dc2626';
            }
            
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            // 绘制点的索引
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(index.toString(), point.x, point.y);
        });
        
        // 绘制原点
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#1f2937';
        ctx.fill();
    }
    
    drawCoordinateAxes(ctx, centerX, centerY, maxRadius) {
        ctx.strokeStyle = '#d1d5db';
        ctx.lineWidth = 1;
        
        // X轴
        ctx.beginPath();
        ctx.moveTo(centerX - maxRadius, centerY);
        ctx.lineTo(centerX + maxRadius, centerY);
        ctx.stroke();
        
        // Y轴
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - maxRadius);
        ctx.lineTo(centerX, centerY + maxRadius);
        ctx.stroke();
        
        // 绘制圆形网格
        ctx.strokeStyle = '#e5e7eb';
        for (let r = maxRadius / 4; r <= maxRadius; r += maxRadius / 4) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
            ctx.stroke();
        }
        
        // 轴标签
        ctx.fillStyle = '#6b7280';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('0', centerX - 10, centerY + 15);
        ctx.fillText('+X', centerX + maxRadius - 10, centerY - 10);
        ctx.fillText('+Y', centerX + 10, centerY - maxRadius + 15);
    }
    
    async processCoordinates() {
        if (this.points.length === 0 || this.points.length % 2 === 0) {
            this.showStatus('error', '请添加奇数个坐标点');
            return;
        }
        
        this.showStatus('info', '正在处理坐标...');
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
                this.showStatus('success', '坐标处理成功！');
            } else {
                this.showStatus('error', `处理失败: ${data.message}`);
            }
        } catch (error) {
            console.error('API请求失败:', error);
            this.showStatus('error', `网络错误: ${error.message}`);
        } finally {
            this.processBtn.disabled = false;
        }
    }
    
    updateResultsList() {
        if (!this.normalizedData) {
            this.resultsListEl.textContent = '等待处理...';
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
                    label = ' (参考点)';
                } else if (index === rightNeighborIndex) {
                    className = 'text-yellow-600 font-semibold';
                    label = ' (右邻居)';
                }
                
                const degrees = (theta * 180 / Math.PI).toFixed(1);
                return `<div class="${className}">${index}: [${r.toFixed(3)}, ${theta.toFixed(3)} (${degrees}°)]${label}</div>`;
            })
            .join('');
        
        // 添加处理信息
        this.resultsListEl.innerHTML += `
            <div class="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                缩放因子: ${this.normalizedData.scale_factor.toFixed(4)}<br>
                平均边长: ${this.normalizedData.average_edge_length.toFixed(2)}<br>
                使用边数: ${this.normalizedData.edges_used_for_scale}
            </div>
        `;
    }
    
    showStatus(type, message) {
        this.statusMessageEl.className = `p-4 rounded-lg border ${type === 'success' ? 'status-success' : type === 'error' ? 'status-error' : 'bg-blue-50 text-blue-700'}`;
        this.statusTextEl.textContent = message;
        this.statusMessageEl.classList.remove('hidden');
        
        // 自动隐藏成功消息
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

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new GeometryViz();
});