# 几何坐标归一化可视化工具

## 功能介绍

这是一个前后端分离的几何坐标归一化可视化工具，可以帮助用户：

1. **交互式绘制**: 在左侧画布中点击绘制奇数个点
2. **自动连接**: 点按顺序连接（最后一个点不连回第一个点）
3. **坐标归一化**: 使用 `utils.angle.normalize_coordinates` 函数处理坐标
4. **可视化结果**: 在右侧画布中显示归一化后的极坐标结果

## 核心算法

- **参考点 (ref_vertex)**: 中间点（第 n//2 个点，n为总点数）
- **右邻居点 (right_neighbor_vertex)**: 参考点的前一个点
- **缩放因子 (scale_factor)**: 1 除以参考点左右各两条边的平均长度
- **归一化**: 将坐标转换为以参考点为原点的极坐标系 (r, θ)

## 文件结构

```
├── src/ui/api/geometry.py          # 后端API蓝图
├── tools/geometry_viz.html         # 前端HTML页面
├── tools/js/geometry-viz.js        # 前端JavaScript逻辑
└── README_GEOMETRY_VIZ.md          # 本文档
```

## 使用方法

### 1. 启动后端服务

```bash
cd D:\Projects\RL-MESH-GENERATION
python -m src.ui.app
```

服务将在 `http://localhost:5000` 启动

### 2. 访问前端页面

在浏览器中打开: `file:///D:/Projects/RL-MESH-GENERATION/tools/geometry_viz.html`

### 3. 使用步骤

1. 在左侧画布中点击添加点（必须为奇数个）
2. 观察点的颜色标识：
   - 🔴 红色：普通点
   - 🟢 绿色：参考点（中间点）
   - 🟡 黄色：右邻居点
3. 点击"处理坐标"按钮
4. 查看右侧画布中的归一化结果

## API接口

### POST /geometry/normalize

归一化坐标处理接口

**请求体:**
```json
{
    "coordinates": [[x1, y1], [x2, y2], ..., [xn, yn]]
}
```

**响应:**
```json
{
    "status": "success",
    "original_coordinates": [[x1, y1], ...],
    "normalized_coordinates": [[r1, theta1], [r2, theta2], ...],
    "ref_vertex_index": 2,
    "right_neighbor_index": 1,
    "scale_factor": 0.01,
    "average_edge_length": 100.0,
    "edges_used_for_scale": 4
}
```

### GET /geometry/health

健康检查接口

## 技术特性

- ✅ 实时交互式坐标绘制
- ✅ 自动奇偶数验证
- ✅ 颜色编码的点标识
- ✅ 极坐标可视化
- ✅ 详细的处理信息显示
- ✅ 错误处理和状态反馈
- ✅ 响应式设计界面

## 测试示例

可以使用curl命令测试API：

```bash
curl -X POST http://localhost:5000/geometry/normalize \
  -H "Content-Type: application/json" \
  -d '{"coordinates": [[100, 100], [200, 150], [300, 100], [250, 50], [150, 50]]}'
```

## 颜色说明

- 🔴 **红色点**: 普通顶点
- 🟢 **绿色点**: 参考点（坐标系原点）
- 🟡 **黄色点**: 右邻居点（X轴正方向参考）
- ⚪ **白色点**: 归一化后坐标系原点
- 📏 **灰色线**: 连接线
- 🌐 **网格**: 极坐标网格