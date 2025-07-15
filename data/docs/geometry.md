# Geometry模块API文档

> **版本**: 1.0.0  
> **作者**: ZhuoQiuMcgill  
> **最后更新**: 2025-01-15

## 概述

Geometry模块提供网格生成所需的几何数据结构和操作功能，主要包含两个核心类：

- `Boundary`: 表示顺时针排列的封闭多边形边界
- `Mesh`: 表示基于边界构建的网格结构

## 数据类型定义

```python
# 基础类型
Point = Tuple[float, float]  # 点坐标 (x, y)
Vertex = Tuple[float, float]  # 顶点坐标 (x, y)
Edge = Tuple[Point, Point]  # 边，由两个点组成
VertexList = List[Point]  # 顶点列表
```

---

## Boundary类

### 构造函数

#### `__init__(self, vertices: List[Tuple[float, float]])`

**功能**: 初始化边界对象

**参数**:

- `vertices: List[Tuple[float, float]]` - 顶点坐标列表，按顺时针顺序排列

**异常**:

- `ValueError` - 当顶点数量少于3个时抛出

**示例**:

```python
vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
boundary = Boundary(vertices)
```

---

### 只读查询方法

#### `get_vertices(self) -> List[Tuple[float, float]]`

**功能**: 返回边界顶点的副本

**参数**: 无

**返回值**: `List[Tuple[float, float]]` - 顶点坐标列表的深拷贝

#### `get_vertex_index(self, v: Tuple[float, float]) -> int`

**功能**: 获取指定顶点在边界中的索引

**参数**:

- `v: Tuple[float, float]` - 要查找的顶点坐标

**返回值**: `int` - 顶点索引，如果未找到返回-1

#### `get_vertex_by_index(self, n: int) -> Tuple[float, float]`

**功能**: 根据索引获取顶点坐标

**参数**:

- `n: int` - 顶点索引（支持负数和越界索引，会自动取模）

**返回值**: `Tuple[float, float]` - 顶点坐标

**异常**:

- `TypeError` - 当索引不是整数时
- `IndexError` - 当边界为空时

#### `get_edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]`

**功能**: 获取边界的所有边

**参数**: 无

**返回值**: `List[Tuple[Tuple[float, float], Tuple[float, float]]]` - 边列表，每条边由两个顶点组成

#### `get_closest_edge_distance(self, vertex: Tuple[float, float], ignore_edges: Set[Edge]) -> float`

**功能**: 计算顶点到边界最近边的距离

**参数**:

- `vertex: Tuple[float, float]` - 目标顶点
- `ignore_edges: Set[Edge]` - 要忽略的边集合

**返回值**: `float` - 到最近边的距离

#### `size(self) -> int`

**功能**: 获取边界顶点数量

**参数**: 无

**返回值**: `int` - 顶点数量

---

### 内角计算方法

#### `get_ref_vertex(self) -> int`

**功能**: 获取具有最小平均内角的参考顶点索引

**参数**: 无

**返回值**: `int` - 参考顶点在边界中的索引

#### `get_avg_interior_angle(self, n: int) -> float`

**功能**: 计算指定顶点的平均内角

**参数**:

- `n: int` - 顶点索引

**返回值**: `float` - 平均内角（度数，0-360）

---

### 边界修改方法

#### `remove_vertex(self, vertex: Tuple[float, float]) -> None`

**功能**: 移除指定顶点

**参数**:

- `vertex: Tuple[float, float]` - 要移除的顶点坐标

**返回值**: `None`

**注意**: 如果顶点不存在，静默忽略

#### `insert_vertex(self, vertex: Tuple[float, float], position: int) -> None`

**功能**: 在指定位置插入顶点

**参数**:

- `vertex: Tuple[float, float]` - 要插入的顶点坐标
- `position: int` - 插入位置索引（0 ≤ pos ≤ len）

**返回值**: `None`

**异常**:

- `IndexError` - 当位置超出范围时

---

### 几何判断方法

#### `part_of_boundary(self, vertex: Tuple[float, float]) -> bool`

**功能**: 检查点是否为边界顶点

**参数**:

- `vertex: Tuple[float, float]` - 要检查的点坐标

**返回值**: `bool` - 如果点是边界顶点返回True，否则返回False

#### `vertex_inside_boundary(self, vertex: Tuple[float, float]) -> bool`

**功能**: 检查点是否在边界内部（严格内部，不包括边界）

**参数**:

- `vertex: Tuple[float, float]` - 要检查的点坐标

**返回值**: `bool` - 如果点在边界内部返回True，否则返回False

**判断规则**:

- 边界组成点 → False
- 位于边界线上的点 → False
- 边界内部的点 → True
- 边界外部的点 → False

#### `edge_cross(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool`

**功能**: 检查边是否与边界相交

**参数**:

- `edge: Tuple[Tuple[float, float], Tuple[float, float]]` - 要检查的边

**返回值**: `bool` - 如果边与边界相交返回True，否则返回False

**判断规则**:

- 输入边与任意边界边相交 → True
- 输入边的端点位于边界边上（非端点） → True

#### `edge_inside_boundary(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool`

**功能**: 检查边是否完全位于边界内部

**参数**:

- `edge: Tuple[Tuple[float, float], Tuple[float, float]]` - 要检查的边

**返回值**: `bool` - 如果边完全在边界内返回True，否则返回False

**判断规则**:

- 两个端点都在边界内 → True
- 边与边界相交 → False
- 一个端点在边界上，另一个在边界内 → True
- 一个端点在边界上，另一个在边界外 → False

---

### 高级几何方法

#### `get_neighbor_info(self, vertex: Tuple[float, float], n: int) -> dict`

**功能**: 获取指定顶点的局部邻域信息

**参数**:

- `vertex: Tuple[float, float]` - 目标顶点坐标
- `n: int` - 邻居顶点数量（前后各n个）

**返回值**: `dict` - 包含以下键的字典：

- `"local_segment_coords": List[Tuple[float, float]]` - 局部线段坐标列表（2n+1个点）
- `"local_avg_edge_length": float` - 局部平均边长

**异常**:

- `ValueError` - 当顶点不在边界中或边界顶点数不足时

#### `get_area(self) -> float`

**功能**: 计算边界围成多边形的面积

**参数**: 无

**返回值**: `float` - 多边形面积（使用鞋带公式计算）

#### `get_fan_points(self, reference_vertex_index: int, g: int, fan_radius: float) -> List[Tuple[float, float]]`

**功能**: 获取前瞻性扇形区域内的代表点

**参数**:

- `reference_vertex_index: int` - 参考顶点索引
- `g: int` - 扇形切片数量
- `fan_radius: float` - 扇形半径

**返回值**: `List[Tuple[float, float]]` - g个代表点的坐标列表

**异常**:

- `IndexError` - 当参考顶点索引超出范围时
- `ValueError` - 当g不为正数时

**算法**:

1. 以参考顶点为中心，左右邻居定义扇形边界
2. 将扇形均分为g个切片
3. 在每个切片中选择距离参考点最近的边界点
4. 如果切片为空，使用角平分线与边界的交点

---

## Mesh类

### 构造函数

#### `__init__(self, boundary: Boundary)`

**功能**: 基于边界创建网格

**参数**:

- `boundary: Boundary` - 边界对象

**功能**: 自动构建初始的顶点邻接关系字典

---

### 网格修改方法

#### `add_vertex(self, vertex: Tuple[float, float]) -> None`

**功能**: 添加新顶点到网格

**参数**:

- `vertex: Tuple[float, float]` - 顶点坐标

**返回值**: `None`

**异常**:

- `ValueError` - 当顶点已存在时

#### `add_edge(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> None`

**功能**: 在两个顶点之间添加边

**参数**:

- `v1: Tuple[float, float]` - 第一个顶点
- `v2: Tuple[float, float]` - 第二个顶点

**返回值**: `None`

**异常**:

- `ValueError` - 当创建自环、顶点不存在或边已存在时

---

### 查询方法

#### `get_mesh(self) -> Dict[Tuple[float, float], List[Tuple[float, float]]]`

**功能**: 获取网格邻接关系的深拷贝

**参数**: 无

**返回值**: `Dict[Tuple[float, float], List[Tuple[float, float]]]` - 顶点到邻接顶点列表的映射

#### `get_adjacency_dict(self) -> Dict[str, List[List[float]]]`

**功能**: 获取适合前端可视化的邻接关系字典

**参数**: 无

**返回值**: `Dict[str, List[List[float]]]` - 字符串化顶点到邻接顶点的映射

**格式示例**:

```json
{
  "[0.0,0.0]": [
    [
      1.0,
      0.0
    ],
    [
      0.0,
      1.0
    ]
  ],
  "[1.0,0.0]": [
    [
      2.0,
      0.0
    ],
    [
      1.0,
      1.0
    ]
  ]
}
```

#### `get_vertices(self) -> List[Tuple[float, float]]`

**功能**: 获取网格中所有顶点

**参数**: 无

**返回值**: `List[Tuple[float, float]]` - 顶点坐标列表

#### `get_vertex_count(self) -> int`

**功能**: 获取顶点数量

**参数**: 无

**返回值**: `int` - 顶点数量

#### `get_edge_count(self) -> int`

**功能**: 获取边数量

**参数**: 无

**返回值**: `int` - 边数量（每条边只计算一次）

#### `has_vertex(self, vertex: Tuple[float, float]) -> bool`

**功能**: 检查顶点是否在网格中

**参数**:

- `vertex: Tuple[float, float]` - 顶点坐标

**返回值**: `bool` - 如果顶点存在返回True，否则返回False

#### `get_neighbors(self, vertex: Tuple[float, float]) -> List[Tuple[float, float]]`

**功能**: 获取指定顶点的所有邻接顶点

**参数**:

- `vertex: Tuple[float, float]` - 顶点坐标

**返回值**: `List[Tuple[float, float]]` - 邻接顶点列表的深拷贝

**异常**:

- `ValueError` - 当顶点不存在时

---

### 特殊方法

#### `__str__(self) -> str`

**功能**: 返回网格的简单字符串表示

**参数**: 无

**返回值**: `str` - 格式为"Mesh(vertices=N, edges=M)"

#### `__repr__(self) -> str`

**功能**: 返回网格的详细字符串表示

**参数**: 无

**返回值**: `str` - 包含完整邻接关系的详细表示

---

## 使用示例

### 基本边界操作

```python
from src.geometry import Boundary, Mesh

# 创建正方形边界
vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
boundary = Boundary(vertices)

# 基本查询
print(f"顶点数量: {boundary.size()}")
print(f"边界面积: {boundary.get_area()}")

# 几何判断
point_inside = (1.0, 1.0)
point_outside = (3.0, 3.0)
print(f"点{point_inside}在边界内: {boundary.vertex_inside_boundary(point_inside)}")
print(f"点{point_outside}在边界内: {boundary.vertex_inside_boundary(point_outside)}")

# 边界修改
new_vertex = (1.0, 0.0)
boundary.insert_vertex(new_vertex, 1)
print(f"插入顶点后的边界: {boundary.get_vertices()}")
```

### 网格操作

```python
# 创建网格
mesh = Mesh(boundary)
print(f"初始网格: {mesh}")

# 添加内部顶点和边
internal_vertex = (1.0, 1.0)
mesh.add_vertex(internal_vertex)
mesh.add_edge((0.0, 0.0), internal_vertex)
mesh.add_edge((2.0, 0.0), internal_vertex)

# 查询网格信息
print(f"顶点数: {mesh.get_vertex_count()}")
print(f"边数: {mesh.get_edge_count()}")
print(f"邻接关系: {mesh.get_adjacency_dict()}")
```

### 高级几何操作

```python
# 获取参考顶点
ref_idx = boundary.get_ref_vertex()
ref_vertex = boundary.get_vertex_by_index(ref_idx)
print(f"参考顶点: {ref_vertex}")

# 获取邻域信息
neighbor_info = boundary.get_neighbor_info(ref_vertex, n=2)
print(f"局部邻域: {neighbor_info['local_segment_coords']}")
print(f"平均边长: {neighbor_info['local_avg_edge_length']}")

# 获取扇形区域点
fan_points = boundary.get_fan_points(ref_idx, g=5, fan_radius=1.0)
print(f"扇形区域点: {fan_points}")
```

---

## 注意事项

1. **坐标精度**: 所有坐标比较使用1e-10的容差来处理浮点数精度问题
2. **边界方向**: 边界顶点必须按顺时针方向排列
3. **内存管理**: 查询方法返回深拷贝，确保数据安全性
4. **异常处理**: 所有可能的错误情况都会抛出相应异常
5. **性能考虑**: 修改操作直接在原对象上进行，避免不必要的拷贝

---

## 依赖关系

Geometry模块依赖以下工具函数：

- `src.utils.angle`: 角度计算相关函数
- `src.utils.segment`: 线段操作相关函数

所有依赖函数的具体API请参考utils模块文档。