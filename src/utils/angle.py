import math
import numpy as np


def euclidean_distance(p1, p2):
    """计算两点间欧几里得距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_interior_angle(right, center, left):
    ax, ay = right[0] - center[0], right[1] - center[1]
    bx, by = left[0] - center[0], left[1] - center[1]
    theta = (math.atan2(ay, ax) - math.atan2(by, bx)) % (2 * math.pi)
    return math.degrees(theta)


def is_angle_in_slice(angle: float, start_angle: float, end_angle: float) -> bool:
    """检查角度是否在切片内（处理角度环绕问题）"""

    # 规范化角度到[0, 2π]
    def normalize_angle(a):
        while a < 0:
            a += 2 * np.pi
        while a >= 2 * np.pi:
            a -= 2 * np.pi
        return a

    angle = normalize_angle(angle)
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)

    if start_angle <= end_angle:
        # 正常情况
        return start_angle <= angle <= end_angle
    else:
        # 跨越0点的情况
        return angle >= start_angle or angle <= end_angle


def calculate_base_length(boundary, reference_vertex_idx, n):
    """
    Calculate base length L according to formula (2)
    
    Args:
        boundary: Boundary object with get_vertices() method
        reference_vertex_idx: Reference vertex index
        n: Number of neighbors to consider on each side
        
    Returns:
        float: Base length
    """
    vertices = boundary.get_vertices()
    boundary_size = len(vertices)

    total_length = 0.0
    count = 0

    n = min(n, boundary_size // 2)

    for j in range(n):
        # Left side edge length
        left_idx1 = (reference_vertex_idx - j) % boundary_size
        left_idx2 = (reference_vertex_idx - j - 1) % boundary_size
        left_length = euclidean_distance(vertices[left_idx1], vertices[left_idx2])

        # Right side edge length
        right_idx1 = (reference_vertex_idx + j) % boundary_size
        right_idx2 = (reference_vertex_idx + j + 1) % boundary_size
        right_length = euclidean_distance(vertices[right_idx1], vertices[right_idx2])

        total_length += left_length + right_length
        count += 2

    return total_length / count if count > 0 else 1.0


def normalize_coordinates(vertices, reference_vertex_idx, boundary, n):
    """
    按照论文方法将坐标标准化为以参考点为中心的坐标系统

    Args:
        vertices: 顶点列表
        reference_vertex_idx: 参考顶点索引
        boundary: 边界对象
        n: 邻居数量参数

    Returns:
        list: 标准化后的坐标列表 [(r, theta), ...]
    """
    if len(vertices) <= reference_vertex_idx:
        return []

    reference_vertex = vertices[reference_vertex_idx]
    boundary_size = len(vertices)

    # 获取参考方向：V0 -> Vr,1 (右侧第一个邻居)
    right_neighbor_idx = (reference_vertex_idx + 1) % boundary_size
    right_neighbor = vertices[right_neighbor_idx]

    # 计算参考方向向量
    ref_direction = np.array([
        right_neighbor[0] - reference_vertex[0],
        right_neighbor[1] - reference_vertex[1]
    ])

    # 计算参考方向的角度
    ref_angle = math.atan2(ref_direction[1], ref_direction[0])

    # 计算基础长度作为缩放因子
    base_length = calculate_base_length(boundary, reference_vertex_idx, n)
    scale_factor = 1.0 / base_length if base_length > 0 else 1.0

    normalized_coords = []

    for vertex in vertices:
        # 1. 平移：以参考顶点为原点
        translated = np.array([
            vertex[0] - reference_vertex[0],
            vertex[1] - reference_vertex[1]
        ])

        # 2. 旋转：以V0Vr,1为参考方向（x轴）
        cos_ref = math.cos(-ref_angle)
        sin_ref = math.sin(-ref_angle)
        rotated = np.array([
            translated[0] * cos_ref - translated[1] * sin_ref,
            translated[0] * sin_ref + translated[1] * cos_ref
        ])

        # 3. 缩放：基于基础长度标准化
        scaled = rotated * scale_factor

        # 4. 转换为极坐标（论文要求）
        r = math.sqrt(scaled[0] ** 2 + scaled[1] ** 2)
        theta = math.atan2(scaled[1], scaled[0])

        normalized_coords.append((r, theta))

    return normalized_coords


def calculate_polygon_area(vertices):
    """
    使用鞋带公式计算多边形面积

    Args:
        vertices: 顶点列表

    Returns:
        float: 多边形面积
    """
    if len(vertices) < 3:
        return 0.0

    area = 0.0
    n = len(vertices)

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0
