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


def normalize_coordinates(vertices,
                          ref_vertex,
                          right_neighbor_vertex,
                          scale_factor):
    """
    Convert a list of vertices to normalized polar coordinates
    in the local frame defined by (ref_vertex, right_neighbor_vertex).

    Args:
        vertices (Iterable[Tuple[float, float]]): points to normalize
        ref_vertex (Tuple[float, float]): origin of the local frame (V0)
        right_neighbor_vertex (Tuple[float, float]): V_{r,1} used as +x axis
        scale_factor (float): 1 / base_length for length normalization

    Returns:
        List[Tuple[float, float]]: (r, theta) for each vertex
    """
    # Reference direction vector and its angle
    ref_direction = (
        right_neighbor_vertex[0] - ref_vertex[0],
        right_neighbor_vertex[1] - ref_vertex[1]
    )
    ref_angle = math.atan2(ref_direction[1], ref_direction[0])

    cos_ref, sin_ref = math.cos(-ref_angle), math.sin(-ref_angle)
    normalized = []

    for vertex in vertices:
        if vertex is None:
            normalized.append((0.0, 0.0))
            continue
        vx, vy = vertex
        # translate
        tx, ty = vx - ref_vertex[0], vy - ref_vertex[1]
        # rotate
        rx = tx * cos_ref - ty * sin_ref
        ry = tx * sin_ref + ty * cos_ref
        # scale
        sx, sy = rx * scale_factor, ry * scale_factor
        # polar
        r = math.hypot(sx, sy)
        theta = math.atan2(sy, sx)
        normalized.append((r, theta))

    return normalized


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
