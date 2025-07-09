from src.utils.angle import euclidean_distance, get_interior_angle
import math


def get_element_quality(element):
    if element is None or len(element) != 4:
        return 0.0

    # 计算边长
    edges = []
    for i in range(4):
        v1 = element[i]
        v2 = element[(i + 1) % 4]
        edge_length = euclidean_distance(v1, v2)
        edges.append(edge_length)

    # 计算对角线长度
    diag1 = euclidean_distance(element[0], element[2])
    diag2 = euclidean_distance(element[1], element[3])
    max_diagonal = max(diag1, diag2)

    # 计算边质量 q_edge
    min_edge = min(edges)
    q_edge = (math.sqrt(2) * min_edge) / max_diagonal if max_diagonal > 0 else 0

    # 计算内角
    angles = []
    for i in range(4):
        v_prev = element[(i - 1) % 4]
        v_curr = element[i]
        v_next = element[(i + 1) % 4]
        angle = get_interior_angle(v_prev, v_curr, v_next)
        angles.append(angle)

    # 计算角度质量 q_angle
    min_angle = min(angles)
    max_angle = max(angles)
    q_angle = min_angle / max_angle if max_angle > 0 else 0

    # 元素质量
    eta_e = math.sqrt(q_edge * q_angle)
    return min(1.0, max(0.0, eta_e))


def get_boundary_quality(boundary, M_angle):
    """
    计算剩余边界质量 η_b，实现公式(8)

    Args:

    Returns:
        float: 边界质量值（-1到0之间）
    """

    # 计算新形成的角度（这里简化处理）
    # 在实际实现中，需要分析新边界的角度变化
    min_angle = 90.0  # 默认值，实际需要从边界几何计算

    # 角度质量部分
    angle_quality = min(min_angle, M_angle) / M_angle

    # 距离质量部分（如果有新顶点添加）
    q_dist = 1.0  # 简化假设

    eta_b = math.sqrt(angle_quality) * q_dist - 1
    return max(-1.0, min(0.0, eta_b))
