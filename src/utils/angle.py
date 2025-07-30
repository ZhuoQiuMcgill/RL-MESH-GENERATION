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
        return start_angle <= angle <= end_angle
    else:
        return angle >= start_angle or angle <= end_angle


def normalize_coordinates(vertices,
                          ref_vertex,
                          right_neighbor_vertex,
                          scale_factor):
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


def decode_coordinate(
        ref_vertex,
        right_neighbor_vertex,
        scale_factor,
        new_r,
        new_theta):
    # Absolute angle of the reference direction
    dx, dy = right_neighbor_vertex[0] - ref_vertex[0], right_neighbor_vertex[1] - ref_vertex[1]
    ref_angle = math.atan2(dy, dx)

    # 1. Polar → Cartesian in normalized frame
    sx = new_r * math.cos(new_theta)
    sy = new_r * math.sin(new_theta)

    # 2. Undo scaling
    rx = sx / scale_factor
    ry = sy / scale_factor

    # 3. Undo rotation (rotate by +ref_angle)
    cos_a, sin_a = math.cos(ref_angle), math.sin(ref_angle)
    tx = rx * cos_a - ry * sin_a
    ty = rx * sin_a + ry * cos_a

    # 4. Undo translation
    vx = tx + ref_vertex[0]
    vy = ty + ref_vertex[1]

    return vx, vy


def calculate_polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0

    area = 0.0
    n = len(vertices)

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0


def valid_element_angle(element):
    v0, v1, v2, v3 = element
    a1 = get_interior_angle(v0, v1, v2)
    a2 = get_interior_angle(v1, v2, v3)
    a3 = get_interior_angle(v2, v3, v0)
    a4 = get_interior_angle(v3, v0, v1)
    for angle in [a1, a2, a3, a4]:
        if angle < 0.01 * 180 or angle > 0.99 * 180:
            return False

    return True


def get_avg_interior_angle(boundary, target, n):
    v_center = boundary.get_vertex_by_index(target)
    angles = []
    for i in range(1, n + 1):
        v_right = boundary.get_vertex_by_index(target - n)
        v_left = boundary.get_vertex_by_index(target + n)
        angles.append(get_interior_angle(v_right, v_center, v_left))
    return sum(angles) / len(angles) if len(angles) else 0.0
