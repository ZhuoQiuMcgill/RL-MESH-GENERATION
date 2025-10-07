import numpy as np
import math


def detransformation(point, dist, p0, p1):
    # dist = np.linalg.norm(p0 - p1)
    theta = 2 * math.pi - math.atan2((p1 - p0)[1], (p1 - p0)[0])
    original_point = np.empty(2)

    # remove rotation
    original_point[0] = np.cos(theta) * point[0] + np.sin(theta) * point[1]
    original_point[1] = -np.sin(theta) * point[0] + np.cos(theta) * point[1]

    # remove scaling
    original_point *= dist

    # remove translation
    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point


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


def detransformation_test():
    base_length = 2
    scale = 1 / base_length

    new_point_ca = np.array([1.0, 1.0])
    new_point_po = np.array([1.41421356237, 0.785398])
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])

    input_point = new_point_po

    p_pj = detransformation(input_point, base_length, p0, p1)
    p_qz = decode_coordinate(p0, p1, scale, input_point[0], input_point[1])

    print(f"Using input point: {input_point}")
    print(f"Point decode by detransformation: {(round(float(p_pj[0]), 3), round(float(p_pj[1]), 3))}")
    print(f"Point decode by decode_coordinate: {(round(p_qz[0], 3), round(p_qz[1], 3))}")


def normalize_coordinates_test():
    base_length = 2
    scale = 1 / base_length

    new_point_po = np.array([1.41421356237, 0.785398])
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])

    vertices = [(2.0, 2.0)]
    print(f"normalized point: {normalize_coordinates(vertices, p0, p1, scale)}")


if __name__ == '__main__':
    normalize_coordinates_test()