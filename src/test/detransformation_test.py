import numpy as np
import math
from ..utils.angle import normalize_coordinates_cartesian, decode_coordinate_cartesian


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


def detransformation_test():
    base_length = 2
    scale = 1 / base_length

    new_point_ca = np.array([1.0, 1.0])
    new_point_po = np.array([1.41421356237, 0.785398])
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])

    input_point = new_point_ca

    p_pj = detransformation(input_point, base_length, p0, p1)
    p_qz = decode_coordinate_cartesian(p0, p1, scale, input_point[0], input_point[1])

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
    print(f"normalized point: {normalize_coordinates_cartesian(vertices, p0, p1, scale)}")


if __name__ == '__main__':
    detransformation_test()
