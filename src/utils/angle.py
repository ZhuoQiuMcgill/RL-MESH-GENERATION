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
