import random

from src.interfaces import ReferencePointSelector
from src.utils import get_avg_interior_angle, get_interior_angle


class RLReferencePointSelector(ReferencePointSelector):

    @staticmethod
    def select_reference_point(boundary, **info):
        n = info['n']
        min_interior_angle = 360
        min_i = 0
        for i in range(boundary.size()):
            avg_interior_angle = get_avg_interior_angle(boundary, i, n)
            if avg_interior_angle < min_interior_angle:
                min_interior_angle = avg_interior_angle
                min_i = i
        return min_i


class RandomReferencePointSelector(ReferencePointSelector):
    @staticmethod
    def select_reference_point(boundary, **info):
        ref_index = random.randrange(boundary.size())
        return ref_index


class MinAngleReferenceSelector(ReferencePointSelector):
    @staticmethod
    def select_reference_point(boundary, **info):
        min_interior_angle = 360
        min_i = 0
        for i in range(boundary.size()):
            v_right = boundary.get_ref_vertex(i - 1)
            v_center = boundary.get_ref_vertex(i)
            v_left = boundary.get_vertex(i + 1)
            angle = get_interior_angle(v_left, v_center, v_right)
            if angle < min_interior_angle:
                min_interior_angle = angle
                min_i = i
        return min_i
