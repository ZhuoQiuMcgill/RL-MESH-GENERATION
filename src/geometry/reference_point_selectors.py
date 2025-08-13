import random

from src.interfaces import ReferencePointSelector
from src.utils import get_avg_interior_angle, get_interior_angle


class RLReferencePointSelector(ReferencePointSelector):
    def __init__(self, n=1):
        super().__init__()
        self.n = n
        self.parameters = ['n']

    def select_reference_point(self, boundary, **info):
        n = info.get('n', self.n)
        predict_mode = info.get('predict_mode', False)
        invalid_point_list = info.get('invalid_point_list', set())

        min_interior_angle = 360
        min_i = 0
        for i in range(boundary.size()):
            if predict_mode:
                if i in invalid_point_list:
                    continue
            avg_interior_angle = get_avg_interior_angle(boundary, i, n)
            if avg_interior_angle < min_interior_angle:
                min_interior_angle = avg_interior_angle
                min_i = i
        return min_i

    def get_interior_angle(self, boundary, ref_index, n=2):
        return get_avg_interior_angle(boundary, ref_index, n)


class RandomReferencePointSelector(ReferencePointSelector):
    def __init__(self, n=1):
        super().__init__()
        self.n = n
        self.parameters = ['n']

    def select_reference_point(self, boundary, **info):
        ref_index = random.randrange(boundary.size())
        return ref_index

    def get_interior_angle(self, boundary, ref_index, n=2):
        v_right = boundary.get_vertex_by_index(ref_index - 1)
        v_center = boundary.get_vertex_by_index(ref_index)
        v_left = boundary.get_vertex_by_index(ref_index + 1)
        return get_interior_angle(v_right, v_center, v_left)


class MinAngleReferenceSelector(ReferencePointSelector):
    def __init__(self, n=1):
        super().__init__()
        self.n = n
        self.parameters = ['n']

    def select_reference_point(self, boundary, **info):
        min_interior_angle = 360
        min_i = 0
        for i in range(boundary.size()):
            v_right = boundary.get_vertex_by_index(i - 1)
            v_center = boundary.get_vertex_by_index(i)
            v_left = boundary.get_vertex_by_index(i + 1)
            angle = get_interior_angle(v_right, v_center, v_left)
            if angle < min_interior_angle:
                min_interior_angle = angle
                min_i = i
        return min_i

    def get_interior_angle(self, boundary, ref_index, n=2):
        v_right = boundary.get_vertex_by_index(ref_index - 1)
        v_center = boundary.get_vertex_by_index(ref_index)
        v_left = boundary.get_vertex_by_index(ref_index + 1)
        return get_interior_angle(v_right, v_center, v_left)
