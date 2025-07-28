from .action import ActionType
from src.utils import get_interior_angle, euclidean_distance
import math


class ActionType0Left(ActionType):
    def get_element(self, boundary, reference_vertex_idx, *coords):
        v0 = boundary.get_vertex_by_index(reference_vertex_idx)
        v1 = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        v2 = boundary.get_vertex_by_index(reference_vertex_idx - 2)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx + 1)
        return [v0, v3, v2, v1]

    def execute(self, mesh, boundary, reference_vertex_idx, *coords):

        quadrilateral = self.get_element(boundary, reference_vertex_idx)
        v0, v3, v2, v1 = quadrilateral
        try:
            mesh.add_edge(v2, v3)
        except ValueError:
            return None

        boundary.remove_vertex(v0)
        boundary.remove_vertex(v1)

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_idx, *coords, alpha=2, n=2):
        if boundary.size() < 4:
            return False

        quadrilateral = self.get_element(boundary, reference_vertex_idx)
        v0, v3, v2, v1 = quadrilateral
        new_internal_edge = (v2, v3)

        if boundary.edge_cross(new_internal_edge):
            return False

        mid_point = ((v2[0] + v3[0]) / 2, (v2[1] + v3[1]) / 2)
        if not boundary.vertex_inside_boundary(mid_point):
            return False

        return True

    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        quadrilateral = self.get_element(boundary, reference_vertex_idx)
        return self.element_quality(quadrilateral)

    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        # locate the two boundary vertices that form the new edge
        v2 = boundary.get_vertex_by_index(reference_vertex_idx - 2)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx + 1)

        # angles created after insertion
        angle_1 = get_interior_angle(v2, v3, boundary.get_vertex_by_index(reference_vertex_idx + 2))
        angle_2 = get_interior_angle(boundary.get_vertex_by_index(reference_vertex_idx - 3), v2, v3)
        angle_quality = self.calculate_angle_quality(angle_1, angle_2, M_angle)

        target_len = euclidean_distance(v2, v3)
        edge_lengths = target_len
        for i in range(1, 3):
            left_v1 = boundary.get_vertex_by_index(reference_vertex_idx + i)
            left_v2 = boundary.get_vertex_by_index(reference_vertex_idx + i + 1)
            right_v1 = boundary.get_vertex_by_index(reference_vertex_idx - 1 - i)
            right_v2 = boundary.get_vertex_by_index(reference_vertex_idx - 2 - i)
            edge_lengths += euclidean_distance(left_v1, left_v2)
            edge_lengths += euclidean_distance(right_v1, right_v2)
        mean_dist = edge_lengths / 5.0

        # smoothness term
        smoothness = (
            min(mean_dist, target_len) / max(mean_dist, target_len)
            if max(mean_dist, target_len) > 0
            else 0.0
        )

        # final score (negative => penalty, 0 => perfect)
        quality = math.sqrt(angle_quality * smoothness)
        return quality
