from src.interfaces import ActionType
from src.utils import get_interior_angle, euclidean_distance, valid_element_angle
from src.quality import calculate_angle_quality


class ActionType1(ActionType):
    def get_element(self, boundary, reference_vertex_idx, *coords):
        v0 = boundary.get_vertex_by_index(reference_vertex_idx)
        v1 = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx + 1)
        v2 = tuple(coords[0])
        return [v0, v3, v2, v1]

    def execute(self, mesh, boundary, reference_vertex_idx, *coords):
        quadrilateral = self.get_element(boundary, reference_vertex_idx, coords[0])
        v0, v3, v2, v1 = quadrilateral

        try:
            mesh.add_vertex(v2)

            mesh.add_edge(v1, v2)
            mesh.add_edge(v2, v3)

            boundary.remove_vertex(v0)
        except ValueError:
            return None

        v1_idx = boundary.get_vertex_index(v1)
        if v1_idx != -1:
            boundary.insert_vertex(v2, v1_idx + 1)
        else:
            raise RuntimeError("Boundary update failed: V3 not found after removing V0.")

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_idx, *coords, alpha=2, n=2):
        if boundary.size() < 3:
            return False

        quadrilateral = self.get_element(boundary, reference_vertex_idx, coords[0])
        v0, v3, v2, v1 = quadrilateral

        if not valid_element_angle(quadrilateral):
            # logging.warning("Invalid element angle!")
            return False

        # if not boundary.vertex_inside_action_space(v2, reference_vertex_idx, alpha, n):
        #     return False

        if not boundary.vertex_inside_boundary(v2):
            # logging.warning("Vertex not inside boundary.")
            return False

        edge_V1_V2 = (v1, v2)
        edge_V2_V3 = (v2, v3)

        if boundary.edge_cross(edge_V1_V2):
            # logging.warning("Edge V1-V2 crosses existing boundary edge.")
            return False
        if boundary.edge_cross(edge_V2_V3):
            # logging.warning("Edge V2-V3 crosses existing boundary edge.")
            return False

        return True

    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        quadrilateral = self.get_element(boundary, reference_vertex_idx, coords[0])
        return self.element_quality(quadrilateral)

    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        # existing neighbors on the boundary
        v1 = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx + 1)

        # new vertex to insert
        v2 = tuple(coords[0])

        # --- 1. angle-quality term (q1) -----------------------------------------
        # angles are measured at v1 and v3
        angle_1 = get_interior_angle(v2, v3, boundary.get_vertex_by_index(reference_vertex_idx + 2))
        angle_2 = get_interior_angle(boundary.get_vertex_by_index(reference_vertex_idx - 2), v1, v2)

        # angle-quality term (saturated by M_angle)
        angle_quality = calculate_angle_quality(angle_1, angle_2, M_angle)

        # --- 2. smoothness term (q_smooth) --------------------------------------
        # lengths around the new edge: 5 consecutive boundary segments (±2 on each side)
        left_edge_1 = euclidean_distance(v2, v3)  # new edge (part 1)
        right_edge_1 = euclidean_distance(v2, v1)  # new edge (part 2)
        left_edge_2 = euclidean_distance(v3, boundary.get_vertex_by_index(reference_vertex_idx + 2))
        right_edge_2 = euclidean_distance(v1, boundary.get_vertex_by_index(reference_vertex_idx - 2))

        edge_lengths_sum = (left_edge_1 + right_edge_1 + left_edge_2 + right_edge_2)
        mean_dist = edge_lengths_sum / 4.0
        target_len = (left_edge_1 + right_edge_1) / 2.0  # average of two half‑edges

        smoothness = (
            min(mean_dist, target_len) / max(mean_dist, target_len)
            if max(mean_dist, target_len) > 0
            else 0.0
        )

        # --- 3. narrow‑gap term (q2) -------------------------------------------
        # distance from the new point to the closest *other* boundary vertex
        v_ref = boundary.get_vertex_by_index(reference_vertex_idx)  # original vertex being split
        closest_dist = boundary.get_closest_vertex_distance(
            v2, ignore_vertices={}
        )

        half_neighbour_span = 0.5 * (left_edge_1 + right_edge_1)
        if closest_dist < half_neighbour_span and half_neighbour_span > 0:
            q_gap = closest_dist / half_neighbour_span
        else:
            q_gap = 1.0  # no penalty

        # --- 4. final score -----------------------------------------------------
        quality = (angle_quality * smoothness * q_gap) ** (1.0 / 3.0)
        return quality
