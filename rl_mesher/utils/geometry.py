import numpy as np
import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LinearRing
from shapely.ops import unary_union
from shapely import affinity
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def calculate_polygon_area(vertices: np.ndarray) -> float:
    """
    Calculate area of polygon using shoelace formula.

    Args:
        vertices: Polygon vertices [N, 2]

    Returns:
        Polygon area
    """
    if len(vertices) < 3:
        return 0.0

    x = vertices[:, 0]
    y = vertices[:, 1]

    area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                         for i in range(len(x))))

    return area


def calculate_element_quality(element_vertices: np.ndarray) -> float:
    """
    Calculate element quality using aspect ratio and area metrics.

    Args:
        element_vertices: Element vertices [4, 2]

    Returns:
        Quality score [0, 1] where 1 is perfect square
    """
    if len(element_vertices) != 4:
        return 0.0

    # Calculate edge lengths
    edges = []
    for i in range(4):
        edge_length = np.linalg.norm(element_vertices[(i + 1) % 4] - element_vertices[i])
        edges.append(edge_length)

    if min(edges) < 1e-8:
        return 0.0

    # Aspect ratio quality (closer to 1 is better)
    max_edge = max(edges)
    min_edge = min(edges)
    aspect_ratio = min_edge / max_edge

    # Angle quality
    angles = []
    for i in range(4):
        p1 = element_vertices[(i - 1) % 4]
        p2 = element_vertices[i]
        p3 = element_vertices[(i + 1) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        angles.append(angle)

    # Ideal angle is 90 degrees for quadrilateral
    angle_quality = 1.0 - (sum(abs(angle - 90.0) for angle in angles) / (4 * 90.0))
    angle_quality = max(0.0, angle_quality)

    # Area quality (prevent degenerate elements)
    area = calculate_polygon_area(element_vertices)
    expected_area = (sum(edges) / 4) ** 2  # Area of square with same perimeter
    area_quality = min(1.0, area / expected_area) if expected_area > 1e-8 else 0.0

    # Combined quality
    quality = (aspect_ratio * 0.4 + angle_quality * 0.4 + area_quality * 0.2)
    return max(0.0, min(1.0, quality))


def is_element_valid_enhanced(element_vertices: np.ndarray, boundary: np.ndarray) -> bool:
    """
    Enhanced validation for generated elements.

    Args:
        element_vertices: Element vertices [4, 2]
        boundary: Current boundary

    Returns:
        True if element is valid
    """
    try:
        # Basic checks
        if len(element_vertices) != 4:
            return False

        # Check for self-intersection and validity
        element_polygon = Polygon(element_vertices)
        if not element_polygon.is_valid or element_polygon.is_empty:
            return False

        # Check minimum area
        if element_polygon.area < 1e-6:
            return False

        # Create boundary polygon with error handling
        try:
            if len(boundary) < 3:
                return False
            boundary_polygon = Polygon(boundary)
            if not boundary_polygon.is_valid:
                # Try to fix invalid boundary
                boundary_polygon = boundary_polygon.buffer(0)
                if not boundary_polygon.is_valid:
                    return False
        except:
            return False

        # Check if element is substantially within boundary
        intersection = element_polygon.intersection(boundary_polygon)
        if intersection.is_empty:
            return False

        # Element should have significant overlap with boundary (at least 80%)
        overlap_ratio = intersection.area / element_polygon.area
        if overlap_ratio < 0.8:
            return False

        # Check that element vertices are reasonable distance from boundary
        for vertex in element_vertices:
            point = Point(vertex)
            if boundary_polygon.distance(point) > 0.1:  # Allow small tolerance
                # Check if point is inside boundary
                if not boundary_polygon.contains(point) and not boundary_polygon.touches(point):
                    return False

        # Additional quality checks
        quality = calculate_element_quality(element_vertices)
        if quality < 0.1:  # Minimum quality threshold
            return False

        return True

    except Exception:
        return False


def update_boundary_with_polygon_ops(boundary: np.ndarray, element_vertices: np.ndarray) -> np.ndarray:
    """
    Update boundary using proper polygon boolean operations.

    Args:
        boundary: Current boundary
        element_vertices: Generated element vertices

    Returns:
        Updated boundary after removing element
    """
    try:
        # Create polygons
        boundary_polygon = Polygon(boundary)
        element_polygon = Polygon(element_vertices)

        # Ensure boundary polygon is valid
        if not boundary_polygon.is_valid:
            boundary_polygon = boundary_polygon.buffer(0)

        # Ensure element polygon is valid
        if not element_polygon.is_valid:
            element_polygon = element_polygon.buffer(0)

        if not boundary_polygon.is_valid or not element_polygon.is_valid:
            return boundary

        # Subtract element from boundary
        result = boundary_polygon.difference(element_polygon)

        # Handle different result types
        if result.is_empty:
            # If result is empty, return minimal boundary
            return boundary[-3:]  # Return last 3 vertices

        # Extract the largest polygon if multiple polygons result
        if hasattr(result, 'geoms'):
            # MultiPolygon case - take the largest
            largest_poly = max(result.geoms, key=lambda p: p.area if hasattr(p, 'area') else 0)
            if hasattr(largest_poly, 'exterior'):
                coords = np.array(largest_poly.exterior.coords[:-1])
            else:
                return boundary
        elif hasattr(result, 'exterior'):
            # Single Polygon case
            coords = np.array(result.exterior.coords[:-1])
        else:
            # Other geometry types - fallback
            return boundary

        # Validate result
        if len(coords) < 3:
            return boundary

        # Check that resulting boundary has reasonable area
        result_area = calculate_polygon_area(coords)
        original_area = calculate_polygon_area(boundary)

        # Result should be smaller but not too small
        if result_area <= 0 or result_area >= original_area:
            return boundary

        return coords

    except Exception as e:
        # If polygon operations fail, fall back to simpler update
        return _update_boundary_simple_fallback(boundary, element_vertices)


def _update_boundary_simple_fallback(boundary: np.ndarray, element_vertices: np.ndarray) -> np.ndarray:
    """
    Fallback boundary update method using vertex removal.
    """
    try:
        # Find boundary vertices that are close to element vertices
        vertices_to_remove = []
        tolerance = 0.05

        for i, boundary_vertex in enumerate(boundary):
            for element_vertex in element_vertices:
                if np.linalg.norm(boundary_vertex - element_vertex) < tolerance:
                    vertices_to_remove.append(i)
                    break

        # Remove duplicates and sort in reverse order
        vertices_to_remove = sorted(list(set(vertices_to_remove)), reverse=True)

        # Don't remove too many vertices
        if len(vertices_to_remove) >= len(boundary) - 2:
            vertices_to_remove = vertices_to_remove[:len(boundary) - 3]

        # Create new boundary
        new_boundary = boundary.copy()
        for idx in vertices_to_remove:
            if 0 <= idx < len(new_boundary):
                new_boundary = np.delete(new_boundary, idx, axis=0)

        # Ensure minimum vertices
        if len(new_boundary) < 3:
            return boundary

        return new_boundary

    except Exception:
        return boundary


def calculate_boundary_quality(element_vertices: np.ndarray, boundary: np.ndarray,
                               M_angle: float) -> float:
    """
    Calculate boundary quality after element removal.

    Args:
        element_vertices: Generated element vertices
        boundary: Current boundary
        M_angle: Angle threshold

    Returns:
        Boundary quality score [-1, 0]
    """
    try:
        # Simulate boundary after element removal
        updated_boundary = update_boundary_with_polygon_ops(boundary, element_vertices)

        if len(updated_boundary) < 3:
            return -1.0

        # Calculate angles in updated boundary
        n_boundary = len(updated_boundary)
        boundary_angles = []

        for i in range(n_boundary):
            p1 = updated_boundary[(i - 1) % n_boundary]
            p2 = updated_boundary[i]
            p3 = updated_boundary[(i + 1) % n_boundary]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            boundary_angles.append(angle)

        min_boundary_angle = min(boundary_angles)

        # Boundary quality calculation
        angle_quality = min(min_boundary_angle, M_angle) / M_angle
        eta_b = math.sqrt(max(0, angle_quality)) - 1

        return max(-1.0, eta_b)

    except Exception:
        return -0.5  # Neutral penalty if calculation fails


def calculate_density_reward(element_area: float, boundary: np.ndarray,
                             v_density: float) -> float:
    """
    Calculate density reward for mesh completion control.

    Args:
        element_area: Area of generated element
        boundary: Current boundary
        v_density: Density parameter

    Returns:
        Density reward
    """
    try:
        # Calculate boundary edge statistics
        edge_lengths = []
        n_vertices = len(boundary)

        for i in range(n_vertices):
            edge_length = np.linalg.norm(boundary[(i + 1) % n_vertices] - boundary[i])
            edge_lengths.append(edge_length)

        if not edge_lengths:
            return 0.0

        e_min = min(edge_lengths)
        e_max = max(edge_lengths)

        # Calculate area bounds
        A_min = v_density * (e_min ** 2)
        kappa = 4.0  # Parameter from paper
        A_max = v_density * ((e_max - e_min) / kappa + e_min) ** 2

        # Density reward
        if element_area < A_min:
            mu_t = -1.0
        elif A_min <= element_area < A_max:
            mu_t = (element_area - A_min) / (A_max - A_min) if A_max > A_min else 0.0
        else:
            mu_t = 0.0

        return mu_t

    except Exception:
        return 0.0


def find_boundary_neighbors(boundary: np.ndarray, ref_idx: int, n_neighbors: int) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    """
    Find left and right neighboring vertices along boundary.

    Args:
        boundary: Boundary vertices
        ref_idx: Reference vertex index
        n_neighbors: Number of neighbors to find

    Returns:
        Tuple of (left_neighbors, right_neighbors)
    """
    n_vertices = len(boundary)

    left_neighbors = []
    right_neighbors = []

    # Get left neighbors (counter-clockwise)
    for i in range(1, min(n_neighbors + 1, n_vertices)):
        left_idx = (ref_idx - i) % n_vertices
        left_neighbors.append(boundary[left_idx])

    # Get right neighbors (clockwise)
    for i in range(1, min(n_neighbors + 1, n_vertices)):
        right_idx = (ref_idx + i) % n_vertices
        right_neighbors.append(boundary[right_idx])

    return left_neighbors, right_neighbors


def calculate_fan_points(boundary: np.ndarray, ref_idx: int, ref_vertex: np.ndarray,
                         v_left_1: np.ndarray, v_right_1: np.ndarray,
                         g: int, radius: float) -> List[np.ndarray]:
    """
    Calculate fan observation points around reference vertex.

    Args:
        boundary: Boundary vertices
        ref_idx: Reference vertex index
        ref_vertex: Reference vertex V_0
        v_left_1: Left neighbor V_{l,1}
        v_right_1: Right neighbor V_{r,1}
        g: Number of sectors
        radius: Observation radius

    Returns:
        List of observation points
    """
    try:
        n_vertices = len(boundary)

        # Calculate vectors from reference vertex
        vec_left = v_left_1 - ref_vertex
        vec_right = v_right_1 - ref_vertex

        # Calculate angles (using atan2 for proper angle handling)
        angle_left = np.arctan2(vec_left[1], vec_left[0])
        angle_right = np.arctan2(vec_right[1], vec_right[0])

        # Ensure proper angle ordering for clockwise traversal
        if angle_left < angle_right:
            angle_left += 2 * np.pi

        # Calculate angular span of the fan area
        angle_span = angle_left - angle_right

        fan_points = []

        for i in range(g):
            # Calculate angle for this sector (from right to left)
            sector_angle = angle_right + (i + 0.5) * angle_span / g

            # Direction vector for this sector
            sector_direction = np.array([np.cos(sector_angle), np.sin(sector_angle)])

            # Find closest boundary point in this sector
            closest_point = None
            min_distance = float('inf')

            # Check intersection with boundary edges
            for j in range(n_vertices):
                p1 = boundary[j]
                p2 = boundary[(j + 1) % n_vertices]

                # Ray-edge intersection
                intersection = line_ray_intersection(ref_vertex, sector_direction, p1, p2)

                if intersection is not None:
                    distance = np.linalg.norm(intersection - ref_vertex)
                    if distance < min_distance and distance <= radius:
                        min_distance = distance
                        closest_point = intersection

            # If no intersection within radius, use point on bisector at radius distance
            if closest_point is None:
                closest_point = ref_vertex + radius * sector_direction

            fan_points.append(closest_point)

        return fan_points

    except Exception:
        # Fallback fan points
        return [ref_vertex + np.array([0.1, 0.1]) for _ in range(g)]


def line_ray_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray,
                          line_p1: np.ndarray, line_p2: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate intersection between ray and line segment.

    Args:
        ray_origin: Ray starting point
        ray_direction: Ray direction vector
        line_p1: Line segment start point
        line_p2: Line segment end point

    Returns:
        Intersection point or None if no intersection
    """
    try:
        line_dir = line_p2 - line_p1

        # Check for parallel lines
        denominator = ray_direction[0] * line_dir[1] - ray_direction[1] * line_dir[0]
        if abs(denominator) < 1e-10:
            return None

        # Solve: ray_origin + t * ray_direction = line_p1 + s * line_dir
        diff = line_p1 - ray_origin
        t = (diff[0] * line_dir[1] - diff[1] * line_dir[0]) / denominator
        s = (diff[0] * ray_direction[1] - diff[1] * ray_direction[0]) / denominator

        # Check if intersection is valid
        if t >= 0 and 0 <= s <= 1:
            intersection = ray_origin + t * ray_direction
            return intersection

        return None

    except Exception:
        return None


def transform_to_relative_coords(points: List[np.ndarray], origin: np.ndarray,
                                 reference_direction: np.ndarray) -> List[np.ndarray]:
    """
    Transform points to relative coordinate system according to paper specifications.
    Uses V_0 as origin and V_0V_{r,1} as reference direction.

    Args:
        points: Points to transform
        origin: Origin of coordinate system (V_0)
        reference_direction: Reference direction vector (V_0V_{r,1})

    Returns:
        Transformed points in relative coordinates
    """
    try:
        # Normalize reference direction
        ref_dir_norm = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)

        # Rotation matrix to align reference direction with x-axis
        cos_theta = ref_dir_norm[0]
        sin_theta = ref_dir_norm[1]

        # Rotation matrix (correct orientation for coordinate transformation)
        rotation_matrix = np.array([[cos_theta, sin_theta],
                                    [-sin_theta, cos_theta]])

        relative_points = []
        for point in points:
            # Translate to origin
            translated = point - origin
            # Rotate to align with reference direction
            rotated = rotation_matrix @ translated
            relative_points.append(rotated)

        return relative_points

    except Exception:
        # Fallback: return points as-is relative to origin
        return [point - origin for point in points]


def generate_action_coordinates(action_vector: np.ndarray, ref_vertex: np.ndarray,
                                reference_direction: np.ndarray,
                                alpha: float, base_length: float) -> np.ndarray:
    """
    Generate absolute coordinates from action vector.

    Args:
        action_vector: Action output from agent [type_prob, x, y]
        ref_vertex: Reference vertex
        reference_direction: Reference direction vector
        alpha: Action radius factor
        base_length: Base length L

    Returns:
        Absolute coordinates of new vertex
    """
    try:
        # Extract relative coordinates
        x_rel, y_rel = action_vector[1], action_vector[2]

        # Scale by action radius
        radius = alpha * base_length

        # Normalize reference direction
        ref_dir_norm = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)

        # Rotation matrix (inverse of the transformation matrix)
        cos_theta = ref_dir_norm[0]
        sin_theta = ref_dir_norm[1]

        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # Apply transformation
        relative_coords = np.array([x_rel, y_rel]) * radius
        absolute_coords = ref_vertex + rotation_matrix @ relative_coords

        return absolute_coords

    except Exception:
        # Fallback coordinates
        return ref_vertex + np.array([base_length * 0.1, base_length * 0.1])


def load_domain_from_file(file_path: str) -> np.ndarray:
    """
    Load domain boundary from text file.

    Args:
        file_path: Path to domain file

    Returns:
        Boundary vertices array
    """
    try:
        boundary = np.loadtxt(file_path)
        if len(boundary) < 3:
            raise ValueError("Boundary must have at least 3 vertices")
        return boundary
    except Exception as e:
        raise ValueError(f"Failed to load domain from {file_path}: {e}")