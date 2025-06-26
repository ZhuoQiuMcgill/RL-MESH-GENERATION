import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


def calculate_reference_vertex(boundary: np.ndarray, nrv: int = 2) -> int:
    """
    Calculate reference vertex using minimum average angle criterion.
    Uses CLOCKWISE orientation as specified in the paper.

    Args:
        boundary: Array of boundary vertices [N, 2]
        nrv: Number of surrounding vertices to consider

    Returns:
        Index of reference vertex
    """
    n_vertices = len(boundary)
    min_avg_angle = float('inf')
    ref_idx = 0

    for i in range(n_vertices):
        angles = []

        for j in range(1, nrv + 1):
            # CLOCKWISE orientation: left side is negative direction, right side is positive
            left_idx = (i - j) % n_vertices  # Clockwise left (negative direction)
            right_idx = (i + j) % n_vertices  # Clockwise right (positive direction)

            v_left = boundary[left_idx]
            v_center = boundary[i]
            v_right = boundary[right_idx]

            angle = calculate_angle(v_left, v_center, v_right)
            angles.append(angle)

        avg_angle = np.mean(angles)
        if avg_angle < min_avg_angle:
            min_avg_angle = avg_angle
            ref_idx = i

    return ref_idx


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by points p1-p2-p3.

    Args:
        p1, p2, p3: Points as [x, y] arrays

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_state_components(boundary: np.ndarray, ref_idx: int, n: int, g: int,
                         beta: float) -> Dict:
    """
    Extract state components around reference vertex using CLOCKWISE orientation.

    Args:
        boundary: Boundary vertices
        ref_idx: Reference vertex index
        n: Number of neighboring vertices
        g: Number of fan-shaped sectors
        beta: Observation radius factor

    Returns:
        Dictionary containing state components
    """
    n_vertices = len(boundary)
    ref_vertex = boundary[ref_idx]

    # Get neighboring vertices using CLOCKWISE orientation
    left_neighbors = []
    right_neighbors = []

    for i in range(1, n + 1):
        # CLOCKWISE: left is negative direction, right is positive direction
        left_idx = (ref_idx - i) % n_vertices  # Clockwise left
        right_idx = (ref_idx + i) % n_vertices  # Clockwise right
        left_neighbors.append(boundary[left_idx])
        right_neighbors.append(boundary[right_idx])

    # Calculate base length L according to Equation (2) from paper
    L = calculate_base_length_equation2(ref_vertex, left_neighbors, right_neighbors, n)
    Lr = beta * L

    # Get fan-shaped observation points using correct V_{l,1} and V_{r,1}
    if len(left_neighbors) > 0 and len(right_neighbors) > 0:
        v_left_1 = left_neighbors[0]  # V_{l,1}
        v_right_1 = right_neighbors[0]  # V_{r,1}
        fan_points = get_fan_observation_points_corrected(
            boundary, ref_idx, ref_vertex, v_left_1, v_right_1, g, Lr
        )
    else:
        # Fallback for edge cases
        fan_points = [ref_vertex] * g

    # Reference direction should be V_0 -> V_{r,1}
    if len(right_neighbors) > 0:
        reference_direction = right_neighbors[0] - ref_vertex
    else:
        # Fallback
        reference_direction = np.array([1.0, 0.0])

    state_components = {
        'ref_vertex': ref_vertex,
        'left_neighbors': left_neighbors,
        'right_neighbors': right_neighbors,
        'fan_points': fan_points,
        'reference_direction': reference_direction,
        'base_length': L
    }

    return state_components


def calculate_base_length_equation2(ref_vertex: np.ndarray, left_neighbors: List[np.ndarray],
                                    right_neighbors: List[np.ndarray], n: int) -> float:
    """
    Calculate base length L according to Equation (2) from the paper:
    L = (1/2n) * Σ(j=0 to n) |V_{l,j}V_{l,j+1}| + |V_{r,j}V_{r,j+1}|
    where V_{l,0} = V_i = V_{r,0} (reference vertex)

    Args:
        ref_vertex: Reference vertex V_0
        left_neighbors: Left neighbors [V_{l,1}, V_{l,2}, ..., V_{l,n}]
        right_neighbors: Right neighbors [V_{r,1}, V_{r,2}, ..., V_{r,n}]
        n: Number of neighbors

    Returns:
        Base length L
    """
    total_length = 0.0

    # Build complete sequences including reference vertex
    # Left sequence: [V_{l,0}, V_{l,1}, ..., V_{l,n}] where V_{l,0} = ref_vertex
    left_sequence = [ref_vertex] + left_neighbors
    # Right sequence: [V_{r,0}, V_{r,1}, ..., V_{r,n}] where V_{r,0} = ref_vertex
    right_sequence = [ref_vertex] + right_neighbors

    # Calculate sum according to Equation (2): Σ(j=0 to n)
    for j in range(n):
        # Left side: |V_{l,j}V_{l,j+1}|
        if j + 1 < len(left_sequence):
            left_edge_length = np.linalg.norm(left_sequence[j + 1] - left_sequence[j])
            total_length += left_edge_length

        # Right side: |V_{r,j}V_{r,j+1}|
        if j + 1 < len(right_sequence):
            right_edge_length = np.linalg.norm(right_sequence[j + 1] - right_sequence[j])
            total_length += right_edge_length

    # Apply the coefficient (1/2n)
    L = total_length / (2 * n) if n > 0 else 1.0

    return L


def get_fan_observation_points_corrected(boundary: np.ndarray, ref_idx: int,
                                         ref_vertex: np.ndarray, v_left_1: np.ndarray,
                                         v_right_1: np.ndarray, g: int, radius: float) -> List[np.ndarray]:
    """
    Get observation points in fan-shaped sectors according to Figure 6 in the paper.
    The fan area should be within angle ∠V_{l,1}V_0V_{r,1} and divided evenly.

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
    n_vertices = len(boundary)

    # Calculate vectors from reference vertex
    vec_left = v_left_1 - ref_vertex
    vec_right = v_right_1 - ref_vertex

    # Calculate angles (using atan2 for proper angle handling)
    angle_left = np.arctan2(vec_left[1], vec_left[0])
    angle_right = np.arctan2(vec_right[1], vec_right[0])

    # Ensure proper angle ordering for clockwise traversal
    # We want to go from right to left in clockwise direction
    if angle_left < angle_right:
        angle_left += 2 * np.pi

    # Calculate angular span of the fan area
    angle_span = angle_left - angle_right

    fan_points = []

    for i in range(g):
        # Calculate angle for this sector (from right to left)
        # Each sector is evenly divided: ζ_1 = ζ_2 = ζ_3 = ... = ζ_g
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


def line_ray_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray,
                          line_p1: np.ndarray, line_p2: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate intersection between ray and line segment.

    Returns:
        Intersection point or None if no intersection
    """
    # Ray: P = ray_origin + t * ray_direction
    # Line: P = line_p1 + s * (line_p2 - line_p1)

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
    # Normalize reference direction
    ref_dir_norm = reference_direction / np.linalg.norm(reference_direction)

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


def calculate_reward_components(element_vertices: np.ndarray,
                                boundary_before: np.ndarray,
                                boundary_after: np.ndarray,
                                area_ratio: float, v_density: float,
                                M_angle: float) -> Tuple[float, float, float]:
    """
    Calculate reward components: element quality, boundary quality, density.

    Args:
        element_vertices: Vertices of generated element [4, 2]
        boundary_before: Boundary before element generation
        boundary_after: Boundary after element generation
        area_ratio: Current area / original area
        v_density: Density control parameter
        M_angle: Angle threshold for boundary quality

    Returns:
        Tuple of (element_quality, boundary_quality, density_reward)
    """
    # Element quality
    eta_e = calculate_element_quality(element_vertices)

    # Boundary quality
    eta_b = calculate_boundary_quality(element_vertices, boundary_after, M_angle)

    # Density control
    element_area = calculate_polygon_area(element_vertices)
    mu_t = calculate_density_reward(element_area, boundary_before, v_density)

    return eta_e, eta_b, mu_t


def calculate_element_quality(vertices: np.ndarray) -> float:
    """
    Calculate element quality based on edge lengths and angles.

    Args:
        vertices: Element vertices [4, 2]

    Returns:
        Element quality score [0, 1]
    """
    # Calculate edge lengths
    edges = []
    for i in range(4):
        edge_length = np.linalg.norm(vertices[(i + 1) % 4] - vertices[i])
        edges.append(edge_length)

    # Calculate diagonals
    diag1 = np.linalg.norm(vertices[2] - vertices[0])
    diag2 = np.linalg.norm(vertices[3] - vertices[1])
    D_max = max(diag1, diag2)

    # Edge quality
    q_edge = (math.sqrt(2) * min(edges)) / D_max

    # Calculate angles
    angles = []
    for i in range(4):
        p1 = vertices[(i - 1) % 4]
        p2 = vertices[i]
        p3 = vertices[(i + 1) % 4]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

    # Angle quality
    min_angle = min(angles)
    max_angle = max(angles)
    q_angle = min_angle / max_angle

    # Combined quality
    eta_e = math.sqrt(q_edge * q_angle)

    return eta_e


def calculate_boundary_quality(element_vertices: np.ndarray,
                               boundary: np.ndarray, M_angle: float) -> float:
    """
    Calculate boundary quality after element removal.

    Args:
        element_vertices: Generated element vertices
        boundary: Updated boundary
        M_angle: Angle threshold

    Returns:
        Boundary quality score [-1, 0]
    """
    # Find new angles formed by element removal
    new_angles = []

    # This is a simplified version - in practice you'd need to identify
    # which boundary vertices are affected by the element removal

    # For now, calculate minimum angle in boundary
    n_boundary = len(boundary)
    boundary_angles = []

    for i in range(n_boundary):
        p1 = boundary[(i - 1) % n_boundary]
        p2 = boundary[i]
        p3 = boundary[(i + 1) % n_boundary]
        angle = calculate_angle(p1, p2, p3)
        boundary_angles.append(angle)

    min_boundary_angle = min(boundary_angles)

    # Distance quality (simplified)
    q_dist = 1.0  # Assume no distance penalty for now

    # Boundary quality calculation
    angle_quality = min(min_boundary_angle, M_angle) / M_angle
    eta_b = math.sqrt(angle_quality) * q_dist - 1

    return eta_b


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
    # Calculate boundary edge statistics
    edge_lengths = []
    n_vertices = len(boundary)

    for i in range(n_vertices):
        edge_length = np.linalg.norm(boundary[(i + 1) % n_vertices] - boundary[i])
        edge_lengths.append(edge_length)

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
        mu_t = (element_area - A_min) / (A_max - A_min)
    else:
        mu_t = 0.0

    return mu_t


def is_element_valid(element_vertices: np.ndarray, boundary: np.ndarray) -> bool:
    """
    Check if generated element is valid (no self-intersection, etc.).

    Args:
        element_vertices: Element vertices [4, 2]
        boundary: Current boundary

    Returns:
        True if element is valid
    """
    # Check for self-intersection
    element_polygon = Polygon(element_vertices)
    if not element_polygon.is_valid:
        return False

    # Check intersection with boundary
    boundary_polygon = Polygon(boundary)

    # Element should be inside boundary
    if not boundary_polygon.contains(element_polygon):
        return False

    return True


def update_boundary(boundary: np.ndarray, element_vertices: np.ndarray) -> np.ndarray:
    """
    Update boundary after element generation.

    Args:
        boundary: Current boundary
        element_vertices: Generated element vertices

    Returns:
        Updated boundary
    """
    # This is a simplified implementation
    # In practice, this requires complex polygon operations

    boundary_polygon = Polygon(boundary)
    element_polygon = Polygon(element_vertices)

    # Subtract element from boundary
    try:
        result = boundary_polygon.difference(element_polygon)

        # Extract exterior coordinates
        if hasattr(result, 'exterior'):
            new_boundary = np.array(result.exterior.coords[:-1])
        else:
            # Handle multipolygon case
            new_boundary = boundary  # Fallback

        return new_boundary

    except Exception:
        # Fallback to original boundary if operation fails
        return boundary


def calculate_polygon_area(vertices: np.ndarray) -> float:
    """
    Calculate area of polygon using shoelace formula.

    Args:
        vertices: Polygon vertices [N, 2]

    Returns:
        Polygon area
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                         for i in range(len(x))))

    return area


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
        return boundary
    except Exception as e:
        raise ValueError(f"Failed to load domain from {file_path}: {e}")


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
    # Extract relative coordinates
    x_rel, y_rel = action_vector[1], action_vector[2]

    # Scale by action radius
    radius = alpha * base_length

    # Transform to absolute coordinates
    ref_dir_norm = reference_direction / np.linalg.norm(reference_direction)

    # Rotation matrix (inverse of the transformation matrix)
    cos_theta = ref_dir_norm[0]
    sin_theta = ref_dir_norm[1]

    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Apply transformation
    relative_coords = np.array([x_rel, y_rel]) * radius
    absolute_coords = ref_vertex + rotation_matrix @ relative_coords

    return absolute_coords
