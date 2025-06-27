import numpy as np
import math
from typing import Tuple, List, Dict


def calculate_reference_vertex(boundary: np.ndarray, nrv: int) -> int:
    """
    Calculate reference vertex with minimum average angle following paper's Equation 1.

    Args:
        boundary: Boundary vertices [N, 2] in clockwise order
        nrv: Number of surrounding vertices to consider

    Returns:
        Index of reference vertex
    """
    n_vertices = len(boundary)
    min_avg_angle = float('inf')
    ref_idx = 0

    for i in range(n_vertices):
        angles = []

        # Calculate angles with surrounding vertices (nrv on each side)
        for j in range(1, min(nrv + 1, n_vertices // 2)):
            left_idx = (i - j) % n_vertices
            right_idx = (i + j) % n_vertices

            v_left = boundary[left_idx]
            v_center = boundary[i]
            v_right = boundary[right_idx]

            angle = calculate_angle(v_left, v_center, v_right)
            angles.append(angle)

        if angles:
            avg_angle = np.mean(angles)
            if avg_angle < min_avg_angle:
                min_avg_angle = avg_angle
                ref_idx = i

    return ref_idx


def get_state_components(boundary: np.ndarray, ref_idx: int, n_neighbors: int,
                         n_fan_points: int, beta_obs: float) -> Dict:
    """
    Get state components for the current boundary and reference vertex following paper's state representation.

    Args:
        boundary: Current boundary vertices (clockwise order)
        ref_idx: Reference vertex index
        n_neighbors: Number of neighbors to include on each side
        n_fan_points: Number of fan points to include
        beta_obs: Observation radius factor

    Returns:
        Dictionary containing state components
    """
    n_vertices = len(boundary)
    ref_vertex = boundary[ref_idx]

    # Get neighboring vertices (clockwise order)
    left_neighbors = []
    right_neighbors = []

    # Right neighbors (clockwise direction)
    for i in range(1, min(n_neighbors + 1, n_vertices)):
        right_idx = (ref_idx + i) % n_vertices
        right_neighbors.append(boundary[right_idx])

    # Left neighbors (counter-clockwise direction)
    for i in range(1, min(n_neighbors + 1, n_vertices)):
        left_idx = (ref_idx - i) % n_vertices
        left_neighbors.append(boundary[left_idx])

    # Pad with zeros if not enough neighbors
    while len(left_neighbors) < n_neighbors:
        left_neighbors.append(ref_vertex)  # Use ref_vertex as fallback
    while len(right_neighbors) < n_neighbors:
        right_neighbors.append(ref_vertex)  # Use ref_vertex as fallback

    # Calculate reference direction (from left to right neighbor)
    if len(right_neighbors) > 0 and len(left_neighbors) > 0:
        # Direction from left neighbor to right neighbor
        direction = right_neighbors[0] - left_neighbors[0]
        if np.linalg.norm(direction) < 1e-8:
            reference_direction = np.array([1.0, 0.0])
        else:
            reference_direction = direction / np.linalg.norm(direction)
    else:
        reference_direction = np.array([1.0, 0.0])

    # Calculate base length L according to paper's Equation 2
    edge_lengths = []
    n_consider = min(n_neighbors, n_vertices // 2)

    for i in range(n_consider):
        # Left side edges
        if i < len(left_neighbors) - 1:
            edge_length = np.linalg.norm(left_neighbors[i + 1] - left_neighbors[i])
            edge_lengths.append(edge_length)

        # Right side edges
        if i < len(right_neighbors) - 1:
            edge_length = np.linalg.norm(right_neighbors[i + 1] - right_neighbors[i])
            edge_lengths.append(edge_length)

    if edge_lengths:
        base_length = np.mean(edge_lengths)
    else:
        # Fallback: use distance to nearest neighbor
        if len(right_neighbors) > 0:
            base_length = np.linalg.norm(right_neighbors[0] - ref_vertex)
        else:
            base_length = 1.0

    # Get fan points within observation radius
    fan_points = []
    obs_radius = beta_obs * base_length

    for i, vertex in enumerate(boundary):
        if i == ref_idx:
            continue
        distance = np.linalg.norm(vertex - ref_vertex)
        if 0 < distance <= obs_radius and len(fan_points) < n_fan_points:
            fan_points.append(vertex)

    # Pad fan points if not enough
    while len(fan_points) < n_fan_points:
        fan_points.append(ref_vertex)

    return {
        'ref_vertex': ref_vertex,
        'left_neighbors': left_neighbors[:n_neighbors],
        'right_neighbors': right_neighbors[:n_neighbors],
        'fan_points': fan_points[:n_fan_points],
        'reference_direction': reference_direction,
        'base_length': base_length
    }


def transform_to_relative_coords(points: List[np.ndarray], ref_vertex: np.ndarray,
                                 reference_direction: np.ndarray) -> List[np.ndarray]:
    """
    Transform points to relative coordinate system with ref_vertex as origin.

    Args:
        points: List of points to transform
        ref_vertex: Reference vertex (origin)
        reference_direction: Reference direction vector

    Returns:
        List of transformed points
    """
    # Normalize reference direction
    ref_dir_norm = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)

    # Create rotation matrix to align reference direction with x-axis
    cos_theta = ref_dir_norm[0]
    sin_theta = ref_dir_norm[1]

    # Rotation matrix (align reference direction with x-axis)
    rotation_matrix = np.array([[cos_theta, sin_theta],
                                [-sin_theta, cos_theta]])

    relative_points = []
    for point in points:
        # Translate to reference vertex
        translated = point - ref_vertex
        # Rotate to align with reference direction
        rotated = rotation_matrix @ translated
        relative_points.append(rotated)

    return relative_points


def calculate_reward_components(element_vertices: np.ndarray,
                                boundary_before: np.ndarray, boundary_after: np.ndarray,
                                area_ratio: float, v_density: float,
                                M_angle: float) -> Tuple[float, float, float]:
    """
    Calculate reward components following paper's reward function.

    Args:
        element_vertices: Vertices of generated element
        boundary_before: Boundary before element generation
        boundary_after: Boundary after element generation
        area_ratio: Current boundary size / original boundary size
        v_density: Density control parameter
        M_angle: Angle threshold for boundary quality

    Returns:
        Tuple of (element_quality, boundary_quality, density_reward)
    """
    # Element quality (η_e)
    eta_e = calculate_element_quality(element_vertices)

    # Boundary quality (η_b)
    eta_b = calculate_boundary_quality(element_vertices, boundary_after, M_angle)

    # Density control (μ_t)
    element_area = calculate_polygon_area(element_vertices)
    mu_t = calculate_density_reward(element_area, boundary_before, v_density)

    return eta_e, eta_b, mu_t


def calculate_element_quality(vertices: np.ndarray) -> float:
    """
    Calculate element quality following paper's Equation 7.

    Args:
        vertices: Element vertices [N, 2]

    Returns:
        Element quality score [0, 1]
    """
    if len(vertices) < 3:
        return 0.0

    n_vertices = len(vertices)

    # Calculate edge lengths
    edges = []
    for i in range(n_vertices):
        edge_length = np.linalg.norm(vertices[(i + 1) % n_vertices] - vertices[i])
        edges.append(edge_length)

    if len(edges) == 0 or min(edges) < 1e-8:
        return 0.0

    # Calculate diagonals for quadrilaterals
    if n_vertices == 4:
        diag1 = np.linalg.norm(vertices[2] - vertices[0])
        diag2 = np.linalg.norm(vertices[3] - vertices[1])
        D_max = max(diag1, diag2)
    else:
        # For triangles, use longest edge
        D_max = max(edges)

    if D_max < 1e-8:
        return 0.0

    # Edge quality
    q_edge = (math.sqrt(2) * min(edges)) / D_max

    # Calculate angles
    angles = []
    for i in range(n_vertices):
        p1 = vertices[(i - 1) % n_vertices]
        p2 = vertices[i]
        p3 = vertices[(i + 1) % n_vertices]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

    # Angle quality
    if len(angles) > 0 and max(angles) > 1e-8:
        min_angle = min(angles)
        max_angle = max(angles)
        q_angle = min_angle / max_angle
    else:
        q_angle = 0.0

    # Combined quality according to paper's formula
    eta_e = math.sqrt(max(0, q_edge * q_angle))

    return min(1.0, max(0.0, eta_e))


def calculate_boundary_quality(element_vertices: np.ndarray,
                               boundary_before: np.ndarray,
                               action_type: int,
                               ref_idx: int,
                               M_angle: float) -> float:
    """
    Calculate boundary quality following the paper's Equation 8.
    This implementation is a faithful reproduction of the paper's formula.

    Args:
        element_vertices: Vertices of the generated element.
        boundary_before: Boundary vertices before the update.
        action_type: The type of action taken (0 for insert, 1/-1 for cut).
        ref_idx: The index of the reference vertex on the boundary_before.
        M_angle: Angle threshold in degrees (typically 60).

    Returns:
        Boundary quality score, typically in [-1, 0].
    """
    if len(boundary_before) < 4:
        return 0.0

    n = len(boundary_before)
    q_dist = 1.0

    # Identify the vertices involved in forming the new boundary angles.
    # These are the vertices of the old boundary that connect to the new element.
    if action_type == 0:  # Type 0 (insert) corresponds to paper's Type 1
        # New vertex is the first vertex of the element.
        V_new = element_vertices[0]
        # Connecting vertices on the old boundary are V_i-1 and V_i+1
        V_prev = boundary_before[(ref_idx - 1) % n]
        V_next = boundary_before[(ref_idx + 1) % n]

        # New angles are formed at V_prev and V_next.
        # Angle at V_prev is formed by (V_i-2, V_i-1, V_new)
        p1 = boundary_before[(ref_idx - 2) % n]
        p2 = V_prev
        p3 = V_new
        zeta1 = calculate_angle(p1, p2, p3)

        # Angle at V_next is formed by (V_new, V_i+1, V_i+2)
        p1 = V_new
        p2 = V_next
        p3 = boundary_before[(ref_idx + 2) % n]
        zeta2 = calculate_angle(p1, p2, p3)

        angles = [zeta1, zeta2]

        # Calculate q_dist for the new vertex V_new
        d1 = np.linalg.norm(V_new - V_prev)
        d2 = np.linalg.norm(V_new - V_next)

        min_dist_to_edge = float('inf')
        # Find shortest distance from V_new to all other boundary edges.
        for i in range(n):
            # Skip edges connected to the reference vertex
            if i == (ref_idx - 1) % n or i == ref_idx:
                continue

            p_edge1 = boundary_before[i]
            p_edge2 = boundary_before[(i + 1) % n]

            # Calculate distance from point to line segment
            l2 = np.sum((p_edge1 - p_edge2) ** 2)
            if l2 == 0.0:
                dist = np.linalg.norm(V_new - p_edge1)
            else:
                t = max(0, min(1, np.dot(V_new - p_edge1, p_edge2 - p_edge1) / l2))
                projection = p_edge1 + t * (p_edge2 - p_edge1)
                dist = np.linalg.norm(V_new - projection)
            min_dist_to_edge = min(min_dist_to_edge, dist)

        d_min = min_dist_to_edge
        avg_d = (d1 + d2) / 2.0
        if avg_d > 1e-8 and d_min < avg_d:
            q_dist = d_min / avg_d
        else:
            q_dist = 1.0

    else:  # Type 1 or -1 (cut) corresponds to paper's Type 0
        q_dist = 1.0  # As per paper Figure 8a caption
        if action_type == 1:  # backward cut
            # Element is (Vi-2, Vi-1, Vi, Vi+1)
            # New edge is (Vi-2, Vi+1)
            # New angles are at Vi-2 and Vi+1
            p1 = boundary_before[(ref_idx - 3) % n]
            p2 = boundary_before[(ref_idx - 2) % n]
            p3 = boundary_before[(ref_idx + 1) % n]
            zeta1 = calculate_angle(p1, p2, p3)

            p1 = boundary_before[(ref_idx - 2) % n]
            p2 = boundary_before[(ref_idx + 1) % n]
            p3 = boundary_before[(ref_idx + 2) % n]
            zeta2 = calculate_angle(p1, p2, p3)
            angles = [zeta1, zeta2]
        elif action_type == -1:  # forward cut
            # Element is (Vi-1, Vi, Vi+1, Vi+2)
            # New edge is (Vi-1, Vi+2)
            # New angles are at Vi-1 and Vi+2
            p1 = boundary_before[(ref_idx - 2) % n]
            p2 = boundary_before[(ref_idx - 1) % n]
            p3 = boundary_before[(ref_idx + 2) % n]
            zeta1 = calculate_angle(p1, p2, p3)

            p1 = boundary_before[(ref_idx - 1) % n]
            p2 = boundary_before[(ref_idx + 2) % n]
            p3 = boundary_before[(ref_idx + 3) % n]
            zeta2 = calculate_angle(p1, p2, p3)
            angles = [zeta1, zeta2]
        else:  # Should not happen
            angles = [M_angle]

    if not angles:
        return -1.0

    # From paper's Equation 8
    # Using min_angle_quality from just one of the new angles, as per formula `min_{k in {1,2}}`
    min_angle_term = min(min(angles), M_angle)
    angle_quality = min_angle_term / M_angle

    # Final eta_b calculation
    eta_b = math.sqrt(max(0, angle_quality * q_dist)) - 1

    return max(-1.0, min(0.0, eta_b))


def calculate_density_reward(element_area: float, boundary: np.ndarray,
                             v_density: float) -> float:
    """
    Calculate density reward following paper's Equation 9.

    Args:
        element_area: Area of generated element
        boundary: Current boundary
        v_density: Density parameter

    Returns:
        Density reward
    """
    if len(boundary) < 3:
        return 0.0

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

    # Calculate area bounds according to paper
    A_min = v_density * (e_min ** 2)
    kappa = 4.0  # Parameter from paper
    A_max = v_density * ((e_max - e_min) / kappa + e_min) ** 2

    # Density reward according to paper's Equation 9
    if element_area < A_min:
        mu_t = -1.0
    elif A_min <= element_area < A_max:
        if A_max > A_min:
            mu_t = (element_area - A_min) / (A_max - A_min)
        else:
            mu_t = 0.0
    else:
        mu_t = 0.0

    return mu_t


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: Points forming the angle

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0

    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm

    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_polygon_area(vertices: np.ndarray) -> float:
    """
    Calculate area of polygon using shoelace formula.

    Args:
        vertices: Polygon vertices [N, 2]

    Returns:
        Polygon area (always positive)
    """
    if len(vertices) < 3:
        return 0.0

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
        if len(boundary.shape) == 1:
            boundary = boundary.reshape(-1, 2)
        return boundary
    except Exception as e:
        raise ValueError(f"Failed to load domain from {file_path}: {e}")


def generate_action_coordinates(action_vector: np.ndarray, ref_vertex: np.ndarray,
                                reference_direction: np.ndarray,
                                alpha: float, base_length: float) -> np.ndarray:
    """
    Generate absolute coordinates from action vector following paper's coordinate transformation.

    Args:
        action_vector: Action output from agent [type_prob, x, y]
        ref_vertex: Reference vertex (origin)
        reference_direction: Reference direction vector
        alpha: Action radius factor
        base_length: Base length L

    Returns:
        Absolute coordinates of new vertex
    """
    # Extract relative coordinates from action
    x_rel, y_rel = action_vector[1], action_vector[2]

    # Scale by action radius (r = α * L)
    radius = alpha * base_length

    # Normalize reference direction
    ref_dir_norm = reference_direction / (np.linalg.norm(reference_direction) + 1e-8)

    # Create rotation matrix to transform from relative to absolute coordinates
    cos_theta = ref_dir_norm[0]
    sin_theta = ref_dir_norm[1]

    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Apply transformation: scale relative coordinates and rotate to absolute frame
    relative_coords = np.array([x_rel, y_rel]) * radius
    absolute_coords = ref_vertex + rotation_matrix @ relative_coords

    return absolute_coords


def is_element_valid(element_vertices: np.ndarray, boundary: np.ndarray) -> bool:
    """
    Basic element validity check.

    Args:
        element_vertices: Element vertices
        boundary: Current boundary

    Returns:
        True if element is valid
    """
    if len(element_vertices) < 3:
        return False

    # Check for degenerate element (zero area)
    area = calculate_polygon_area(element_vertices)
    return area > 1e-8


def update_boundary(boundary: np.ndarray, element_vertices: np.ndarray) -> np.ndarray:
    """
    Simplified boundary update function for compatibility.
    Note: The main boundary update logic is now in the environment.

    Args:
        boundary: Current boundary
        element_vertices: Generated element vertices

    Returns:
        Updated boundary (same as input for compatibility)
    """
    return boundary
