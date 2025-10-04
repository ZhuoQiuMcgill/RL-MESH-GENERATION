"""
Coordinate transformation utilities for mesh generation animation.

This module provides functions to transform local environment coordinates
from absolute mesh space to normalized polar coordinates centered at the
reference point with the x-axis aligned to the previous neighbor.
"""

import numpy as np
from typing import List, Tuple, Optional


def normalize_local_env(
    neighbors: List[List[float]],
    fan_points: List[Optional[List[float]]],
    ref_point_idx: int = 2
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """
    Normalize local environment coordinates to polar representation.
    
    The transformation:
    1. Sets reference point (neighbors[ref_point_idx]) as origin (0, 0)
    2. Aligns x-axis with the vector from ref to previous neighbor (neighbors[ref_point_idx-1])
    3. Transforms all neighbors and fan_points to this new coordinate system
    
    Args:
        neighbors: List of neighbor coordinates, where neighbors[ref_point_idx] is the reference
        fan_points: List of fan point coordinates (can contain None)
        ref_point_idx: Index of reference point in neighbors list (default: 2)
        
    Returns:
        Tuple of (normalized_neighbors, normalized_fan_points)
        - normalized_neighbors: numpy array of shape (n, 2) with transformed neighbor coords
        - normalized_fan_points: list of transformed fan points (None preserved)
    """
    # Extract reference point
    ref_point = np.array(neighbors[ref_point_idx])
    
    # Extract previous neighbor (the one used to define x-axis)
    prev_neighbor_idx = ref_point_idx - 1
    prev_neighbor = np.array(neighbors[prev_neighbor_idx])
    
    # Calculate x-axis direction vector (from ref to prev neighbor)
    x_axis_vec = prev_neighbor - ref_point
    x_axis_length = np.linalg.norm(x_axis_vec)
    
    # Handle edge case where points coincide
    if x_axis_length < 1e-10:
        x_axis_vec = np.array([1.0, 0.0])
        x_axis_length = 1.0
    
    # Normalize to unit vector
    x_axis_unit = x_axis_vec / x_axis_length
    
    # Calculate y-axis (perpendicular, rotated 90° counter-clockwise)
    y_axis_unit = np.array([-x_axis_unit[1], x_axis_unit[0]])
    
    # Create transformation matrix (from old coords to new coords)
    # New coordinate = [x_axis_unit · (p - ref), y_axis_unit · (p - ref)]
    transform_matrix = np.array([x_axis_unit, y_axis_unit])
    
    # Transform all neighbors
    normalized_neighbors = []
    for neighbor in neighbors:
        neighbor_vec = np.array(neighbor) - ref_point
        normalized_coord = transform_matrix @ neighbor_vec
        normalized_neighbors.append(normalized_coord)
    
    normalized_neighbors = np.array(normalized_neighbors)
    
    # Transform fan points (preserve None values)
    normalized_fan_points = []
    for fan_point in fan_points:
        if fan_point is None:
            normalized_fan_points.append(None)
        else:
            fan_vec = np.array(fan_point) - ref_point
            normalized_coord = transform_matrix @ fan_vec
            normalized_fan_points.append(normalized_coord)
    
    return normalized_neighbors, normalized_fan_points


def to_polar_coordinates(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to polar coordinates (r, theta).
    
    Args:
        points: numpy array of shape (n, 2) with [x, y] coordinates
        
    Returns:
        numpy array of shape (n, 2) with [r, theta] coordinates
        theta is in radians, range [-π, π]
    """
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack([r, theta])


def calculate_bounds(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box for a set of points.
    
    Args:
        points: List of [x, y] coordinates
        
    Returns:
        Tuple of (min_x, max_x, min_y, max_y)
    """
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    
    points_array = np.array(points)
    min_x, min_y = points_array.min(axis=0)
    max_x, max_y = points_array.max(axis=0)
    
    return min_x, max_x, min_y, max_y


def scale_to_fit(
    points: List[List[float]],
    target_width: float,
    target_height: float,
    padding_ratio: float = 0.1
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Scale and center points to fit within target dimensions.
    
    Args:
        points: List of [x, y] coordinates
        target_width: Target width for fitting
        target_height: Target height for fitting
        padding_ratio: Fraction of space to leave as padding (default: 0.1)
        
    Returns:
        Tuple of (scaled_points, scale_factor, center_offset)
        - scaled_points: numpy array of transformed points
        - scale_factor: factor used for scaling
        - center_offset: (x, y) offset to center in target space
    """
    if not points:
        return np.array([]), 1.0, (0.0, 0.0)
    
    min_x, max_x, min_y, max_y = calculate_bounds(points)
    
    # Calculate original dimensions
    width = max_x - min_x
    height = max_y - min_y
    
    # Avoid division by zero
    if width < 1e-10:
        width = 1.0
    if height < 1e-10:
        height = 1.0
    
    # Calculate scale factor to fit within target (with padding)
    available_width = target_width * (1 - 2 * padding_ratio)
    available_height = target_height * (1 - 2 * padding_ratio)
    
    scale_x = available_width / width
    scale_y = available_height / height
    scale_factor = min(scale_x, scale_y)
    
    # Calculate center of original points
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Transform points: translate to origin, scale, then translate to target center
    points_array = np.array(points)
    centered_points = points_array - np.array([center_x, center_y])
    scaled_points = centered_points * scale_factor
    
    # Center offset in target space (assuming target is centered at origin)
    center_offset = (0.0, 0.0)
    
    return scaled_points, scale_factor, center_offset


def extract_mesh_edges(mesh_points: dict) -> List[Tuple[List[float], List[float]]]:
    """
    Extract edge list from mesh_points adjacency dictionary.
    
    Args:
        mesh_points: Dictionary mapping vertex coords (as string) to list of adjacent vertices
        
    Returns:
        List of edges, where each edge is a tuple of two [x, y] coordinate lists
    """
    edges = []
    seen_edges = set()
    
    for vertex_str, adjacents in mesh_points.items():
        # Parse vertex coordinates from string "[x,y]"
        vertex = eval(vertex_str)
        
        for adjacent in adjacents:
            # Create edge tuple (sorted to avoid duplicates)
            edge = tuple(sorted([tuple(vertex), tuple(adjacent)]))
            
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append((list(edge[0]), list(edge[1])))
    
    return edges


def get_all_mesh_vertices(mesh_points: dict) -> List[List[float]]:
    """
    Extract all unique vertices from mesh_points dictionary.
    
    Args:
        mesh_points: Dictionary mapping vertex coords (as string) to list of adjacent vertices
        
    Returns:
        List of unique [x, y] coordinate lists
    """
    vertices_set = set()
    
    for vertex_str in mesh_points.keys():
        vertex = eval(vertex_str)
        vertices_set.add(tuple(vertex))
    
    return [list(v) for v in vertices_set]
