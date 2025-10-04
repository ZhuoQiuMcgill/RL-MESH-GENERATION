"""
Mesh renderer for visualizing the mesh region.

This module renders the mesh points, edges, boundary vertices, local environment,
and reference point with proper layering and colors.
"""

from manim import *
import numpy as np
from typing import List, Tuple, Dict, Optional
from ..utils.coordinate_transform import extract_mesh_edges, get_all_mesh_vertices
from .. import config


class MeshRenderer:
    """Renderer for the mesh visualization region."""
    
    def __init__(self, region_bounds: Tuple[float, float, float, float], scale_factor: float, mesh_center: Tuple[float, float]):
        """
        Initialize the mesh renderer.
        
        Args:
            region_bounds: (left, right, bottom, top) bounds of the mesh region in Manim space
            scale_factor: Scale factor to convert from mesh coordinates to Manim coordinates
            mesh_center: (center_x, center_y) center of the mesh in original coordinates
        """
        self.region_bounds = region_bounds
        self.scale_factor = scale_factor
        self.left, self.right, self.bottom, self.top = region_bounds
        self.mesh_center_x, self.mesh_center_y = mesh_center
        
        # Calculate region center for positioning
        self.region_center_x = (self.left + self.right) / 2
        self.region_center_y = (self.bottom + self.top) / 2
        
    def transform_coord(self, coord: List[float]) -> np.ndarray:
        """
        Transform mesh coordinate to Manim coordinate.
        
        Args:
            coord: [x, y] mesh coordinate
            
        Returns:
            numpy array with Manim coordinate
        """
        x, y = coord
        # First translate to center mesh at origin
        centered_x = (x - self.mesh_center_x) * self.scale_factor
        centered_y = (y - self.mesh_center_y) * self.scale_factor
        # Then translate to region center
        manim_x = centered_x + self.region_center_x
        manim_y = centered_y + self.region_center_y
        return np.array([manim_x, manim_y, 0])
    
    def create_mesh_mobjects(
        self,
        mesh_points: Dict,
        boundary: List[List[float]],
        local_env: Dict,
        ref_coords: List[float]
    ) -> VGroup:
        """
        Create all mesh visualization mobjects for a state.
        
        Args:
            mesh_points: Dictionary of mesh adjacency data
            boundary: List of boundary vertex coordinates
            local_env: Local environment data (neighbors, fan_points, etc.)
            ref_coords: Reference point coordinates
            
        Returns:
            VGroup containing all mesh mobjects
        """
        group = VGroup()
        
        # Extract mesh data
        mesh_edges = extract_mesh_edges(mesh_points)
        mesh_vertices = get_all_mesh_vertices(mesh_points)
        
        # 1. Render mesh edges (lowest layer)
        mesh_edge_group = self._create_mesh_edges(mesh_edges)
        mesh_edge_group.set_z_index(config.Z_INDEX_MESH_EDGES)
        group.add(mesh_edge_group)
        
        # 2. Render mesh points
        mesh_points_group = self._create_mesh_points(mesh_vertices)
        mesh_points_group.set_z_index(config.Z_INDEX_MESH_POINTS)
        group.add(mesh_points_group)
        
        # 3. Render boundary (higher layer)
        boundary_group = self._create_boundary(boundary)
        boundary_group.set_z_index(config.Z_INDEX_BOUNDARY)
        group.add(boundary_group)
        
        # 4. Render local environment (even higher)
        if local_env:
            local_env_group = self._create_local_env(local_env)
            local_env_group.set_z_index(config.Z_INDEX_LOCAL_ENV_POINTS)
            group.add(local_env_group)
        
        # 5. Render reference point (top layer)
        ref_point_group = self._create_ref_point(ref_coords)
        ref_point_group.set_z_index(config.Z_INDEX_REF_POINT)
        group.add(ref_point_group)
        
        return group
    
    def _create_mesh_edges(self, edges: List[Tuple[List[float], List[float]]]) -> VGroup:
        """Create mesh edge lines."""
        edge_group = VGroup()
        
        for edge in edges:
            start_coord = self.transform_coord(edge[0])
            end_coord = self.transform_coord(edge[1])
            
            line = Line(
                start_coord,
                end_coord,
                stroke_color=config.MESH_EDGE_COLOR,
                stroke_width=config.MESH_EDGE_STROKE_WIDTH
            )
            edge_group.add(line)
        
        return edge_group
    
    def _create_mesh_points(self, vertices: List[List[float]]) -> VGroup:
        """Create mesh vertex dots."""
        points_group = VGroup()
        
        for vertex in vertices:
            pos = self.transform_coord(vertex)
            dot = Dot(
                point=pos,
                radius=config.MESH_POINT_RADIUS,
                color=config.MESH_POINT_COLOR
            )
            points_group.add(dot)
        
        return points_group
    
    def _create_boundary(self, boundary: List[List[float]]) -> VGroup:
        """Create boundary vertices and edges."""
        boundary_group = VGroup()
        
        if not boundary:
            return boundary_group
        
        # Create boundary edges (connecting consecutive vertices)
        for i in range(len(boundary)):
            start = boundary[i]
            end = boundary[(i + 1) % len(boundary)]
            
            start_pos = self.transform_coord(start)
            end_pos = self.transform_coord(end)
            
            line = Line(
                start_pos,
                end_pos,
                stroke_color=config.BOUNDARY_COLOR,
                stroke_width=config.BOUNDARY_EDGE_STROKE_WIDTH
            )
            boundary_group.add(line)
        
        # Create boundary points
        for vertex in boundary:
            pos = self.transform_coord(vertex)
            dot = Dot(
                point=pos,
                radius=config.BOUNDARY_POINT_RADIUS,
                color=config.BOUNDARY_COLOR
            )
            boundary_group.add(dot)
        
        return boundary_group
    
    def _create_local_env(self, local_env: Dict) -> VGroup:
        """Create local environment visualization (neighbors only, no fan_points)."""
        local_group = VGroup()
        
        neighbors = local_env.get('neighbors', [])
        ref_idx = local_env.get('reference_vertex_idx', 2)
        
        if not neighbors:
            return local_group
        
        # Draw edges between consecutive neighbors
        for i in range(len(neighbors) - 1):
            start = neighbors[i]
            end = neighbors[i + 1]
            
            start_pos = self.transform_coord(start)
            end_pos = self.transform_coord(end)
            
            line = Line(
                start_pos,
                end_pos,
                stroke_color=config.LOCAL_ENV_EDGE_COLOR,
                stroke_width=config.LOCAL_ENV_EDGE_STROKE_WIDTH
            )
            local_group.add(line)
        
        # Draw neighbor points
        for i, neighbor in enumerate(neighbors):
            pos = self.transform_coord(neighbor)
            dot = Dot(
                point=pos,
                radius=config.LOCAL_ENV_POINT_RADIUS,
                color=config.LOCAL_ENV_POINT_COLOR
            )
            local_group.add(dot)
        
        return local_group
    
    def _create_ref_point(self, ref_coords: List[float]) -> VGroup:
        """Create reference point marker."""
        ref_group = VGroup()
        
        pos = self.transform_coord(ref_coords)
        dot = Dot(
            point=pos,
            radius=config.REF_POINT_RADIUS,
            color=config.REF_POINT_COLOR
        )
        ref_group.add(dot)
        
        return ref_group
