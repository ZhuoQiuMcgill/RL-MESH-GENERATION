"""
Mesh renderer for visualizing the mesh region.

This module renders the mesh points, edges, boundary vertices, local environment,
and reference point with proper layering and colors.
"""

from manim import *
import numpy as np
import math
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
        
        # 3. Render sector area (if local_env exists)
        if local_env:
            sector_area = self._create_sector_area(local_env)
            if sector_area:
                sector_area.set_z_index(config.Z_INDEX_SECTOR_AREA)
                group.add(sector_area)
        
        # 4. Render boundary (higher layer)
        boundary_group = self._create_boundary(boundary)
        boundary_group.set_z_index(config.Z_INDEX_BOUNDARY)
        group.add(boundary_group)
        
        # 5. Render local environment (even higher)
        if local_env:
            local_env_group = self._create_local_env(local_env)
            local_env_group.set_z_index(config.Z_INDEX_LOCAL_ENV_POINTS)
            group.add(local_env_group)
        
        # 6. Render fan points in mesh region (if they exist)
        if local_env:
            fan_points_group = self._create_fan_points_mesh(local_env)
            if fan_points_group:
                fan_points_group.set_z_index(config.Z_INDEX_FAN_POINTS_MESH)
                group.add(fan_points_group)
        
        # 7. Render reference point (top layer)
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
    
    def _create_sector_area(self, local_env: Dict) -> Optional[VGroup]:
        """Create sector area visualization.
        
        The sector area is defined by:
        - Vertex: v0 (reference point)
        - Radius: r = l × beta, where l is the average length of 4 edges
        - Right boundary: direction from v0 to vr1
        - Left boundary: direction from v0 to vl1
        """
        neighbors = local_env.get('neighbors', [])
        beta = local_env.get('beta', 6)
        g = local_env.get('g', 3)  # Number of fan sectors
        
        if len(neighbors) < 5:  # Need at least 5 neighbors (vr2, vr1, v0, vl1, vl2)
            return None
        
        # Extract vertices
        vr2 = np.array(neighbors[0])
        vr1 = np.array(neighbors[1])
        v0 = np.array(neighbors[2])   # reference point
        vl1 = np.array(neighbors[3])
        vl2 = np.array(neighbors[4])
        
        # Calculate average length l
        edge1_len = np.linalg.norm(vr1 - vr2)
        edge2_len = np.linalg.norm(v0 - vr1)
        edge3_len = np.linalg.norm(vl1 - v0)
        edge4_len = np.linalg.norm(vl2 - vl1)
        l = (edge1_len + edge2_len + edge3_len + edge4_len) / 4
        
        # Calculate radius
        radius = l * beta
        
        # Calculate angles
        vec_to_vr1 = vr1 - v0
        vec_to_vl1 = vl1 - v0
        
        angle_vr1 = math.atan2(vec_to_vr1[1], vec_to_vr1[0])
        angle_vl1 = math.atan2(vec_to_vl1[1], vec_to_vl1[0])
        
        # Calculate the angle span (counter-clockwise from vr1 to vl1)
        # We want the INNER sector, not the outer one
        # Since neighbors are in clockwise order, we need to go counter-clockwise from vr1 to vl1
        angle_diff = angle_vl1 - angle_vr1
        if angle_diff < 0:
            angle_diff += 2 * math.pi
        
        # If the angle is greater than π, we're going the wrong way - take the shorter arc
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2 * math.pi
        
        # Transform v0 to Manim coordinates
        v0_manim = self.transform_coord(v0.tolist())
        
        # Scale radius to Manim space
        radius_manim = radius * self.scale_factor
        
        # Create sector using Sector or Polygon for proper filling
        sector_group = VGroup()
        
        # Calculate points on the arc for polygon approximation
        num_arc_points = 30  # Number of points to approximate the arc
        arc_points = [v0_manim]  # Start from center
        
        # Add points along the arc from angle_vr1 to angle_vl1
        for i in range(num_arc_points + 1):
            t = i / num_arc_points
            angle = angle_vr1 + angle_diff * t
            point = v0_manim + radius_manim * np.array([math.cos(angle), math.sin(angle), 0])
            arc_points.append(point)
        
        # Create filled polygon for sector area
        sector_fill = Polygon(
            *arc_points,
            color=config.SECTOR_AREA_COLOR,
            fill_color=config.SECTOR_AREA_COLOR,
            fill_opacity=config.SECTOR_AREA_OPACITY,
            stroke_width=0  # No outline for the fill
        )
        sector_group.add(sector_fill)
        
        # Create the arc outline
        arc = Arc(
            radius=radius_manim,
            start_angle=angle_vr1,
            angle=angle_diff,
            arc_center=v0_manim,
            stroke_color=config.SECTOR_AREA_COLOR,
            stroke_width=2
        )
        sector_group.add(arc)
        
        # Create the two radial lines
        # Line from v0 to arc start (vr1 direction)
        arc_start = v0_manim + radius_manim * np.array([math.cos(angle_vr1), math.sin(angle_vr1), 0])
        line1 = Line(
            v0_manim,
            arc_start,
            stroke_color=config.SECTOR_AREA_COLOR,
            stroke_width=2
        )
        sector_group.add(line1)
        
        # Line from v0 to arc end (vl1 direction)
        arc_end = v0_manim + radius_manim * np.array([math.cos(angle_vl1), math.sin(angle_vl1), 0])
        line2 = Line(
            v0_manim,
            arc_end,
            stroke_color=config.SECTOR_AREA_COLOR,
            stroke_width=2
        )
        sector_group.add(line2)
        
        # Add subdivision lines to divide sector into g parts
        if g > 1:
            for i in range(1, g):
                # Calculate angle for this subdivision line
                subdivision_angle = angle_vr1 + (angle_diff * i / g)
                
                # Calculate end point on arc
                subdivision_end = v0_manim + radius_manim * np.array([
                    math.cos(subdivision_angle),
                    math.sin(subdivision_angle),
                    0
                ])
                
                # Create subdivision line
                subdivision_line = Line(
                    v0_manim,
                    subdivision_end,
                    stroke_color=config.SECTOR_AREA_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.6
                )
                sector_group.add(subdivision_line)
        
        return sector_group
    
    def _create_fan_points_mesh(self, local_env: Dict) -> Optional[VGroup]:
        """Create fan points visualization in mesh region.
        
        Fan points are vf1, vf2, vf3 from the fan_points list.
        Only non-null fan points are rendered.
        """
        fan_points = local_env.get('fan_points', [])
        
        if not fan_points:
            return None
        
        fan_group = VGroup()
        
        for fan_point in fan_points:
            if fan_point is None:
                continue
            
            # Transform to Manim coordinates
            pos = self.transform_coord(fan_point)
            
            # Create dot for fan point
            dot = Dot(
                point=pos,
                radius=config.FAN_POINT_RADIUS_MESH,
                color=config.FAN_POINT_COLOR
            )
            fan_group.add(dot)
        
        if len(fan_group) == 0:
            return None
        
        return fan_group
