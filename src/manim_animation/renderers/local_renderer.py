"""
Local environment renderer for normalized polar visualization.

This module renders the local environment in a normalized coordinate system
centered at the reference point with x-axis aligned to the previous neighbor.
"""

from manim import *
import numpy as np
from typing import List, Dict, Optional, Tuple
from ..utils.coordinate_transform import normalize_local_env
from .. import config


class LocalRenderer:
    """Renderer for the local environment region (normalized coordinates)."""
    
    def __init__(self, region_bounds: Tuple[float, float, float, float]):
        """
        Initialize the local environment renderer.
        
        Args:
            region_bounds: (left, right, bottom, top) bounds of the local region in Manim space
        """
        self.region_bounds = region_bounds
        self.left, self.right, self.bottom, self.top = region_bounds
        
        # Calculate region center
        self.center_x = (self.left + self.right) / 2
        self.center_y = (self.bottom + self.top) / 2
        
        # Calculate available space
        self.width = self.right - self.left
        self.height = self.top - self.bottom
        
        # Scale factor (calculated per state)
        self.scale_factor = 1.0
        
    def create_local_mobjects(self, local_env: Dict) -> VGroup:
        """
        Create all local environment visualization mobjects.
        
        Args:
            local_env: Local environment data containing neighbors, fan_points, etc.
            
        Returns:
            VGroup containing all local environment mobjects
        """
        group = VGroup()
        
        if not local_env:
            return group
        
        neighbors = local_env.get('neighbors', [])
        fan_points = local_env.get('fan_points', [])
        ref_idx = 2  # Reference point is typically at index 2
        
        if not neighbors:
            return group
        
        # Normalize coordinates
        normalized_neighbors, normalized_fan_points = normalize_local_env(
            neighbors, fan_points, ref_idx
        )
        
        # Calculate scale factor based on actual point distribution
        self._calculate_scale_factor(normalized_neighbors, normalized_fan_points)
        
        # 1. Create coordinate axes (background)
        axes_group = self._create_axes()
        group.add(axes_group)
        
        # 2. Create neighbor edges
        neighbor_edges = self._create_neighbor_edges(normalized_neighbors)
        group.add(neighbor_edges)
        
        # 3. Create neighbor points
        neighbor_points = self._create_neighbor_points(normalized_neighbors, ref_idx)
        group.add(neighbor_points)
        
        # 4. Create fan points
        fan_point_group = self._create_fan_points(normalized_fan_points)
        group.add(fan_point_group)
        
        # 5. Create reference point (on top)
        ref_point = self._create_ref_point(normalized_neighbors[ref_idx])
        group.add(ref_point)
        
        return group
    
    def _calculate_scale_factor(self, normalized_neighbors: np.ndarray, normalized_fan_points: List[Optional[np.ndarray]]):
        """
        Calculate scale factor to fit all points in the region.
        Uses the maximum distance from origin to ensure all points fit.
        
        Args:
            normalized_neighbors: Array of normalized neighbor coordinates
            normalized_fan_points: List of normalized fan point coordinates
        """
        # Collect all points to find maximum distance
        max_distance = 0.0
        
        # Check all neighbors
        for point in normalized_neighbors:
            distance = np.linalg.norm(point)
            max_distance = max(max_distance, distance)
        
        # Check non-null fan points
        for fan_point in normalized_fan_points:
            if fan_point is not None:
                distance = np.linalg.norm(fan_point)
                max_distance = max(max_distance, distance)
        
        if max_distance < 1e-6:
            self.scale_factor = 1.0
            return
        
        # Calculate scale to fit in region with generous padding
        # Use smaller dimension (width or height) to ensure it fits in both
        padding_ratio = 0.25  # 25% padding on all sides
        available_radius = min(self.width, self.height) * (1 - 2 * padding_ratio) / 2
        
        # Scale factor to fit the furthest point within available radius
        self.scale_factor = available_radius / max_distance
    
    def _transform_to_manim(self, coord: np.ndarray) -> np.ndarray:
        """
        Transform normalized coordinate to Manim coordinate in local region.
        
        Args:
            coord: [x, y] normalized coordinate
            
        Returns:
            numpy array with Manim coordinate [x, y, z]
        """
        # Use calculated scale factor
        manim_x = coord[0] * self.scale_factor + self.center_x
        manim_y = coord[1] * self.scale_factor + self.center_y
        
        return np.array([manim_x, manim_y, 0])
    
    def _create_axes(self) -> VGroup:
        """Create coordinate axes (x and y axes)."""
        axes_group = VGroup()
        
        # Calculate axis length based on region size
        # Make axes span most of the region width/height
        axis_extent = min(self.width, self.height) * 0.4 / self.scale_factor
        
        # X-axis (horizontal)
        x_start = self._transform_to_manim(np.array([-axis_extent, 0]))
        x_end = self._transform_to_manim(np.array([axis_extent, 0]))
        x_axis = Line(
            x_start,
            x_end,
            stroke_color=config.LOCAL_AXIS_COLOR,
            stroke_width=config.AXIS_STROKE_WIDTH
        )
        axes_group.add(x_axis)
        
        # Y-axis (vertical)
        y_start = self._transform_to_manim(np.array([0, -axis_extent]))
        y_end = self._transform_to_manim(np.array([0, axis_extent]))
        y_axis = Line(
            y_start,
            y_end,
            stroke_color=config.LOCAL_AXIS_COLOR,
            stroke_width=config.AXIS_STROKE_WIDTH
        )
        axes_group.add(y_axis)
        
        # Add axis labels
        x_label = Text("X", font_size=16, color=config.LOCAL_AXIS_COLOR)
        x_label.move_to(x_end + RIGHT * 0.2)
        axes_group.add(x_label)
        
        y_label = Text("Y", font_size=16, color=config.LOCAL_AXIS_COLOR)
        y_label.move_to(y_end + UP * 0.2)
        axes_group.add(y_label)
        
        return axes_group
    
    def _create_neighbor_edges(self, normalized_neighbors: np.ndarray) -> VGroup:
        """Create edges connecting consecutive neighbors."""
        edge_group = VGroup()
        
        # Connect each neighbor to the next (except the last one)
        for i in range(len(normalized_neighbors) - 1):
            start_pos = self._transform_to_manim(normalized_neighbors[i])
            end_pos = self._transform_to_manim(normalized_neighbors[i + 1])
            
            line = Line(
                start_pos,
                end_pos,
                stroke_color=config.LOCAL_NEIGHBOR_COLOR,
                stroke_width=config.LOCAL_REGION_EDGE_STROKE_WIDTH
            )
            edge_group.add(line)
        
        return edge_group
    
    def _create_neighbor_points(
        self,
        normalized_neighbors: np.ndarray,
        ref_idx: int
    ) -> VGroup:
        """Create neighbor point dots (excluding reference point)."""
        points_group = VGroup()
        
        for i, neighbor in enumerate(normalized_neighbors):
            # Skip reference point (will be drawn separately)
            if i == ref_idx:
                continue
            
            pos = self._transform_to_manim(neighbor)
            dot = Dot(
                point=pos,
                radius=config.LOCAL_REGION_POINT_RADIUS,
                color=config.LOCAL_NEIGHBOR_COLOR
            )
            points_group.add(dot)
        
        return points_group
    
    def _create_fan_points(self, normalized_fan_points: List[Optional[np.ndarray]]) -> VGroup:
        """Create fan point dots (skip None values)."""
        fan_group = VGroup()
        
        for fan_point in normalized_fan_points:
            if fan_point is None:
                continue
            
            pos = self._transform_to_manim(fan_point)
            dot = Dot(
                point=pos,
                radius=config.FAN_POINT_RADIUS,
                color=config.LOCAL_FAN_COLOR
            )
            fan_group.add(dot)
        
        return fan_group
    
    def _create_ref_point(self, normalized_ref: np.ndarray) -> VGroup:
        """Create reference point marker at origin."""
        ref_group = VGroup()
        
        pos = self._transform_to_manim(normalized_ref)
        dot = Dot(
            point=pos,
            radius=config.REF_POINT_RADIUS,
            color=config.LOCAL_REF_COLOR
        )
        ref_group.add(dot)
        
        return ref_group
