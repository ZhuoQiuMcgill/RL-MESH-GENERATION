"""
Boundary renderer for visualizing the boundary region.

This module renders the current boundary vertices in the dedicated boundary region.
"""

from manim import *
import numpy as np
from typing import List, Tuple
from ..utils.coordinate_transform import calculate_bounds
from .. import config


class BoundaryRenderer:
    """Renderer for the boundary visualization region."""
    
    def __init__(self, region_bounds: Tuple[float, float, float, float]):
        """
        Initialize the boundary renderer.
        
        Args:
            region_bounds: (left, right, bottom, top) bounds of the boundary region in Manim space
        """
        self.region_bounds = region_bounds
        self.left, self.right, self.bottom, self.top = region_bounds
        
        # Calculate region center
        self.center_x = (self.left + self.right) / 2
        self.center_y = (self.bottom + self.top) / 2
        
        # Calculate available space
        self.width = self.right - self.left
        self.height = self.top - self.bottom
        
    def create_boundary_mobjects(self, boundary: List[List[float]]) -> VGroup:
        """
        Create boundary visualization mobjects.
        
        Args:
            boundary: List of boundary vertex coordinates
            
        Returns:
            VGroup containing boundary visualization
        """
        group = VGroup()
        
        if not boundary or len(boundary) < 2:
            return group
        
        # Calculate scale to fit boundary in region
        scale_factor = self._calculate_scale_factor(boundary)
        boundary_center = self._calculate_boundary_center(boundary)
        
        # Create boundary edges (connecting consecutive vertices in a loop)
        for i in range(len(boundary)):
            start = boundary[i]
            end = boundary[(i + 1) % len(boundary)]
            
            start_pos = self._transform_to_manim(start, scale_factor, boundary_center)
            end_pos = self._transform_to_manim(end, scale_factor, boundary_center)
            
            line = Line(
                start_pos,
                end_pos,
                stroke_color=config.BOUNDARY_REGION_COLOR,
                stroke_width=config.BOUNDARY_EDGE_STROKE_WIDTH
            )
            group.add(line)
        
        # Create boundary points
        for vertex in boundary:
            pos = self._transform_to_manim(vertex, scale_factor, boundary_center)
            dot = Dot(
                point=pos,
                radius=config.BOUNDARY_POINT_RADIUS,
                color=config.BOUNDARY_REGION_COLOR
            )
            group.add(dot)
        
        return group
    
    def _calculate_scale_factor(self, boundary: List[List[float]]) -> float:
        """Calculate appropriate scale factor to fit boundary in region."""
        if not boundary:
            return 1.0
        
        min_x, max_x, min_y, max_y = calculate_bounds(boundary)
        
        # Calculate boundary dimensions
        boundary_width = max_x - min_x
        boundary_height = max_y - min_y
        
        # Avoid division by zero
        if boundary_width < 1e-10:
            boundary_width = 1.0
        if boundary_height < 1e-10:
            boundary_height = 1.0
        
        # Calculate scale to fit with padding
        padding_ratio = 0.15  # Leave 15% padding
        available_width = self.width * (1 - 2 * padding_ratio)
        available_height = self.height * (1 - 2 * padding_ratio)
        
        scale_x = available_width / boundary_width
        scale_y = available_height / boundary_height
        
        return min(scale_x, scale_y)
    
    def _calculate_boundary_center(self, boundary: List[List[float]]) -> Tuple[float, float]:
        """Calculate the center of the boundary."""
        if not boundary:
            return (0.0, 0.0)
        
        min_x, max_x, min_y, max_y = calculate_bounds(boundary)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        return (center_x, center_y)
    
    def _transform_to_manim(
        self,
        coord: List[float],
        scale_factor: float,
        boundary_center: Tuple[float, float]
    ) -> np.ndarray:
        """
        Transform boundary coordinate to Manim coordinate in boundary region.
        
        Args:
            coord: [x, y] boundary coordinate
            scale_factor: Scale factor for fitting
            boundary_center: Center of the boundary for centering
            
        Returns:
            numpy array with Manim coordinate [x, y, z]
        """
        # Translate to origin, scale, then translate to region center
        x = (coord[0] - boundary_center[0]) * scale_factor + self.center_x
        y = (coord[1] - boundary_center[1]) * scale_factor + self.center_y
        
        return np.array([x, y, 0])
