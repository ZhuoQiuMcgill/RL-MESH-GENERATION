"""
Action Introduction Scene for mesh generation animation.

This module displays the three available action types (Type 0, Type 1, Type 2)
with their visual representations and a legend explaining the color scheme.
"""

from manim import *
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from . import config


class ActionIntroScene(Scene):
    """Scene for introducing the three action types."""
    
    def __init__(self, actions_json_path: str = None, **kwargs):
        """
        Initialize the action introduction scene.
        
        Args:
            actions_json_path: Path to actions.json file
            **kwargs: Additional arguments passed to Scene
        """
        super().__init__(**kwargs)
        self.actions_json_path = actions_json_path or "data/animation_data/actions.json"
        self.actions_data = None
        
    def construct(self):
        """Main construct method for the scene."""
        # Set background color
        self.camera.background_color = config.BACKGROUND_COLOR
        
        # Load actions data
        self._load_actions_data()
        
        # Create scene elements
        title = self._create_title()
        action_panels = self._create_action_panels()
        legend = self._create_legend()
        
        # Group all elements
        all_elements = VGroup(title, action_panels, legend)
        
        # Animate
        self.play(FadeIn(all_elements, run_time=0.5))
        self.wait(3.0)
        self.play(FadeOut(all_elements, run_time=0.5))
    
    def _load_actions_data(self):
        """Load actions data from JSON file."""
        # Handle relative paths from project root
        json_path = Path(self.actions_json_path)
        if not json_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            json_path = project_root / self.actions_json_path
        
        with open(json_path, 'r') as f:
            self.actions_data = json.load(f)
        
        print(f"Loaded {len(self.actions_data)} action types from {json_path}")
    
    def _create_title(self) -> Text:
        """Create the scene title."""
        title = Text(
            "Actions Available",
            font="Serif",
            font_size=48,
            weight=BOLD,
            color="#58C4DD"
        )
        title.to_edge(UP, buff=0.3)
        return title
    
    def _create_action_panels(self) -> VGroup:
        """Create the three action panels side by side."""
        panels = VGroup()
        
        # Calculate panel dimensions and positions
        frame_width = config.config.frame_width
        frame_height = config.config.frame_height
        
        # Panel dimensions
        panel_width = frame_width * 0.28  # 28% of screen width each
        panel_height = frame_height * 0.45  # 45% of screen height
        spacing = frame_width * 0.02  # 2% spacing between panels
        
        # Y position for panels (slightly above center)
        panel_y = frame_height * 0.08
        
        # X positions for three panels (left, center, right)
        total_width = 3 * panel_width + 2 * spacing
        start_x = -total_width / 2 + panel_width / 2
        
        x_positions = [
            start_x,
            start_x + panel_width + spacing,
            start_x + 2 * (panel_width + spacing)
        ]
        
        # Create each action panel
        for i, action_data in enumerate(self.actions_data):
            panel = self._create_single_action_panel(
                action_data,
                panel_width,
                panel_height,
                x_positions[i],
                panel_y
            )
            panels.add(panel)
        
        return panels
    
    def _create_single_action_panel(
        self,
        action_data: Dict,
        width: float,
        height: float,
        x_pos: float,
        y_pos: float
    ) -> VGroup:
        """
        Create a single action visualization panel.
        
        Args:
            action_data: Action data from JSON
            width: Panel width
            height: Panel height
            x_pos: X position of panel center
            y_pos: Y position of panel center
            
        Returns:
            VGroup containing the action visualization and name label
        """
        panel_group = VGroup()
        
        # Extract action data
        action_name = action_data.get('name', 'Unknown')
        ref_point = action_data.get('reference point', [0, 0])
        neighbors = action_data.get('neighbors', [])
        action_attempt = action_data.get('action attempt', {})
        new_vertices = action_attempt.get('vertices', [])
        new_edges = action_attempt.get('edges', [])
        
        # Calculate scale factor to fit in panel
        scale_factor = self._calculate_scale_factor(
            neighbors, new_vertices, width * 0.7, height * 0.7
        )
        
        # Create visualization area
        viz_center_y = y_pos + height * 0.05
        
        # 1. Draw existing neighbor edges (yellow)
        for i in range(len(neighbors) - 1):
            start = self._transform_coord(neighbors[i], scale_factor, x_pos, viz_center_y)
            end = self._transform_coord(neighbors[i + 1], scale_factor, x_pos, viz_center_y)
            
            line = Line(
                start, end,
                stroke_color=config.LOCAL_ENV_EDGE_COLOR,
                stroke_width=2.5
            )
            panel_group.add(line)
        
        # 2. Draw new edges (blue) - from action attempt
        for edge_group in new_edges:
            for edge in edge_group:
                start = self._transform_coord(edge[0], scale_factor, x_pos, viz_center_y)
                end = self._transform_coord(edge[1], scale_factor, x_pos, viz_center_y)
                
                line = Line(
                    start, end,
                    stroke_color=config.MESH_EDGE_COLOR,
                    stroke_width=2.5
                )
                panel_group.add(line)
        
        # 3. Draw neighbor points (yellow)
        for i, neighbor in enumerate(neighbors):
            # Skip reference point (will be drawn separately)
            if neighbor == ref_point:
                continue
            
            pos = self._transform_coord(neighbor, scale_factor, x_pos, viz_center_y)
            dot = Dot(
                point=pos,
                radius=0.06,
                color=config.LOCAL_ENV_POINT_COLOR
            )
            panel_group.add(dot)
        
        # 4. Draw new vertices (blue) - from action attempt
        for new_vertex in new_vertices:
            pos = self._transform_coord(new_vertex, scale_factor, x_pos, viz_center_y)
            dot = Dot(
                point=pos,
                radius=0.06,
                color=config.MESH_POINT_COLOR
            )
            panel_group.add(dot)
        
        # 5. Draw reference point (green) - on top
        ref_pos = self._transform_coord(ref_point, scale_factor, x_pos, viz_center_y)
        ref_dot = Dot(
            point=ref_pos,
            radius=0.08,
            color=config.REF_POINT_COLOR
        )
        panel_group.add(ref_dot)
        
        # 6. Create action name label below the panel
        label = Text(
            action_name,
            font="Serif",
            font_size=32,
            weight=BOLD,
            color=WHITE
        )
        # Position name closer to visualization
        label.move_to([x_pos, y_pos - height * 0.56, 0])
        panel_group.add(label)
        
        return panel_group
    
    def _calculate_scale_factor(
        self,
        neighbors: List[List[float]],
        new_vertices: List[List[float]],
        target_width: float,
        target_height: float
    ) -> float:
        """
        Calculate scale factor to fit all points in the target area.
        
        Args:
            neighbors: List of neighbor coordinates
            new_vertices: List of new vertex coordinates
            target_width: Target width for fitting
            target_height: Target height for fitting
            
        Returns:
            Scale factor
        """
        # Combine all points
        all_points = neighbors + new_vertices
        
        if not all_points:
            return 1.0
        
        # Calculate bounds
        points_array = np.array(all_points)
        min_x, min_y = points_array.min(axis=0)
        max_x, max_y = points_array.max(axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Avoid division by zero
        if width < 1e-10:
            width = 1.0
        if height < 1e-10:
            height = 1.0
        
        # Calculate scale with padding
        padding_ratio = 0.15
        available_width = target_width * (1 - 2 * padding_ratio)
        available_height = target_height * (1 - 2 * padding_ratio)
        
        scale_x = available_width / width
        scale_y = available_height / height
        
        return min(scale_x, scale_y)
    
    def _transform_coord(
        self,
        coord: List[float],
        scale: float,
        center_x: float,
        center_y: float
    ) -> np.ndarray:
        """
        Transform coordinate to Manim space.
        
        Args:
            coord: [x, y] coordinate
            scale: Scale factor
            center_x: X center position
            center_y: Y center position
            
        Returns:
            Transformed coordinate as numpy array
        """
        x = coord[0] * scale + center_x
        y = coord[1] * scale + center_y
        return np.array([x, y, 0])
    
    def _create_legend(self) -> VGroup:
        """Create the legend explaining color meanings."""
        legend_group = VGroup()
        
        # Legend position (bottom center)
        legend_y = -3.0
        
        # Item 1: Reference Point (Green) - text color matches dot color
        item1 = VGroup()
        ref_dot = Dot(point=[0, 0, 0], radius=0.06, color=config.REF_POINT_COLOR)
        ref_label = Text("Reference Point", font_size=22, color=config.REF_POINT_COLOR)
        ref_label.next_to(ref_dot, RIGHT, buff=0.15)
        item1.add(ref_dot, ref_label)
        
        # Item 2: Neighbors (Yellow) - text color matches dot color
        item2 = VGroup()
        neighbor_dot = Dot(point=[0, 0, 0], radius=0.06, color=config.LOCAL_ENV_POINT_COLOR)
        neighbor_label = Text("Neighbors", font_size=22, color=config.LOCAL_ENV_POINT_COLOR)
        neighbor_label.next_to(neighbor_dot, RIGHT, buff=0.15)
        item2.add(neighbor_dot, neighbor_label)
        
        # Item 3: New Elements (Blue) - text color matches dot color
        item3 = VGroup()
        new_dot = Dot(point=[0, 0, 0], radius=0.06, color=config.MESH_POINT_COLOR)
        new_label = Text("New Elements", font_size=22, color=config.MESH_POINT_COLOR)
        new_label.next_to(new_dot, RIGHT, buff=0.15)
        item3.add(new_dot, new_label)
        
        # Arrange items horizontally with proper spacing
        item1.move_to([0, legend_y, 0])
        item2.next_to(item1, RIGHT, buff=0.8)
        item3.next_to(item2, RIGHT, buff=0.8)
        
        # Center the entire group
        items_group = VGroup(item1, item2, item3)
        items_group.move_to([0, legend_y, 0])
        
        legend_group.add(items_group)
        
        return legend_group
