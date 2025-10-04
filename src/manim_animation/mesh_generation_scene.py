"""
Main animation scene for mesh generation visualization.

This module contains the MeshGenerationScene class that orchestrates
the 3-panel layout and handles state transitions.
"""

from manim import *
import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from . import config
from .renderers import MeshRenderer, LocalRenderer, BoundaryRenderer
from .utils.coordinate_transform import calculate_bounds, get_all_mesh_vertices


class MeshGenerationScene(Scene):
    """Main scene for mesh generation animation."""
    
    def __init__(self, json_path: str = None, **kwargs):
        """
        Initialize the mesh generation scene.
        
        Args:
            json_path: Path to JSON sequence data file
            **kwargs: Additional arguments passed to Scene
        """
        super().__init__(**kwargs)
        self.json_path = json_path or config.DEFAULT_JSON_PATH
        self.data = None
        self.states = None
        self.metadata = None
        
        # Region bounds (will be calculated)
        self.mesh_bounds = None
        self.local_bounds = None
        self.boundary_bounds = None
        
        # Renderers
        self.mesh_renderer = None
        self.local_renderer = None
        self.boundary_renderer = None
        
        # Scale factor and center for mesh
        self.mesh_scale_factor = 1.0
        self.mesh_center = (0.0, 0.0)
        
    def construct(self):
        """Main construct method for the scene."""
        # Set background color
        self.camera.background_color = config.BACKGROUND_COLOR
        
        # Load data
        self._load_data()
        
        # Calculate region bounds
        self._calculate_region_bounds()
        
        # Create renderers
        self._create_renderers()
        
        # Animate through all states (borders will be added inside)
        self._animate_states()
    
    def _load_data(self):
        """Load JSON sequence data."""
        # Handle relative paths from project root
        if not Path(self.json_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.json_path = str(project_root / self.json_path)
        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.states = self.data.get('states', [])
        
        print(f"Loaded {len(self.states)} states from {self.json_path}")
        print(f"Mesh: {self.metadata.get('mesh_name', 'unknown')}")
        print(f"Model: {self.metadata.get('model_name', 'unknown')}")
    
    def _calculate_region_bounds(self):
        """Calculate bounds for each region in Manim coordinate space."""
        # Get camera frame dimensions
        frame_width = config.config.frame_width
        frame_height = config.config.frame_height
        
        # Calculate region boundaries
        # Mesh region: left 60%
        mesh_left = -frame_width / 2
        mesh_right = mesh_left + frame_width * config.MESH_REGION_WIDTH_RATIO
        mesh_bottom = -frame_height / 2
        mesh_top = frame_height / 2
        self.mesh_bounds = (mesh_left, mesh_right, mesh_bottom, mesh_top)
        
        # Local region: right upper (40% width × 50% height)
        local_left = mesh_right
        local_right = frame_width / 2
        local_bottom = 0
        local_top = frame_height / 2
        self.local_bounds = (local_left, local_right, local_bottom, local_top)
        
        # Boundary region: right lower (40% width × 50% height)
        boundary_left = mesh_right
        boundary_right = frame_width / 2
        boundary_bottom = -frame_height / 2
        boundary_top = 0
        self.boundary_bounds = (boundary_left, boundary_right, boundary_bottom, boundary_top)
        
        # Calculate mesh scale factor
        self._calculate_mesh_scale()
    
    def _calculate_mesh_scale(self):
        """Calculate scale factor and center to fit mesh in mesh region."""
        if not self.states:
            self.mesh_scale_factor = 1.0
            self.mesh_center = (0.0, 0.0)
            return
        
        # Get all vertices from first state (to determine bounds)
        first_state = self.states[0]
        mesh_points = first_state.get('mesh_points', {})
        all_vertices = get_all_mesh_vertices(mesh_points)
        
        if not all_vertices:
            self.mesh_scale_factor = 1.0
            self.mesh_center = (0.0, 0.0)
            return
        
        # Calculate mesh dimensions and center
        min_x, max_x, min_y, max_y = calculate_bounds(all_vertices)
        mesh_width = max_x - min_x
        mesh_height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.mesh_center = (center_x, center_y)
        
        # Calculate available space in mesh region
        mesh_region_width = self.mesh_bounds[1] - self.mesh_bounds[0]
        mesh_region_height = self.mesh_bounds[3] - self.mesh_bounds[2]
        
        # Apply padding
        padding = config.MESH_PADDING
        available_width = mesh_region_width * (1 - 2 * padding)
        available_height = mesh_region_height * (1 - 2 * padding)
        
        # Calculate scale factor
        if mesh_width > 0 and mesh_height > 0:
            scale_x = available_width / mesh_width
            scale_y = available_height / mesh_height
            self.mesh_scale_factor = min(scale_x, scale_y)
        else:
            self.mesh_scale_factor = 0.01
        
        print(f"Mesh scale factor: {self.mesh_scale_factor:.6f}")
        print(f"Mesh center: {self.mesh_center}")
    
    def _create_renderers(self):
        """Initialize renderer objects."""
        self.mesh_renderer = MeshRenderer(self.mesh_bounds, self.mesh_scale_factor, self.mesh_center)
        self.local_renderer = LocalRenderer(self.local_bounds)
        self.boundary_renderer = BoundaryRenderer(self.boundary_bounds)
    
    def _draw_region_borders(self):
        """Draw borders between regions for visualization (deprecated - use _create_region_borders)."""
        pass
    
    def _create_region_borders(self) -> VGroup:
        """Create border lines between regions."""
        borders = VGroup()
        
        # Vertical line separating mesh and right panels
        v_line = Line(
            start=[self.mesh_bounds[1], self.mesh_bounds[2], 0],
            end=[self.mesh_bounds[1], self.mesh_bounds[3], 0],
            stroke_color=GRAY,
            stroke_width=1
        )
        borders.add(v_line)
        
        # Horizontal line separating local and boundary regions
        h_line = Line(
            start=[self.local_bounds[0], 0, 0],
            end=[self.local_bounds[1], 0, 0],
            stroke_color=GRAY,
            stroke_width=1
        )
        borders.add(h_line)
        
        return borders
    
    def _animate_states(self):
        """Animate through all states.
        
        Note: Manim creates partial movie files for each play() call.
        This is normal behavior and they will be combined into one final video.
        """
        # Create and add region borders once at the start
        if config.SHOW_REGION_BORDERS:
            borders = self._create_region_borders()
            self.add(borders)
        
        if not self.states:
            return
        
        print(f"\nAnimating {len(self.states)} states...")
        print("Note: Partial movie files are normal - they will be combined at the end.\n")
        
        # Add first state without animation
        first_state_mobjects = self._create_state_mobjects(self.states[0])
        self.add(first_state_mobjects)
        
        # Animate through remaining states
        for i in range(1, len(self.states)):
            state = self.states[i]
            state_id = state.get('state_id', i)
            prev_state_id = self.states[i-1].get('state_id', i-1)
            
            # Get timing
            prev_duration = config.get_state_duration(prev_state_id)
            transition_time = config.DEFAULT_TRANSITION_DURATION
            
            # Create mobjects for current state
            current_mobjects = self._create_state_mobjects(state)
            
            # Remove previous state and add current state with transition
            if config.FADE_TRANSITIONS:
                self.play(
                    FadeOut(first_state_mobjects, run_time=transition_time),
                    FadeIn(current_mobjects, run_time=transition_time),
                    run_time=transition_time
                )
            else:
                self.remove(first_state_mobjects)
                self.add(current_mobjects)
                self.wait(transition_time)
            
            # Hold current state
            if prev_duration > 0:
                self.wait(prev_duration)
            
            first_state_mobjects = current_mobjects
            
            # Progress indicator
            if i % 10 == 0 or i == len(self.states) - 1:
                print(f"Rendered state {i + 1}/{len(self.states)}")
        
        # Hold final state
        final_duration = config.get_state_duration(self.states[-1].get('state_id', len(self.states)-1))
        if final_duration > 0:
            self.wait(final_duration)
    
    def _create_state_mobjects(self, state: Dict) -> VGroup:
        """
        Create all mobjects for a single state.
        
        Args:
            state: State dictionary from JSON
            
        Returns:
            VGroup containing all mobjects for the state
        """
        group = VGroup()
        
        # Extract state data
        mesh_points = state.get('mesh_points', {})
        boundary = state.get('boundary', [])
        local_env = state.get('local_env', {})
        ref_coords = local_env.get('reference_coords', [0, 0]) if local_env else [0, 0]
        
        # 1. Mesh region
        mesh_mobjects = self.mesh_renderer.create_mesh_mobjects(
            mesh_points, boundary, local_env, ref_coords
        )
        group.add(mesh_mobjects)
        
        # 2. Local region
        local_mobjects = self.local_renderer.create_local_mobjects(local_env)
        group.add(local_mobjects)
        
        # 3. Boundary region
        boundary_mobjects = self.boundary_renderer.create_boundary_mobjects(boundary)
        group.add(boundary_mobjects)
        
        # 4. Optional state label
        if config.SHOW_STATE_LABELS:
            label = self._create_state_label(state)
            group.add(label)
        
        return group
    
    def _create_state_label(self, state: Dict) -> VGroup:
        """Create label showing state information."""
        state_id = state.get('state_id', 0)
        step = state.get('step', 0)
        
        label_text = f"State: {state_id} | Step: {step}"
        label = Text(label_text, font_size=20, color=WHITE)
        
        # Position at bottom of screen
        label.to_edge(DOWN, buff=0.2)
        
        return VGroup(label)
