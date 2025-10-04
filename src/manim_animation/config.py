"""
Configuration file for mesh generation animation.

This module contains all configuration parameters for the animation including
resolution, frame rate, colors, timings, and state-specific overrides.
"""

from manim import *

# ============================================================================
# Video Quality Settings
# ============================================================================
RESOLUTION = "1080p"
FRAME_RATE = 60

# ============================================================================
# Layout Configuration
# ============================================================================
# Screen is divided into 3 regions:
# - Mesh region: left 60%
# - Local region: right upper 20% (40% width × 50% height)
# - Boundary region: right lower 20% (40% width × 50% height)

MESH_REGION_WIDTH_RATIO = 0.60      # 60% of screen width
LOCAL_REGION_WIDTH_RATIO = 0.40     # 40% of screen width
LOCAL_REGION_HEIGHT_RATIO = 0.50    # 50% of screen height (upper half)
BOUNDARY_REGION_HEIGHT_RATIO = 0.50 # 50% of screen height (lower half)

# ============================================================================
# Color Scheme
# ============================================================================
BACKGROUND_COLOR = BLACK

# Mesh region colors
MESH_POINT_COLOR = "#87CEEB"        # Light blue (SkyBlue)
MESH_EDGE_COLOR = "#87CEEB"         # Light blue
BOUNDARY_COLOR = RED                # Red for boundary
LOCAL_ENV_POINT_COLOR = YELLOW      # Yellow for local environment points
LOCAL_ENV_EDGE_COLOR = YELLOW       # Yellow for local environment edges
REF_POINT_COLOR = GREEN             # Green for reference point

# Local region colors
LOCAL_AXIS_COLOR = GRAY             # Gray for coordinate axes
LOCAL_NEIGHBOR_COLOR = YELLOW       # Yellow for neighbors
LOCAL_REF_COLOR = GREEN             # Green for reference point
LOCAL_FAN_COLOR = RED               # Red for fan points

# Boundary region colors
BOUNDARY_REGION_COLOR = RED         # Red for boundary visualization

# ============================================================================
# Size and Style Settings
# ============================================================================
# Point/Dot sizes
MESH_POINT_RADIUS = 0.03
BOUNDARY_POINT_RADIUS = 0.04
LOCAL_ENV_POINT_RADIUS = 0.04
REF_POINT_RADIUS = 0.05
LOCAL_REGION_POINT_RADIUS = 0.06
FAN_POINT_RADIUS = 0.06

# Stroke widths
MESH_EDGE_STROKE_WIDTH = 1.5
BOUNDARY_EDGE_STROKE_WIDTH = 2.5
LOCAL_ENV_EDGE_STROKE_WIDTH = 2.0
LOCAL_REGION_EDGE_STROKE_WIDTH = 2.5
AXIS_STROKE_WIDTH = 1.0

# ============================================================================
# Timing Configuration
# ============================================================================
DEFAULT_STATE_DURATION = 1.0        # Default time (seconds) to show each state
DEFAULT_TRANSITION_DURATION = 0.3   # Default transition time between states

# State-specific timing overrides
# Format: {state_id: duration} or {(start, end): duration} for ranges
STATE_DURATION_OVERRIDES = {
    # Example: Fast-forward through boring middle states
    # (40, 80): 0.5,  # States 40-80 only show for 0.5 seconds each
    
    # Example: Emphasize important states
    # 0: 2.0,         # Initial state shows for 2 seconds
    # 114: 3.0,       # Final state shows for 3 seconds
}

# ============================================================================
# Coordinate Scaling
# ============================================================================
# Scale factor to fit mesh coordinates into Manim space
# Will be auto-calculated based on mesh bounds, but can be overridden
AUTO_SCALE = True
MANUAL_SCALE_FACTOR = 0.01

# Padding around mesh (as fraction of mesh size)
MESH_PADDING = 0.1

# Local region display settings
LOCAL_REGION_SCALE = 2.0            # Scale factor for local environment display
LOCAL_AXIS_LENGTH = 1.5             # Length of coordinate axes in local region

# ============================================================================
# Data Configuration
# ============================================================================
DEFAULT_JSON_PATH = "data/animation_data/basic1_sequence.json"

# ============================================================================
# Animation Options
# ============================================================================
SHOW_STATE_LABELS = True            # Show state_id/step as text
SHOW_REGION_BORDERS = True          # Show borders between regions
FADE_TRANSITIONS = True             # Use fade transitions between states

# Z-Index layering (higher = on top)
Z_INDEX_MESH_EDGES = 1
Z_INDEX_MESH_POINTS = 2
Z_INDEX_BOUNDARY = 3
Z_INDEX_LOCAL_ENV_EDGES = 4
Z_INDEX_LOCAL_ENV_POINTS = 5
Z_INDEX_REF_POINT = 6


def get_state_duration(state_id):
    """
    Get the duration for a specific state.
    
    Args:
        state_id (int): The state identifier
        
    Returns:
        float: Duration in seconds for this state
    """
    # Check for exact state_id match
    if state_id in STATE_DURATION_OVERRIDES:
        return STATE_DURATION_OVERRIDES[state_id]
    
    # Check for range matches
    for key, duration in STATE_DURATION_OVERRIDES.items():
        if isinstance(key, tuple) and len(key) == 2:
            start, end = key
            if start <= state_id <= end:
                return duration
    
    # Return default
    return DEFAULT_STATE_DURATION
