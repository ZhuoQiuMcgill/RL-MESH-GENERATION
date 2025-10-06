"""
Debug script for Action Introduction Scene.

This script displays only the ActionIntroScene for debugging purposes.
Run from project root directory.

Usage:
    manim src/manim_animation/scene_test/debug_action_intro.py ActionIntroDebug -p --resolution 1080p
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from manim import *
from src.manim_animation.action_intro_scene import ActionIntroScene
from src.manim_animation import config as anim_config


class ActionIntroDebug(ActionIntroScene):
    """Debug wrapper for ActionIntroScene with preview settings."""
    
    def __init__(self, **kwargs):
        """Initialize with default actions.json path."""
        super().__init__(
            actions_json_path="data/animation_data/actions.json",
            **kwargs
        )
    
    def construct(self):
        """Run the action intro scene."""
        # Set background color
        self.camera.background_color = anim_config.BACKGROUND_COLOR
        
        # Call parent construct
        super().construct()


# Run this script directly with Python:
# python src/manim_animation/scene_test/debug_action_intro.py

if __name__ == "__main__":
    from manim import config as manim_config
    
    # Configure for preview mode
    manim_config.preview = True
    manim_config.write_to_movie = False
    manim_config.pixel_height = 720
    manim_config.pixel_width = 1280
    manim_config.frame_rate = 30
    
    # Render the scene
    scene = ActionIntroDebug()
    scene.render()
