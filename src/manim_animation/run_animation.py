"""
Main entry point for running mesh generation animations.

This script provides a command-line interface for generating
animations from JSON sequence data.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from manim import config as manim_config
from src.manim_animation import config
from src.manim_animation.mesh_generation_scene import MeshGenerationScene


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate mesh generation animation from JSON sequence data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default JSON file and settings
  python run_animation.py

  # Specify custom JSON file
  python run_animation.py --json data/animation_data/custom_sequence.json

  # Change output quality
  python run_animation.py --quality low

  # Specify output filename
  python run_animation.py --output my_animation

  # Preview only (no file output)
  python run_animation.py --preview

  # Generate with custom resolution
  python run_animation.py --resolution 720p --fps 30
        '''
    )
    
    # Input/Output options
    parser.add_argument(
        '--json',
        type=str,
        default=config.DEFAULT_JSON_PATH,
        help=f'Path to JSON sequence data file (default: {config.DEFAULT_JSON_PATH})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (without extension, default: uses mesh name from JSON)'
    )
    
    # Quality options
    parser.add_argument(
        '--quality', '-q',
        type=str,
        choices=['low', 'medium', 'high', 'production'],
        default='high',
        help='Video quality preset (default: high)'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=str,
        choices=['480p', '720p', '1080p', '1440p', '2160p'],
        default=config.RESOLUTION,
        help=f'Video resolution (default: {config.RESOLUTION})'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=config.FRAME_RATE,
        help=f'Frame rate (default: {config.FRAME_RATE})'
    )
    
    # Preview options
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Preview animation without saving to file'
    )
    
    parser.add_argument(
        '--save-last-frame',
        action='store_true',
        help='Save only the last frame as an image'
    )
    
    # Animation options
    parser.add_argument(
        '--no-borders',
        action='store_true',
        help='Hide region borders'
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide state labels'
    )
    
    # Performance options
    parser.add_argument(
        '--write-all',
        action='store_true',
        help='Write all frames to video file (slower but more compatible)'
    )
    
    return parser.parse_args()


def configure_manim(args):
    """Configure Manim settings based on arguments."""
    # Set quality
    quality_map = {
        'low': {'pixel_height': 480, 'pixel_width': 854, 'frame_rate': 15},
        'medium': {'pixel_height': 720, 'pixel_width': 1280, 'frame_rate': 30},
        'high': {'pixel_height': 1080, 'pixel_width': 1920, 'frame_rate': 60},
        'production': {'pixel_height': 1080, 'pixel_width': 1920, 'frame_rate': 60}
    }
    
    # Resolution map
    resolution_map = {
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '2160p': (3840, 2160)
    }
    
    # Apply quality preset
    if args.quality in quality_map:
        preset = quality_map[args.quality]
        manim_config.pixel_height = preset['pixel_height']
        manim_config.pixel_width = preset['pixel_width']
        manim_config.frame_rate = preset['frame_rate']
    
    # Override with specific resolution if provided
    if args.resolution:
        width, height = resolution_map.get(args.resolution, (1920, 1080))
        manim_config.pixel_width = width
        manim_config.pixel_height = height
    
    # Override with specific FPS if provided
    if args.fps:
        manim_config.frame_rate = args.fps
    
    # Preview mode
    if args.preview:
        manim_config.preview = True
        manim_config.write_to_movie = False
    
    # Save last frame only
    if args.save_last_frame:
        manim_config.write_to_movie = False
        manim_config.save_last_frame = True
    
    # Write all frames
    if args.write_all:
        manim_config.write_all = True
    
    # Output file
    if args.output:
        manim_config.output_file = args.output
    
    # Background color
    manim_config.background_color = config.BACKGROUND_COLOR
    
    print("\n" + "="*60)
    print("Manim Configuration:")
    print("="*60)
    print(f"Resolution: {manim_config.pixel_width}x{manim_config.pixel_height}")
    print(f"Frame Rate: {manim_config.frame_rate} FPS")
    print(f"Quality: {args.quality}")
    print(f"Preview Mode: {args.preview}")
    print(f"Output Path: {manim_config.output_file or 'default'}")
    print("="*60 + "\n")


def update_config(args):
    """Update config module based on arguments."""
    if args.no_borders:
        config.SHOW_REGION_BORDERS = False
    
    if args.no_labels:
        config.SHOW_STATE_LABELS = False


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Verify JSON file exists
    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = Path(__file__).parent.parent.parent / args.json
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    print(f"\nLoading animation data from: {json_path}")
    
    # Configure Manim
    configure_manim(args)
    
    # Update config
    update_config(args)
    
    # Create and render scene
    print("\nGenerating animation...\n")
    scene = MeshGenerationScene(json_path=str(json_path))
    scene.render()
    
    print("\n" + "="*60)
    print("Animation generation complete!")
    print("="*60)
    
    # Show output location
    if not args.preview:
        # Manim saves to: media/videos/{scene_name}/{resolution}/ (relative to root dir)
        import os
        root_dir = Path(os.getcwd())
        scene_name = "mesh_generation_scene"
        quality_dir = f"{manim_config.pixel_height}p{manim_config.frame_rate}"
        output_path = root_dir / "media" / "videos" / scene_name / quality_dir
        video_file = output_path / "MeshGenerationScene.mp4"
        print(f"Output location: {output_path}")
        print(f"Video file: {video_file}")
    
    print()


if __name__ == "__main__":
    main()
