"""Animation Sequence Data Generator

This script generates complete sequence data for mesh generation visualization.
It captures all states from initial boundary to final mesh, including:
- Mesh point data (adjacency relationships)
- Boundary data (current boundary vertices)
- Local environment data (reference point and neighbors)
- Action data (action type, coordinates, validation status)

The output is a JSON file ready for Manim animation rendering.

Usage:
    Simply run: python src/scripts/animation.py
    All parameters are configured at the top of this file.
"""

import os
import sys
import json
import datetime
import traceback

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mesh_generator.mesh_generator import MeshGenerator
from src.mesh_generator.rl_predictor import RLPredictor
from src.utils.importer import MeshImporter
from src.geometry.reference_point_selectors import RLReferencePointSelector


# ============================================================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================================================

# ------------- Path Configuration -------------
DATA_DIR = os.path.join(os.getcwd(), 'data')
SAVE_DIR = os.path.join(DATA_DIR, 'animation_data')
MESH_DIR = os.path.join(DATA_DIR, 'mesh')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# ------------- Mesh and Model Configuration -------------
MODEL_NAME = 'basic1-robust68.886'     # Model name (without .zip extension)
MESH_NAME = 'basic1'                   # Mesh name (without .txt extension)

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + '.zip')
MESH_PATH = os.path.join(MESH_DIR, MESH_NAME + '.txt')

# ------------- Predictor Configuration -------------
PREDICTOR_TYPE = 'RL'                  # Predictor type (currently only 'RL' supported)
PREDICTOR_N = 2                        # Number of neighbor vertices to consider
PREDICTOR_G = 3                        # Number of fan sector points
PREDICTOR_BETA = 6                     # Fan angle parameter

# ------------- Reference Point Selector Configuration -------------
REF_SELECTOR_TYPE = 'RL'               # Reference selector type ('RL' or 'default')
REF_SELECTOR_N = 2                     # Selector n parameter (should match PREDICTOR_N)

# ------------- Output Configuration -------------
OUTPUT_FILENAME = f'{MESH_NAME}_sequence.json'
OUTPUT_PATH = os.path.join(SAVE_DIR, OUTPUT_FILENAME)
PRETTY_PRINT = True                    # Format JSON with indentation
INCLUDE_METADATA = True                # Include generation metadata in output

# ------------- Execution Configuration -------------
MAX_STEPS = 1000                       # Maximum inference steps (safety limit)
VERBOSE = True                         # Print detailed progress information

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


def get_local_env(generator, n, g, beta):
    """Extract local environment data around the reference point.
    
    Args:
        generator: MeshGenerator instance
        n: Number of neighbors on each side of reference point
        g: Number of fan sector points
        beta: Fan radius factor
        
    Returns:
        dict: Local environment data containing:
            - reference_vertex_idx: Index of the reference vertex
            - reference_coords: Coordinates of reference vertex
            - neighbors: List of neighbor coordinates
            - fan_points: List of fan sector sampled points (can contain None)
            - n: Number of neighbors parameter
            - g: Number of fan sectors
            - beta: Fan radius factor
    """
    state_info = generator.get_current_state_info()
    boundary = state_info['boundary']
    ref_idx = state_info['reference_vertex_idx']
    
    # Get neighbor vertices (includes reference point)
    local_env_vertices = boundary.get_neighbors(ref_idx, n)
    
    # Get fan sector points
    try:
        fan_points = boundary.get_fan_points(ref_idx, n, beta, g)
        # Convert to list format, preserving None values
        fan_points_list = [list(p) if p is not None else None for p in fan_points]
    except Exception:
        fan_points_list = [None] * g
    
    return {
        'reference_vertex_idx': ref_idx,
        'reference_coords': list(boundary.get_vertex_by_index(ref_idx)),
        'neighbors': [list(v) for v in local_env_vertices],
        'fan_points': fan_points_list,
        'n': n,
        'g': g,
        'beta': beta
    }


def capture_current_state(generator, n, g, beta, action=None, element=None, is_initial=False):
    """Capture complete state at current step.
    
    Args:
        generator: MeshGenerator instance
        n: Number of neighbors for local environment
        g: Number of fan sector points
        beta: Fan radius factor
        action: Action information dict (None for initial state)
        element: Generated element vertices (None if no element generated)
        is_initial: Whether this is the initial state
        
    Returns:
        dict: Complete state data including mesh, boundary, local_env, and action
    """
    status = generator.get_status()
    
    state = {
        'state_id': generator.current_step,
        'step': generator.current_step,
        'mesh_points': status['mesh_data'],
        'boundary': status['boundary_vertices'],
        'local_env': get_local_env(generator, n, g, beta),
        'action': action,
        'generated_element': element if element else None,
        'is_initial': is_initial,
        'is_terminal': status['is_completed']
    }
    
    return state


def generate_sequence(generator, predictor_n, predictor_g, predictor_beta, max_steps, verbose):
    """Generate complete state sequence from initial boundary to completion.
    
    Args:
        generator: Initialized MeshGenerator instance
        predictor_n: Number of neighbors for predictor
        predictor_g: Number of fan sector points
        predictor_beta: Fan radius factor
        max_steps: Maximum number of steps to prevent infinite loops
        verbose: Whether to print progress information
        
    Returns:
        list: List of state dictionaries representing the complete sequence
    """
    states = []
    
    # Capture initial state (State 0)
    initial_boundary_size = generator.boundary.size()
    if verbose:
        print(f"[State 0] Initial state - Boundary size: {initial_boundary_size}")
    
    states.append(capture_current_state(generator, predictor_n, predictor_g, predictor_beta, is_initial=True))
    
    # Main inference loop
    step_count = 0
    while not generator.check_complete() and step_count < max_steps:
        # Execute one step
        step_result = generator.step()
        
        # Check if step was successful
        if not step_result['success']:
            if verbose:
                code = step_result.get('code', -1)
                message = step_result.get('message', 'Unknown error')
                print(f"[State {step_count+1}] Failed (code={code}): {message}")
            break
        
        # Extract action and element information
        action_info = step_result.get('action_info')
        element = step_result.get('element')
        
        # Capture new state
        state = capture_current_state(
            generator, 
            predictor_n,
            predictor_g,
            predictor_beta,
            action=action_info, 
            element=element
        )
        states.append(state)
        
        # Print progress
        if verbose:
            action_type = action_info.get('action_type', 'unknown') if action_info else 'unknown'
            boundary_size = generator.boundary.size()
            print(f"[State {step_count+1}] Action: {action_type}, Boundary size: {boundary_size}")
        
        step_count += 1
    
    # Check termination reason
    if verbose:
        if generator.check_complete():
            print(f"\n✓ Generation completed successfully after {step_count} steps")
        elif step_count >= max_steps:
            print(f"\n⚠ Reached maximum step limit ({max_steps})")
        else:
            print(f"\n✗ Generation stopped after {step_count} steps")
    
    return states


def save_sequence_data(states, output_path, mesh_name, model_name, 
                       predictor_config, ref_selector_config,
                       pretty_print, include_metadata):
    """Save sequence data to JSON file.
    
    Args:
        states: List of state dictionaries
        output_path: Output file path
        mesh_name: Name of the mesh
        model_name: Name of the model
        predictor_config: Predictor configuration dict
        ref_selector_config: Reference selector configuration dict
        pretty_print: Whether to format JSON with indentation
        include_metadata: Whether to include metadata section
    """
    data = {}
    
    # Add metadata if requested
    if include_metadata:
        data['metadata'] = {
            'mesh_name': mesh_name,
            'model_name': model_name,
            'total_states': len(states),
            'predictor_type': 'RL',
            'predictor_config': predictor_config,
            'ref_selector_config': ref_selector_config,
            'timestamp': datetime.datetime.now().isoformat(),
            'generation_completed': states[-1]['is_terminal'] if states else False,
            'initial_boundary_size': states[0]['boundary'].__len__() if states else 0,
            'final_boundary_size': states[-1]['boundary'].__len__() if states else 0,
            'total_elements_generated': len(states) - 1 if states else 0
        }
    
    data['states'] = states
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    
    # Print file size
    file_size = os.path.getsize(output_path)
    file_size_kb = file_size / 1024
    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {file_size_kb:.2f} KB")


def validate_paths():
    """Validate that all required files and directories exist.
    
    Raises:
        FileNotFoundError: If required files are missing
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not os.path.exists(MESH_PATH):
        raise FileNotFoundError(f"Mesh file not found: {MESH_PATH}")
    
    if VERBOSE:
        print("✓ All required files found")


def main():
    """Main execution function."""
    try:
        print("="*70)
        print("Animation Sequence Data Generator")
        print("="*70)
        print(f"\nMesh: {MESH_NAME}")
        print(f"Model: {MODEL_NAME}")
        print(f"Output: {OUTPUT_PATH}")
        print()
        
        # Validate paths
        if VERBOSE:
            print("Validating paths...")
        validate_paths()
        
        # Load mesh
        if VERBOSE:
            print(f"\nLoading mesh from: {MESH_PATH}")
        importer = MeshImporter()
        boundary = importer.load_boundary_by_name(MESH_NAME, "mesh")
        boundary_vertices = boundary.get_vertices()
        if VERBOSE:
            print(f"✓ Loaded boundary with {len(boundary_vertices)} vertices")
        
        # Initialize MeshGenerator
        if VERBOSE:
            print(f"\nInitializing MeshGenerator...")
        generator = MeshGenerator(boundary_vertices)
        
        # Initialize predictor
        if VERBOSE:
            print(f"Initializing {PREDICTOR_TYPE} predictor (n={PREDICTOR_N}, g={PREDICTOR_G}, beta={PREDICTOR_BETA})...")
        predictor = RLPredictor(n=PREDICTOR_N, g=PREDICTOR_G, beta=PREDICTOR_BETA)
        
        # Load model
        if VERBOSE:
            print(f"Loading model from: {MODEL_PATH}")
        predictor.init_agent(agent_path=MODEL_PATH)
        if VERBOSE:
            print("✓ Model loaded successfully")
        
        # Set predictor
        generator.set_predictor(predictor)
        generator.update_activated_predictor(PREDICTOR_TYPE)
        
        # Initialize reference selector
        if REF_SELECTOR_TYPE != 'default':
            if VERBOSE:
                print(f"Setting up {REF_SELECTOR_TYPE} reference selector (n={REF_SELECTOR_N})...")
            ref_selector = RLReferencePointSelector()
            generator.set_ref_selector(ref_selector, {'n': REF_SELECTOR_N})
        
        if VERBOSE:
            print(f"\n{'='*70}")
            print("Starting mesh generation...")
            print(f"{'='*70}\n")
        
        # Generate sequence
        states = generate_sequence(generator, PREDICTOR_N, PREDICTOR_G, PREDICTOR_BETA, MAX_STEPS, VERBOSE)
        
        if not states:
            print("\n✗ No states generated. Aborting.")
            return 1
        
        # Prepare configuration for metadata
        predictor_config = {
            'n': PREDICTOR_N,
            'g': PREDICTOR_G,
            'beta': PREDICTOR_BETA
        }
        
        ref_selector_config = {
            'type': REF_SELECTOR_TYPE,
            'n': REF_SELECTOR_N
        }
        
        # Save data
        if VERBOSE:
            print(f"\n{'='*70}")
            print("Saving sequence data...")
            print(f"{'='*70}")
        
        save_sequence_data(
            states, 
            OUTPUT_PATH, 
            MESH_NAME, 
            MODEL_NAME,
            predictor_config,
            ref_selector_config,
            PRETTY_PRINT,
            INCLUDE_METADATA
        )
        
        print(f"\n{'='*70}")
        print("✓ Animation sequence generation completed successfully!")
        print(f"{'='*70}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        if VERBOSE:
            print("\nFull traceback:")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

