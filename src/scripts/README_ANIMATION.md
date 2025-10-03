# Animation Sequence Data Generator

## Overview

This script (`animation.py`) generates complete sequence data for mesh generation visualization. It captures all states from the initial boundary to the final mesh, producing a JSON file ready for Manim animation rendering.

## Output Data Structure

The script generates a JSON file containing:

1. **Mesh Point Data** - Complete adjacency relationships between vertices
2. **Boundary Data** - Current boundary vertices at each step
3. **Local Environment Data** - Reference point and its neighbors
4. **Action Data** - Action type, coordinates, and validation status

## Prerequisites

### Environment Setup

The script requires the `rl-mesh-generation` conda environment:

```bash
# Activate the environment
conda activate rl-mesh-generation
```

### Required Files

Ensure the following files exist:
- **Mesh file**: `data/mesh/{MESH_NAME}.txt`
- **Model file**: `data/models/{MODEL_NAME}.zip`

## Configuration

All parameters are configured at the top of `animation.py`:

```python
# ------------- Mesh and Model Configuration -------------
MODEL_NAME = 'basic1-robust68.886'     # Model name (without .zip)
MESH_NAME = 'basic1'                   # Mesh name (without .txt)

# ------------- Predictor Configuration -------------
PREDICTOR_N = 2                        # Number of neighbor vertices
PREDICTOR_G = 3                        # Number of fan sector points
PREDICTOR_BETA = 6                     # Fan angle parameter

# ------------- Reference Point Selector Configuration -------------
REF_SELECTOR_TYPE = 'RL'               # 'RL' or 'default'
REF_SELECTOR_N = 2                     # Should match PREDICTOR_N

# ------------- Output Configuration -------------
OUTPUT_FILENAME = f'{MESH_NAME}_sequence.json'
PRETTY_PRINT = True                    # Format JSON with indentation
INCLUDE_METADATA = True                # Include metadata section

# ------------- Execution Configuration -------------
MAX_STEPS = 1000                       # Safety limit
VERBOSE = True                         # Print progress
```

## Usage

### Basic Usage

```bash
# Make sure you're in the project root directory
cd D:\Projects\RL-MESH-GENERATION

# Activate the conda environment
conda activate rl-mesh-generation

# Run the script
python src/scripts/animation.py
```

### Customizing Parameters

Edit the configuration section at the top of `animation.py`:

```python
# Example: Use a different mesh and model
MODEL_NAME = 'dolphine3-reward85.123'
MESH_NAME = 'dolphine3'

# Example: Adjust predictor parameters
PREDICTOR_N = 3
PREDICTOR_G = 5
PREDICTOR_BETA = 8
```

## Output

### Output Location

By default, the JSON file is saved to:
```
data/animation_data/{MESH_NAME}_sequence.json
```

### Output Format

```json
{
  "metadata": {
    "mesh_name": "basic1",
    "model_name": "basic1-robust68.886",
    "total_states": 15,
    "predictor_type": "RL",
    "predictor_config": {"n": 2, "g": 3, "beta": 6},
    "ref_selector_config": {"type": "RL", "n": 2},
    "timestamp": "2025-10-03T22:44:16",
    "generation_completed": true,
    "initial_boundary_size": 20,
    "final_boundary_size": 4,
    "total_elements_generated": 16
  },
  "states": [
    {
      "state_id": 0,
      "step": 0,
      "mesh_points": {
        "[0,0]": [[1,0], [0,1]],
        "[1,0]": [[0,0], [1,1]],
        ...
      },
      "boundary": [[0,0], [1,0], [1,1], ...],
      "local_env": {
        "reference_vertex_idx": 2,
        "reference_coords": [0.5, 0.5],
        "neighbors": [[0.3,0.4], [0.4,0.5], ...],
        "fan_points": [[0.6,0.7], null, [0.8,0.6]],
        "n": 2,
        "g": 3,
        "beta": 6
      },
      "action": null,
      "generated_element": null,
      "is_initial": true,
      "is_terminal": false
    },
    {
      "state_id": 1,
      "step": 1,
      "mesh_points": {...},
      "boundary": [...],
      "local_env": {...},
      "action": {
        "action_type": "type1",
        "reference_vertex_idx": 2,
        "new_coords": [0.55, 0.45],
        "is_valid": true,
        "action_attempted": {
          "edge": [...],
          "vertex": [0.55, 0.45]
        }
      },
      "generated_element": [[0,0], [1,0], [0.55,0.45], [0,1]],
      "is_terminal": false
    },
    ...
  ]
}
```

## Data Fields Explanation

### State Object

| Field | Type | Description |
|-------|------|-------------|
| `state_id` | int | Sequential state identifier (0-indexed) |
| `step` | int | Current generation step |
| `mesh_points` | dict | Vertex adjacency dictionary (stringified coords as keys) |
| `boundary` | array | Current boundary vertex coordinates |
| `local_env` | object | Local environment around reference point |
| `action` | object/null | Action that led to this state (null for initial) |
| `generated_element` | array/null | Generated quadrilateral vertices |
| `is_initial` | bool | Whether this is the initial state |
| `is_terminal` | bool | Whether generation is completed |

### Local Environment Object

| Field | Type | Description |
|-------|------|-------------|
| `reference_vertex_idx` | int | Index of reference vertex in boundary |
| `reference_coords` | array | [x, y] coordinates of reference vertex |
| `neighbors` | array | List of neighbor coordinates (includes ref) |
| `fan_points` | array | Fan sector sampled points (can contain null) |
| `n` | int | Number of neighbors on each side |
| `g` | int | Number of fan sectors |
| `beta` | float | Fan radius factor (multiplier of base length) |

### Action Object

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | string | 'type0_left', 'type0_right', or 'type1' |
| `reference_vertex_idx` | int | Reference vertex used for action |
| `new_coords` | array/null | New vertex coordinates (type1 only) |
| `is_valid` | bool | Whether action passed validation |
| `action_attempted` | object | Edges/vertices the action attempted to create |

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure you're in the correct environment:

```bash
conda activate rl-mesh-generation
python src/scripts/animation.py
```

### File Not Found Errors

Check that your mesh and model files exist:

```bash
# Check mesh file
ls data/mesh/basic1.txt

# Check model file
ls data/models/basic1-robust68.886.zip
```

### Generation Failures

If generation stops early:
1. Check the console output for error messages
2. Try a different mesh or model
3. Adjust `MAX_STEPS` if needed
4. Set `VERBOSE = True` for detailed debugging

## Performance Notes

- **Typical execution time**: 1-10 seconds for small meshes (10-30 vertices)
- **Output file size**: 10-500 KB depending on mesh complexity
- **Memory usage**: Minimal (<100 MB for most cases)

## Integration with Manim

The generated JSON file is designed to be consumed by Manim animation scripts. Each state represents a frame or keyframe in the animation.

Example Manim usage:

```python
import json
from manim import *

class MeshGeneration(Scene):
    def construct(self):
        with open('data/animation_data/basic1_sequence.json') as f:
            data = json.load(f)
        
        for state in data['states']:
            # Render mesh
            self.render_mesh(state['mesh_points'])
            # Render boundary
            self.render_boundary(state['boundary'])
            # Highlight local env
            self.highlight_local_env(state['local_env'])
            # Show action
            if state['action']:
                self.show_action(state['action'])
            self.wait(0.5)
```

## Fan Points Explanation

### What are Fan Points?

Fan points are **sampled boundary vertices** within a fan-shaped sector around the reference point. They provide information about the boundary structure in the action space.

### How Fan Points Work

1. **Fan Construction**:
   - Center: Reference vertex
   - Radius: `beta × average_neighbor_length` (default: 6× average edge length)
   - Angular range: From right neighbor to left neighbor (clockwise)

2. **Sector Sampling**:
   - The fan is divided into `g` equal sectors (default: 3)
   - For each sector, find the **closest boundary vertex** within the radius
   - If no vertex exists in a sector, that entry is `null`

3. **Example** (with g=3):
   ```
   Sector 0: [0.6, 0.7]  ← Found a vertex at this location
   Sector 1: null         ← No vertex found in this sector
   Sector 2: [0.8, 0.6]  ← Found a vertex here
   ```

### Why Fan Points Matter

- They help the RL model understand the **local boundary geometry**
- They indicate **potential action space** for Type1 actions (adding new vertices)
- They provide spatial awareness beyond immediate neighbors
- Null values indicate "free space" where new vertices could be placed

### Parameters

- **`g` (default: 3)**: Number of fan sectors
  - More sectors = finer sampling resolution
  - Typical values: 3-5

- **`beta` (default: 6)**: Fan radius multiplier
  - Larger beta = wider search area
  - Typical values: 4-8

## Notes

- The script **does not modify any source code** - it only reads data through existing APIs
- All data is captured in real-time during generation
- The output is deterministic given the same model and mesh
- Coordinates are stored as-is without normalization for visualization flexibility
- **Fan points are essential** for understanding the RL model's perception of the action space
