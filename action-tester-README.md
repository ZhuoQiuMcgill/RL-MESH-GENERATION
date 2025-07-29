# Action Tester - RL Mesh Generation Interactive Tool

## Overview

Action Tester is a new interactive web tool that allows users to test reinforcement learning actions on mesh boundaries. This tool provides a visual interface for understanding how RL actions work in the mesh generation process.

## Features

### 1. Mesh Selection
- Select from available mesh files in the data/mesh directory
- View mesh information including vertex count and file size
- Real-time boundary visualization on canvas

### 2. Reference Point Selection
- **Find Reference Point** button automatically selects the optimal reference point
- Uses the same algorithm as the RL environment (`get_ref_vertex()`)
- Displays reference point information:
  - Index position
  - Coordinates
  - Interior angle value
- Visual highlighting of reference point and neighbor vertices

### 3. Action Testing
Three action types are supported, matching the RL environment:

#### Type0 Left
- Generates a quadrilateral element on the left side of the reference point
- No additional user input required

#### Type0 Right  
- Generates a quadrilateral element on the right side of the reference point
- No additional user input required

#### Type1 (Click to Draw)
- Allows user to click on canvas to place a new vertex
- Shows instruction prompt for user interaction
- Converts clicked coordinates to normalized polar coordinates
- Uses the same normalization method as the RL environment

### 4. Action Execution & Validation
- **Execute** button validates and tests the selected action
- Uses the exact same validation logic as the RL environment
- Shows execution results:
  - Whether the action is valid
  - For Type1 actions: displays normalized polar coordinates (r, θ)
- Real-time logging of all operations

### 5. Visual Feedback
- Canvas visualization matches the training page layout
- Reference point and neighbors are highlighted
- Real-time coordinate display for Type1 actions
- Status indicators for all components

## Technical Implementation

### Frontend (`action-tester.html` + `action-tester.js`)
- Built using the same architecture as existing tools
- Reuses `CanvasRenderer`, `UIController`, and `ApiClient` modules
- Responsive design with left control panel and right visualization area

### Backend API (`/action/*` endpoints)
- **`GET /action/health`** - Health check
- **`GET /action/find-ref-point/{mesh_name}`** - Find reference point for mesh
- **`POST /action/execute`** - Execute and validate action
- **`GET /action/info`** - Get action configuration information

### Key Components
1. **ActionTesterService** - Core backend logic
2. **ActionManager** integration - Uses exact same action processing as RL environment
3. **Boundary** class integration - Same reference point selection algorithm
4. **Coordinate normalization** - Identical to RL environment implementation

## Usage Workflow

1. **Start Backend Server**
   ```bash
   cd D:\Projects\RL-MESH-GENERATION
   python -m src.ui.app
   ```

2. **Open Action Tester**
   - Navigate to `http://localhost:5000/tools/action-tester.html`
   - Or use the dashboard link from `index.html`

3. **Test Actions**
   - Select a mesh from dropdown
   - Click "Find Reference Point"
   - Choose an action type (Type0 Left/Right/Type1)
   - For Type1: click on canvas to place point
   - Click "Execute" to validate action

## Integration with RL Environment

The Action Tester uses the **identical** algorithms and code paths as the actual RL training:

- **Reference Point Selection**: `boundary.get_ref_vertex()`
- **Action Decoding**: `action_manager.decode_action()`
- **Action Validation**: `action_manager.is_valid()`
- **Coordinate Normalization**: `normalize_coordinates()` with same parameters

This ensures that the testing results accurately reflect what would happen during actual RL training.

## Files Added/Modified

### New Files
- `tools/action-tester.html` - Main frontend page
- `tools/js/action-tester.js` - Frontend JavaScript module
- `src/ui/api/action.py` - Backend API blueprint
- `test_action_api.py` - API testing script

### Modified Files
- `src/ui/api/__init__.py` - Register new action blueprint
- `src/ui/app.py` - Add CORS configuration for /action/* routes
- `tools/js/api-client.js` - Add action API methods
- `tools/index.html` - Add Action Tester to dashboard

## Testing

Run the test script to verify API functionality:
```bash
python test_action_api.py
```

This will test:
- API health checks
- Mesh loading
- Reference point finding  
- Action execution for all three types

## Future Enhancements

Potential improvements for the Action Tester:

1. **Batch Testing** - Test multiple actions in sequence
2. **Action History** - Track previously executed actions
3. **Mesh Generation** - Actually generate mesh elements (not just validate)
4. **Quality Metrics** - Show element and boundary quality scores
5. **Export Results** - Save test results to file
6. **Animation** - Animate the action execution process

## API Documentation

### Find Reference Point
```
GET /action/find-ref-point/{mesh_name}

Response:
{
  "success": true,
  "reference_point": {
    "index": 20,
    "coordinates": [177.014, 400.568],
    "interior_angle": 89.83,
    "neighbor_vertices": [...]
  }
}
```

### Execute Action
```
POST /action/execute
Content-Type: application/json

Request:
{
  "mesh_name": "1",
  "action_type": "type1",
  "reference_point_index": 20,
  "clicked_point": [0.5, 0.5]  // Only for type1
}

Response:
{
  "success": true,
  "result": {
    "valid": false,
    "action_name": "type1",
    "decoded_coords": [[0.5, 0.5]],
    "polar_coordinates": {  // Only for type1
      "r": 4.002,
      "theta": -2.012
    }
  }
}
```

This tool provides an excellent way to understand and debug RL action behavior interactively!