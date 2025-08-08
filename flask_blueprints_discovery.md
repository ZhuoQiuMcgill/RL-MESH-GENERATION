# Flask Blueprints API Discovery

This document provides a comprehensive overview of all Flask blueprints in the RL-MESH-GENERATION project, listing every route, method, path, parameters, and response schema.

## 1. Training Blueprint (`training_bp`)

**URL Prefix:** `/training`

### Routes:

#### 1.1 Start Training
- **Path:** `/training/start`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `mesh_name` (string): Name of the mesh file
    - `subfolder` (string, optional): Subfolder name, defaults to "mesh"
    - `max_timesteps` (integer, optional): Maximum training timesteps
    - `max_steps` (integer, optional): Maximum steps per episode
    - `description` (string, optional): Training description
    - `checkpoint_name` (string, optional): Name of checkpoint to save
    - `from_checkpoint` (boolean, optional): Whether to resume from checkpoint
- **Response Schema:**
  ```json
  {
    "success": boolean,
    "error": string (if error),
    // Additional training result fields
  }
  ```
- **Status Codes:** 200 (Success), 400 (Runtime/Value Error), 500 (Internal Error)

#### 1.2 Stop Training
- **Path:** `/training/stop`
- **Method:** `POST`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "success": boolean,
    "error": string (if error),
    // Additional stop result fields
  }
  ```
- **Status Codes:** 200 (Success), 500 (Internal Error)

#### 1.3 Get Training Status
- **Path:** `/training/status`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "running": boolean,
    "status": string,
    "stats": object,
    "progress": object,
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Always returns 200)

#### 1.4 Health Check
- **Path:** `/training/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "training-api",
    "manager_running": boolean,
    "error": string (if error),
    "timestamp": number
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 2. Mesh Blueprint (`mesh_bp`)

**URL Prefix:** `/mesh`

### Routes:

#### 2.1 List Meshes
- **Path:** `/mesh/list`
- **Method:** `GET`
- **Parameters:**
  - **Query:**
    - `subfolder` (string, optional): Subfolder name, defaults to "mesh"
- **Response Schema:**
  ```json
  {
    "meshes": string[],
    "count": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 2.2 Get Mesh Info
- **Path:** `/mesh/info/<n>`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `n` (string): Mesh file name
  - **Query:**
    - `subfolder` (string, optional): Subfolder name, defaults to "mesh"
- **Response Schema:**
  ```json
  {
    "name": string,
    "exists": boolean,
    "error": string (if error),
    // Additional mesh info fields
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 2.3 Get Mesh Boundary
- **Path:** `/mesh/boundary/<n>`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `n` (string): Mesh file name
  - **Query:**
    - `subfolder` (string, optional): Subfolder name, defaults to "mesh"
- **Response Schema:**
  ```json
  {
    "mesh_name": string,
    "subfolder": string,
    "boundary_vertices": number[][],
    "vertex_count": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (File Not Found), 500 (Error)

#### 2.4 Health Check
- **Path:** `/mesh/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "mesh-api",
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 3. Predict Blueprint (`predict_bp`)

**URL Prefix:** `/predict`

### Routes:

#### 3.1 List Components
- **Path:** `/predict/components`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "predictors": {
      [key: string]: {
        "name": string,
        "description": string,
        "parameters": string[]
      }
    },
    "reference_selectors": {
      [key: string]: {
        "name": string,
        "description": string,
        "parameters": string[]
      }
    },
    "initial_meshes": string[],
    "trained_models": {
      "name": string,
      "path": string,
      "size": number,
      "description": string
    }[],
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 3.2 Create Session
- **Path:** `/predict/session/create`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `mesh_name` (string): Name of the mesh file
    - `predictor_type` (string): Type of predictor ("RL")
    - `predictor_config` (object): Predictor configuration
      - `model_path` (string): Path to trained model
      - `n` (number, optional): Parameter n, defaults to 2
      - `g` (number, optional): Parameter g, defaults to 3
      - `beta` (number, optional): Parameter beta, defaults to 6
    - `ref_selector_type` (string, optional): Reference selector type, defaults to "default"
    - `ref_selector_config` (object, optional): Reference selector configuration
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "initial_status": object,
    "config": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 3.3 Get Session Status
- **Path:** `/predict/session/<session_id>/status`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "status": object,
    "config": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.4 Update Session Config
- **Path:** `/predict/session/<session_id>/config`
- **Method:** `PUT`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
  - **Body (JSON):**
    - `predictor_type` (string, optional): New predictor type
    - `predictor_config` (object, optional): New predictor configuration
    - `ref_selector_type` (string, optional): New reference selector type
    - `ref_selector_config` (object, optional): New reference selector configuration
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "status": object,
    "config": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 404 (Session Not Found), 500 (Error)

#### 3.5 Next Step
- **Path:** `/predict/session/<session_id>/next`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "step_result": {
      "success": boolean,
      "element": array,
      "message": string,
      "action_info": object
    },
    "status": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.6 Previous Step (Undo)
- **Path:** `/predict/session/<session_id>/prev`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "undo_result": object,
    "status": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.7 Process All Steps
- **Path:** `/predict/session/<session_id>/process_all`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "steps_executed": number,
    "completion_reason": string,
    "step_history": object[],
    "final_status": object,
    "initial_step": number,
    "final_step": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.8 Reset Session
- **Path:** `/predict/session/<session_id>/reset`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "reset_result": object,
    "status": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.9 Get Session History
- **Path:** `/predict/session/<session_id>/history`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "history": object[],
    "total_actions": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.10 Delete Session
- **Path:** `/predict/session/<session_id>`
- **Method:** `DELETE`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
- **Response Schema:**
  ```json
  {
    "message": string,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Session Not Found), 500 (Error)

#### 3.11 List Sessions
- **Path:** `/predict/sessions`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "sessions": {
      [session_id: string]: {
        "config": object,
        "status": object,
        "history_length": number
      }
    },
    "total_sessions": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 3.12 Get Reference Point
- **Path:** `/predict/session/<session_id>/reference_point`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
  - **Query:**
    - `selector_type` (string, optional): Override selector type
    - `selector_config` (string, optional): Override selector config as JSON string
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "reference_point": {
      "reference_vertex_idx": number,
      "reference_vertex_coords": number[],
      "selector_info": object,
      "boundary_context": object,
      "session_status": object
    },
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 404 (Session Not Found), 500 (Error)

#### 3.13 Preview Reference Point
- **Path:** `/predict/reference_point/preview`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `mesh_name` (string): Name of the mesh file
    - `ref_selector_type` (string): Reference selector type
    - `ref_selector_config` (object, optional): Reference selector configuration
- **Response Schema:**
  ```json
  {
    "preview": {
      "mesh_name": string,
      "reference_vertex_idx": number,
      "reference_vertex_coords": number[],
      "selector_info": object,
      "boundary_context": object,
      "boundary_vertices": number[][]
    },
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 3.14 Get Session Quality
- **Path:** `/predict/session/<session_id>/quality`
- **Method:** `GET`, `OPTIONS`
- **Parameters:**
  - **Path:**
    - `session_id` (string): Session identifier
  - **Query:**
    - `method` (string, optional): Quality calculation method, defaults to "hybrid"
    - `gamma` (number, optional): Gamma parameter for hybrid method, defaults to 1.0
- **Response Schema:**
  ```json
  {
    "session_id": string,
    "element_count": number,
    "valid_element_count": number,
    "average_quality": number,
    "min_quality": number,
    "max_quality": number,
    "quality_scores": object[],
    "method": string,
    "gamma": number,
    "success": boolean,
    "message": string (if no elements),
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 404 (Session Not Found), 500 (Error)

#### 3.15 Get Quality Methods
- **Path:** `/predict/quality/methods`
- **Method:** `GET`, `OPTIONS`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "methods": string[],
    "method_info": object,
    "count": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 3.16 Health Check
- **Path:** `/predict/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "predict-api",
    "active_sessions": number,
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 4. Training History Blueprint (`training_history_bp`)

**URL Prefix:** `/training/history`

### Routes:

#### 4.1 List Training History
- **Path:** `/training/history/list`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "training_ids": string[],
    "count": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 4.2 Get Training Info
- **Path:** `/training/history/info/<training_id>`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `training_id` (string): Training session ID
- **Response Schema:**
  ```json
  {
    "training_id": string,
    "detail_length": number,
    "best_episode": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Training Not Found), 500 (Error)

#### 4.3 Get Episode Data
- **Path:** `/training/history/episode/<training_id>/<int:episode_index>`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `training_id` (string): Training session ID
    - `episode_index` (integer): Episode index
- **Response Schema:**
  ```json
  {
    "training_id": string,
    "episode_index": number,
    "episode_data": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Index Out of Range), 404 (Training Not Found), 500 (Error)

#### 4.4 Health Check
- **Path:** `/training/history/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "training-history-api",
    "available_trainings": number,
    "current_focus": string,
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 5. Quality Blueprint (`quality_bp`)

**URL Prefix:** `/quality`

### Routes:

#### 5.1 Get Quality Methods
- **Path:** `/quality/methods`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "methods": string[],
    "method_info": object,
    "count": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 5.2 Calculate Quality
- **Path:** `/quality/calculate`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `vertices` (number[][]): Array of 4 vertex coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    - `method` (string): Quality calculation method name
    - `gamma` (number, optional): Gamma parameter for hybrid method, defaults to 1.0
- **Response Schema:**
  ```json
  {
    "quality_score": number,
    "method": string,
    "vertices": number[][],
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 5.3 Health Check
- **Path:** `/quality/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "quality-api",
    "available_methods": number,
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 6. Action Blueprint (`action_bp`)

**URL Prefix:** `/action`

### Routes:

#### 6.1 Health Check
- **Path:** `/action/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy",
    "service": "action_tester"
  }
  ```
- **Status Codes:** 200 (Success)

#### 6.2 Find Reference Point
- **Path:** `/action/find-ref-point/<mesh_name>`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `mesh_name` (string): Name of the mesh file
- **Response Schema:**
  ```json
  {
    "success": boolean,
    "reference_point": {
      "index": number,
      "coordinates": number[],
      "interior_angle": number,
      "neighbor_vertices": number[][]
    },
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 6.3 Execute Action
- **Path:** `/action/execute`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `mesh_name` (string): Name of the mesh file
    - `action_type` (string): Type of action ("type0_left", "type0_right", "type1")
    - `reference_point_index` (number): Index of the reference point
    - `clicked_point` (number[], optional): Clicked point coordinates for type1 actions
- **Response Schema:**
  ```json
  {
    "success": boolean,
    "result": {
      "valid": boolean,
      "action_name": string,
      "decoded_coords": number[],
      "generated_element": number[][],
      "polar_coordinates": {
        "r": number,
        "theta": number
      } // Only for type1 actions
    },
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 6.4 Validate Action
- **Path:** `/action/validate/<action_type>`
- **Method:** `POST`
- **Parameters:**
  - **Path:**
    - `action_type` (string): Type of action to validate
  - **Body (JSON):** Same as execute action
- **Response Schema:** Same as execute action
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 6.5 Get Action Info
- **Path:** `/action/info`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "success": boolean,
    "action_info": object,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## 7. Geometry Blueprint (`geometry_bp`)

**URL Prefix:** `/geometry`

### Routes:

#### 7.1 Normalize Coordinates
- **Path:** `/geometry/normalize`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `coordinates` (number[][]): Array of coordinate pairs (must be odd number of points)
- **Response Schema:**
  ```json
  {
    "status": "success|error",
    "original_coordinates": number[][],
    "normalized_coordinates": number[][],
    "ref_vertex_index": number,
    "right_neighbor_index": number,
    "scale_factor": number,
    "average_edge_length": number,
    "edges_used_for_scale": number,
    "message": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 7.2 Health Check
- **Path:** `/geometry/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "success",
    "message": "Geometry API is running",
    "endpoints": string[]
  }
  ```
- **Status Codes:** 200 (Success)

## 8. Checkpoint Blueprint (`checkpoint_bp`)

**URL Prefix:** `/checkpoint`

### Routes:

#### 8.1 List Checkpoints
- **Path:** `/checkpoint/list`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "checkpoints": string[],
    "count": number,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 8.2 Get Checkpoint Info
- **Path:** `/checkpoint/info/<checkpoint_name>`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `checkpoint_name` (string): Checkpoint name (without .pth extension)
- **Response Schema:**
  ```json
  {
    "checkpoint_info": object,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Checkpoint Not Found), 500 (Error)

#### 8.3 Validate Checkpoint
- **Path:** `/checkpoint/validate/<checkpoint_name>`
- **Method:** `GET`
- **Parameters:**
  - **Path:**
    - `checkpoint_name` (string): Checkpoint name (without .pth extension)
- **Response Schema:**
  ```json
  {
    "checkpoint_name": string,
    "is_valid": boolean,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

#### 8.4 Delete Checkpoint
- **Path:** `/checkpoint/delete/<checkpoint_name>`
- **Method:** `DELETE`
- **Parameters:**
  - **Path:**
    - `checkpoint_name` (string): Checkpoint name (without .pth extension)
- **Response Schema:**
  ```json
  {
    "message": string,
    "checkpoint_name": string,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 404 (Checkpoint Not Found), 500 (Error)

#### 8.5 Copy Checkpoint from History
- **Path:** `/checkpoint/copy`
- **Method:** `POST`
- **Parameters:**
  - **Body (JSON):**
    - `training_id` (string): Training session ID to copy from
    - `checkpoint_name` (string, optional): New checkpoint name, defaults to training_id
- **Response Schema:**
  ```json
  {
    "message": string,
    "training_id": string,
    "checkpoint_name": string,
    "success": boolean,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 400 (Bad Request), 500 (Error)

#### 8.6 Health Check
- **Path:** `/checkpoint/health`
- **Method:** `GET`
- **Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy|unhealthy",
    "service": "checkpoint-api",
    "checkpoint_count": number,
    "timestamp": number,
    "error": string (if error)
  }
  ```
- **Status Codes:** 200 (Success), 500 (Error)

## Summary

The Flask application contains 8 blueprints with a total of:
- **52 API endpoints**
- **8 different URL prefixes**
- **4 HTTP methods** (GET, POST, PUT, DELETE)
- **Comprehensive error handling** with structured JSON responses
- **Health check endpoints** for all services
- **Consistent response schemas** with success/error indicators

### Blueprint Distribution:
1. **Predict Blueprint**: 16 endpoints (most complex)
2. **Checkpoint Blueprint**: 6 endpoints
3. **Training Blueprint**: 4 endpoints
4. **Training History Blueprint**: 4 endpoints
5. **Mesh Blueprint**: 4 endpoints
6. **Action Blueprint**: 5 endpoints
7. **Quality Blueprint**: 3 endpoints
8. **Geometry Blueprint**: 2 endpoints

All blueprints follow RESTful conventions and provide comprehensive error handling with appropriate HTTP status codes.
