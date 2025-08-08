# API Endpoint Comparison Report

**Generated**: 2025-08-07 16:39:57
**Purpose**: Compare discovered endpoints against harvested documentation

## Executive Summary

- **Total Discovered Endpoints**: 46
- **Total Documented Endpoints**: 24
- **Exact Matches**: 14
- **Undocumented Endpoints**: 32
- **Extra Documented Endpoints**: 10
- **Documentation Coverage**: 30.4%

**Overall Status**: 🔴 **Poor** - Major documentation update needed

---

## Blueprint Coverage Analysis

| Blueprint | Discovered | Matched | Coverage | Status |
|-----------|------------|---------|----------|---------|
| **training** | 4 | 4 | 100.0% | 🟢 |
| **mesh** | 4 | 2 | 50.0% | 🟠 |
| **predict** | 18 | 2 | 11.1% | 🔴 |
| **training_history** | 4 | 2 | 50.0% | 🟠 |
| **quality** | 3 | 2 | 66.7% | 🟠 |
| **action** | 5 | 1 | 20.0% | 🔴 |
| **geometry** | 2 | 1 | 50.0% | 🟠 |
| **checkpoint** | 6 | 0 | 0.0% | 🔴 |

---

## 🚨 Undocumented Endpoints

These endpoints exist in the code but are missing from documentation:

### Blueprint: `checkpoint`

#### `GET /checkpoint/list`
- **Description**: List all available checkpoints
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

#### `GET /checkpoint/info/<checkpoint_name>`
- **Description**: Get checkpoint detailed information
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

#### `DELETE /checkpoint/delete/<checkpoint_name>`
- **Description**: Delete a specific checkpoint
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

#### `POST /checkpoint/copy`
- **Description**: Copy checkpoint from training history
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

#### `GET /checkpoint/validate/<checkpoint_name>`
- **Description**: Validate if checkpoint is valid
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

#### `GET /checkpoint/health`
- **Description**: Checkpoint service health check
- **Blueprint**: checkpoint
- **Action**: Add to API documentation

### Blueprint: `predict`

#### `PUT /predict/session/<session_id>/config`
- **Description**: Update session configuration (predictor or reference selector)
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/session/<session_id>/status`
- **Description**: Get current status of prediction session
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/session/<session_id>/reference_point`
- **Description**: Get current reference point information based on session's reference selector
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/session/<session_id>/history`
- **Description**: Get session history
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/session/<session_id>/quality`
- **Description**: Calculate average element quality for all generated elements in the session
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `POST /predict/session/<session_id>/next`
- **Description**: Execute next prediction step
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/quality/methods`
- **Description**: Get all available quality calculation methods
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `OPTIONS /predict/quality/methods`
- **Description**: Get all available quality calculation methods
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/sessions`
- **Description**: List all active prediction sessions
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `OPTIONS /predict/session/<session_id>/quality`
- **Description**: Calculate average element quality for all generated elements in the session
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `POST /predict/session/<session_id>/reset`
- **Description**: Reset session to initial state
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `POST /predict/session/<session_id>/process_all`
- **Description**: Execute all steps until invalid action or completion
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `POST /predict/session/<session_id>/prev`
- **Description**: Undo previous step (go back to previous state)
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `GET /predict/health`
- **Description**: Predict service health check
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `POST /predict/reference_point/preview`
- **Description**: Preview reference point selection for a given mesh and selector configuration
- **Blueprint**: predict
- **Action**: Add to API documentation

#### `DELETE /predict/session/<session_id>`
- **Description**: Delete prediction session
- **Blueprint**: predict
- **Action**: Add to API documentation

### Blueprint: `action`

#### `POST /action/validate/<action_type>`
- **Description**: Validate a specific action type
- **Blueprint**: action
- **Action**: Add to API documentation

#### `GET /action/find-ref-point/<mesh_name>`
- **Description**: Find the reference point for a given mesh
- **Blueprint**: action
- **Action**: Add to API documentation

#### `GET /action/info`
- **Description**: Get information about available actions
- **Blueprint**: action
- **Action**: Add to API documentation

#### `GET /action/health`
- **Description**: Action service health check
- **Blueprint**: action
- **Action**: Add to API documentation

### Blueprint: `mesh`

#### `GET /mesh/boundary/<n>`
- **Description**: Retrieve boundary vertices for a specific mesh
- **Blueprint**: mesh
- **Action**: Add to API documentation

#### `GET /mesh/info/<n>`
- **Description**: Retrieve detailed information about a specific mesh file
- **Blueprint**: mesh
- **Action**: Add to API documentation

### Blueprint: `training_history`

#### `POST /training/history/info/<training_id>`
- **Description**: Get training info including detail length and best episode
- **Blueprint**: training_history
- **Action**: Add to API documentation

#### `POST /training/history/episode/<training_id>/<int:episode_index>`
- **Description**: Get detailed data for a specific episode
- **Blueprint**: training_history
- **Action**: Add to API documentation

### Blueprint: `geometry`

#### `GET /geometry/health`
- **Description**: Geometry service health check
- **Blueprint**: geometry
- **Action**: Add to API documentation

### Blueprint: `quality`

#### `GET /quality/health`
- **Description**: Quality service health check
- **Blueprint**: quality
- **Action**: Add to API documentation

## ❓ Extra Documented Endpoints

These endpoints exist in documentation but weren't found in the code:

#### `POST /predict/session/{session_id}/next`
- **Source**: predict-api.md
- **Blueprint**: predict
- **Context**: 789 deleted successfully",   "success": true } ```  ---  ### Execute Steps  #### Execute Next Step  ```http POST /predict/session/{session_id}/next ```  **Success (200 OK):** ```json {   "session_id":...
- **Action**: Verify if endpoint was removed or renamed

#### `GET /predict/session/{session_id}/status`
- **Source**: predict-api.md
- **Blueprint**: predict
- **Context**: nfig": {"n": 2}   },   "success": true } ```  ---  ### Session Management  #### Get Session Status  ```http GET /predict/session/{session_id}/status ```  **Success (200 OK):** ```json {   "session_id"...
- **Action**: Verify if endpoint was removed or renamed

#### `POST /predict/session/{session_id}/process_all`
- **Source**: predict-api.md
- **Blueprint**: predict
- **Context**: ated_elements_count": 2,     "can_undo": true   },   "success": true } ```  #### Process All Steps  ```http POST /predict/session/{session_id}/process_all?max_steps=100 ```  **Success (200 OK):** ```j...
- **Action**: Verify if endpoint was removed or renamed

#### `GET /mesh/info/{mesh_name}`
- **Source**: mesh-api.md
- **Blueprint**: mesh
- **Context**: `  ---  ### Get Mesh Info  Retrieve detailed information about a specific mesh file.  #### Request  ```http GET /mesh/info/{mesh_name}?subfolder=mesh ```  **Path Parameters:**  | Parameter | Type | De...
- **Action**: Verify if endpoint was removed or renamed

#### `GET /mesh/boundary/{mesh_name}`
- **Source**: mesh-api.md
- **Blueprint**: mesh
- **Context**: " } ```  ---  ### Get Mesh Boundary  Retrieve boundary vertices for a specific mesh.  #### Request  ```http GET /mesh/boundary/{mesh_name}?subfolder=mesh ```  **Path Parameters:**  | Parameter | Type ...
- **Action**: Verify if endpoint was removed or renamed

#### `POST /predict/session/{session_id}/prev`
- **Source**: predict-api.md
- **Blueprint**: predict
- **Context**: "is_completed": false,     "can_undo": true   },   "success": true } ```  #### Undo Previous Step  ```http POST /predict/session/{session_id}/prev ```  **Success (200 OK):** ```json {   "session_id": ...
- **Action**: Verify if endpoint was removed or renamed

#### `POST /training/history/episode/{training_id}/{episode_index}`
- **Source**: training-history-api.md
- **Blueprint**: training_history
- **Context**: isode Data  Retrieve detailed data for a specific episode within a training session.  #### Request  ```http POST /training/history/episode/{training_id}/{episode_index} ```  **Path Parameters:**  | Pa...
- **Action**: Verify if endpoint was removed or renamed

#### `GET /action/find-ref-point/{mesh_name}`
- **Source**: quality-action-apis.md
- **Blueprint**: action
- **Context**: 1:5000/action`  ### Find Reference Point  Find the reference point for a given mesh.  #### Request  ```http GET /action/find-ref-point/{mesh_name} ```  #### Response  **Success (200 OK):** ```json {  ...
- **Action**: Verify if endpoint was removed or renamed

#### `DELETE /predict/session/{session_id}`
- **Source**: predict-api.md
- **Blueprint**: predict
- **Context**: ictor_type": "RL",     "ref_selector_type": "RL"   },   "success": true } ```  #### Delete Session  ```http DELETE /predict/session/{session_id} ```  **Success (200 OK):** ```json {   "message": "Sess...
- **Action**: Verify if endpoint was removed or renamed

#### `POST /training/history/info/{training_id}`
- **Source**: training-history-api.md
- **Blueprint**: training_history
- **Context**: ### Get Training Info  Retrieve basic information about a specific training session.  #### Request  ```http POST /training/history/info/{training_id} ```  **Path Parameters:**  | Parameter | Type | De...
- **Action**: Verify if endpoint was removed or renamed

## 🔄 Potential Renames/Changes

These appear to be potential renames or changes:

#### Potential Match Found
- **Discovered**: `GET /predict/session/<session_id>/status`
- **Documented**: `GET /predict/session/{session_id}/status`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /predict/session/<session_id>/reference_point`
- **Documented**: `GET /predict/session/{session_id}/status`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /action/find-ref-point/<mesh_name>`
- **Documented**: `GET /action/find-ref-point/{mesh_name}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: action
- **Doc Source**: quality-action-apis.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /predict/session/<session_id>/history`
- **Documented**: `GET /predict/session/{session_id}/status`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /predict/session/<session_id>/quality`
- **Documented**: `GET /predict/session/{session_id}/status`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/next`
- **Documented**: `POST /predict/session/{session_id}/next`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/next`
- **Documented**: `POST /predict/session/{session_id}/process_all`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/next`
- **Documented**: `POST /predict/session/{session_id}/prev`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /mesh/boundary/<n>`
- **Documented**: `GET /mesh/boundary/{mesh_name}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: mesh
- **Doc Source**: mesh-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /training/history/info/<training_id>`
- **Documented**: `POST /training/history/episode/{training_id}/{episode_index}`
- **Reason**: Same prefix: /training/history
- **Blueprint**: training_history
- **Doc Source**: training-history-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /training/history/info/<training_id>`
- **Documented**: `POST /training/history/info/{training_id}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: training_history
- **Doc Source**: training-history-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `GET /mesh/info/<n>`
- **Documented**: `GET /mesh/info/{mesh_name}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: mesh
- **Doc Source**: mesh-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /training/history/episode/<training_id>/<int:episode_index>`
- **Documented**: `POST /training/history/episode/{training_id}/{episode_index}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: training_history
- **Doc Source**: training-history-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /training/history/episode/<training_id>/<int:episode_index>`
- **Documented**: `POST /training/history/info/{training_id}`
- **Reason**: Same prefix: /training/history
- **Blueprint**: training_history
- **Doc Source**: training-history-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/reset`
- **Documented**: `POST /predict/session/{session_id}/next`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/reset`
- **Documented**: `POST /predict/session/{session_id}/process_all`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/reset`
- **Documented**: `POST /predict/session/{session_id}/prev`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/process_all`
- **Documented**: `POST /predict/session/{session_id}/next`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/process_all`
- **Documented**: `POST /predict/session/{session_id}/process_all`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/process_all`
- **Documented**: `POST /predict/session/{session_id}/prev`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/prev`
- **Documented**: `POST /predict/session/{session_id}/next`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/prev`
- **Documented**: `POST /predict/session/{session_id}/process_all`
- **Reason**: Same prefix: /predict/session
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `POST /predict/session/<session_id>/prev`
- **Documented**: `POST /predict/session/{session_id}/prev`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

#### Potential Match Found
- **Discovered**: `DELETE /predict/session/<session_id>`
- **Documented**: `DELETE /predict/session/{session_id}`
- **Reason**: Same normalized path (different parameter formats)
- **Blueprint**: predict
- **Doc Source**: predict-api.md
- **Action**: Verify if this is the same endpoint with different naming

## ✅ Successfully Matched Endpoints

**Total Matched**: 14

<details>
<summary>Click to expand full list</summary>

- `GET /mesh/health` - Mesh service health check
- `GET /mesh/list` - Retrieve a list of all available mesh files
- `GET /predict/components` - Get available predictors, reference selectors, models, and meshes
- `GET /quality/methods` - Get all available quality calculation methods
- `GET /training/health` - Training service health check
- `GET /training/history/health` - Training history service health check
- `GET /training/history/list` - List all available training history sessions
- `GET /training/status` - Get current training status with real-time statistics
- `POST /action/execute` - Execute and validate a specific action
- `POST /geometry/normalize` - Convert coordinates to normalized polar coordinates
- `POST /predict/session/create` - Create a new prediction session
- `POST /quality/calculate` - Calculate quality score for a given quadrilateral
- `POST /training/start` - Start a new training session with specified parameters
- `POST /training/stop` - Stop the currently running training session

</details>

## 📋 Action Items & Recommendations

### High Priority

1. **Document 32 missing endpoints** - These are implemented but not documented
2. **Verify 24 potential renames** - Check if these are the same endpoints with different naming
3. **Review 10 extra documented endpoints** - Verify if these were removed or renamed

### Documentation Quality Improvements

- Ensure all endpoint descriptions are clear and complete
- Add request/response examples for complex endpoints
- Include error handling documentation
- Keep parameter specifications up to date
- Add integration examples for frontend developers

### Process Improvements

- Set up automated endpoint discovery to catch changes early
- Implement documentation review process for new endpoints
- Consider generating OpenAPI/Swagger specs from code
- Regular documentation audits (recommended monthly)

---

*Report generated by endpoint comparison tool - 502 lines*