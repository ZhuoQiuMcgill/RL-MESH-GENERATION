# Predict API Documentation

## Overview

The Predict API provides endpoints for mesh generation prediction using trained RL models. It supports session-based prediction with reversible operations, allowing users to step through the mesh generation process with different predictors and reference point selectors.

## Base URL

All endpoints are prefixed with `/predict`

## Endpoints

### 1. List Available Components

**GET** `/predict/components`

Lists all available components for prediction including predictors, reference selectors, initial meshes, and trained models.

**Response:**
```json
{
  "predictors": {
    "RL": {
      "name": "RL",
      "class": "RLPredictor",
      "description": "Reinforcement Learning predictor using trained SAC model",
      "parameters": ["n", "g", "beta"]
    }
  },
  "reference_selectors": {
    "RL": {
      "name": "RL",
      "description": "RL-based reference point selector (minimum interior angle)",
      "parameters": ["n"]
    },
    "default": {
      "name": "default",
      "description": "Default boundary reference point selector",
      "parameters": []
    }
  },
  "initial_meshes": ["basic1.txt", "basic2.txt", "dolphine3.txt"],
  "trained_models": [
    {
      "name": "basic1-reward68.026.zip",
      "path": "data/models/basic1-reward68.026.zip",
      "size": 1234567,
      "description": "Trained SAC model: basic1-reward68.026.zip"
    }
  ],
  "success": true
}
```

### 2. Create Prediction Session

**POST** `/predict/session/create`

Creates a new prediction session with specified configuration.

**Request Body:**
```json
{
  "mesh_name": "basic1.txt",
  "predictor_type": "RL",
  "predictor_config": {
    "model_path": "data/models/basic1-reward68.026.zip",
    "n": 2,
    "g": 3,
    "beta": 6
  },
  "ref_selector_type": "RL",
  "ref_selector_config": {
    "n": 2
  }
}
```

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "initial_status": {
    "current_step": 0,
    "boundary_size": 5,
    "generated_elements_count": 0,
    "is_completed": false,
    "active_predictor": "RL",
    "can_undo": false,
    "total_steps_possible": 1
  },
  "config": { /* session configuration */ },
  "success": true
}
```

### 3. Get Session Status

**GET** `/predict/session/{session_id}/status`

Gets the current status of a prediction session.

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "status": {
    "current_step": 0,
    "boundary_size": 5,
    "generated_elements_count": 0,
    "is_completed": false,
    "active_predictor": "RL",
    "can_undo": false
  },
  "config": { /* session configuration */ },
  "success": true
}
```

### 4. Update Session Configuration

**PUT** `/predict/session/{session_id}/config`

Updates the predictor or reference selector for an existing session.

**Request Body:**
```json
{
  "predictor_type": "RL",
  "predictor_config": {
    "model_path": "data/models/new_model.zip",
    "n": 2,
    "g": 3,
    "beta": 6
  },
  "ref_selector_type": "default"
}
```

### 5. Execute Next Step

**POST** `/predict/session/{session_id}/next`

Executes the next prediction step in the session.

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "step_result": {
    "success": true,
    "element": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "message": "Successfully executed action type1"
  },
  "status": {
    "current_step": 1,
    "boundary_size": 4,
    "generated_elements_count": 1,
    "is_completed": false
  },
  "success": true
}
```

### 6. Undo Previous Step

**POST** `/predict/session/{session_id}/prev`

Undoes the previous step and returns to the previous state.

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "undo_result": {
    "success": true,
    "message": "Successfully undone step 1"
  },
  "status": {
    "current_step": 0,
    "boundary_size": 5,
    "generated_elements_count": 0,
    "can_undo": false
  },
  "success": true
}
```

### 7. Process All Steps

**POST** `/predict/session/{session_id}/process_all?max_steps=100`

Executes all steps until invalid action or completion.

**Query Parameters:**
- `max_steps` (optional, default: 100): Maximum number of steps to execute

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "steps_executed": 3,
  "results": [
    {
      "success": true,
      "element": [...],
      "message": "Successfully executed action type0_left"
    },
    {
      "success": true,
      "element": [...],
      "message": "Successfully executed action type1"
    },
    {
      "success": false,
      "element": null,
      "message": "Invalid action type1 at reference 2"
    }
  ],
  "final_status": {
    "current_step": 3,
    "boundary_size": 4,
    "generated_elements_count": 2,
    "is_completed": false
  },
  "success": true
}
```

### 8. Reset Session

**POST** `/predict/session/{session_id}/reset`

Resets the session to its initial state.

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "reset_result": {
    "success": true,
    "message": "Successfully reset to initial state"
  },
  "status": {
    "current_step": 0,
    "boundary_size": 5,
    "generated_elements_count": 0,
    "is_completed": false
  },
  "success": true
}
```

### 9. Get Session History

**GET** `/predict/session/{session_id}/history`

Gets the complete history of actions performed in the session.

**Response:**
```json
{
  "session_id": "session_0_123456789",
  "history": [
    {
      "action": "next",
      "result": { /* step result */ },
      "timestamp": 1640995200.0
    },
    {
      "action": "prev",
      "result": { /* undo result */ },
      "timestamp": 1640995260.0
    }
  ],
  "total_actions": 2,
  "success": true
}
```

### 10. Delete Session

**DELETE** `/predict/session/{session_id}`

Deletes a prediction session and frees up resources.

**Response:**
```json
{
  "message": "Session session_0_123456789 deleted successfully",
  "success": true
}
```

### 11. List All Sessions

**GET** `/predict/sessions`

Lists all active prediction sessions.

**Response:**
```json
{
  "sessions": {
    "session_0_123456789": {
      "config": { /* session config */ },
      "status": { /* current status */ },
      "history_length": 5
    }
  },
  "total_sessions": 1,
  "success": true
}
```

### 12. Health Check

**GET** `/predict/health`

Health check endpoint for the predict API.

**Response:**
```json
{
  "status": "healthy",
  "service": "predict-api",
  "active_sessions": 2,
  "timestamp": 1640995200.0
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error description",
  "success": false
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (session not found)
- `500`: Internal Server Error

## Usage Flow

1. **GET** `/predict/components` - List available components
2. **POST** `/predict/session/create` - Create session with desired configuration
3. Use one of the following approaches:
   - **POST** `/predict/session/{id}/next` - Step-by-step execution
   - **POST** `/predict/session/{id}/process_all` - Execute all at once
4. Optionally:
   - **POST** `/predict/session/{id}/prev` - Undo steps
   - **PUT** `/predict/session/{id}/config` - Change predictor/selector
   - **POST** `/predict/session/{id}/reset` - Reset to initial state
5. **DELETE** `/predict/session/{id}` - Clean up when done

## Notes

- Sessions are stored in memory and will be lost on server restart
- Each session maintains its own MeshGenerator instance with command history
- Configuration changes (predictor/selector) can be made at any time
- The API supports concurrent sessions with different configurations