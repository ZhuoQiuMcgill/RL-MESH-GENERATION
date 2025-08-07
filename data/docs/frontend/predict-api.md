# Prediction API

> **Status**: `Official`  
> **Version**: v2.0.0  
> **Maintainer**: @ZhuoQiuMcgill  
> **Last Updated**: 2025-01-07  
> **Blueprint**: `predict`  
> **URL Prefix**: `/predict`

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [List Components](#list-components)
  - [Create Session](#create-session)
  - [Session Management](#session-management)
  - [Execute Steps](#execute-steps)
  - [Session Monitoring](#session-monitoring)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Frontend Integration Guide](#frontend-integration-guide)

## Overview

The Prediction API provides mesh generation prediction using trained reinforcement learning models. It supports session-based prediction with step-by-step mesh generation for interactive visualization.

**Key Features:**
- **Session-based Prediction**: Manage multiple prediction sessions
- **Step-by-step Generation**: Execute individual mesh generation steps
- **RL Model Integration**: Use trained SAC models for prediction
- **Interactive Visualization**: Support for real-time mesh generation display

## Base URL

```
http://127.0.0.1:5000/predict
```

---

## Endpoints

### List Components

Get available predictors, reference selectors, models, and meshes.

#### Request

```http
GET /predict/components
```

#### Response

**Success (200 OK):**
```json
{
  "predictors": {
    "RL": {
      "name": "RL",
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
      "size": 1621539,
      "description": "Trained SAC model: basic1-reward68.026.zip"
    }
  ],
  "success": true
}
```

---

### Create Session

Create a new prediction session with specified configuration.

#### Request

```http
POST /predict/session/create
Content-Type: application/json
```

**Body Parameters:**
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

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| **mesh_name** | string | Yes | - | Initial mesh file name |
| **predictor_type** | string | Yes | - | Predictor type ("RL") |
| **predictor_config** | object | Yes | - | Predictor configuration |
| **predictor_config.model_path** | string | Yes | - | Path to trained SAC model |
| **predictor_config.n** | integer | No | 2 | Number of neighbor vertices |
| **predictor_config.g** | integer | No | 3 | Number of observation points |
| **predictor_config.beta** | integer | No | 6 | State observation radius factor |
| ref_selector_type | string | No | "default" | Reference point selector type |
| ref_selector_config | object | No | {} | Reference selector configuration |

#### Response

**Success (200 OK):**
```json
{
  "session_id": "session_0_123456789",
  "initial_status": {
    "current_step": 0,
    "boundary_size": 38,
    "generated_elements_count": 0,
    "is_completed": false,
    "active_predictor": "RL",
    "can_undo": false,
    "total_steps_possible": 34
  },
  "config": {
    "mesh_name": "basic1.txt",
    "predictor_type": "RL",
    "predictor_config": {
      "model_path": "data/models/basic1-reward68.026.zip",
      "n": 2,
      "g": 3,
      "beta": 6
    },
    "ref_selector_type": "RL",
    "ref_selector_config": {"n": 2}
  },
  "success": true
}
```

---

### Session Management

#### Get Session Status

```http
GET /predict/session/{session_id}/status
```

**Success (200 OK):**
```json
{
  "session_id": "session_0_123456789",
  "status": {
    "current_step": 5,
    "boundary_size": 33,
    "generated_elements_count": 5,
    "is_completed": false,
    "active_predictor": "RL",
    "can_undo": true,
    "total_steps_possible": 29
  },
  "config": {
    "mesh_name": "basic1.txt",
    "predictor_type": "RL",
    "ref_selector_type": "RL"
  },
  "success": true
}
```

#### Delete Session

```http
DELETE /predict/session/{session_id}
```

**Success (200 OK):**
```json
{
  "message": "Session session_0_123456789 deleted successfully",
  "success": true
}
```

---

### Execute Steps

#### Execute Next Step

```http
POST /predict/session/{session_id}/next
```

**Success (200 OK):**
```json
{
  "session_id": "session_0_123456789",
  "step_result": {
    "success": true,
    "element": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "message": "Successfully executed action type1",
    "action_info": {
      "action_type": "type1",
      "reference_vertex_idx": 15,
      "new_coords": [[0.5, 0.5]],
      "is_valid": true,
      "validation_message": null
    }
  },
  "status": {
    "current_step": 1,
    "boundary_size": 37,
    "generated_elements_count": 1,
    "is_completed": false,
    "can_undo": true
  },
  "success": true
}
```

#### Undo Previous Step

```http
POST /predict/session/{session_id}/prev
```

**Success (200 OK):**
```json
{
  "session_id": "session_0_123456789",
  "undo_result": {
    "success": true,
    "message": "Successfully undone step 3"
  },
  "status": {
    "current_step": 2,
    "boundary_size": 36,
    "generated_elements_count": 2,
    "can_undo": true
  },
  "success": true
}
```

#### Process All Steps

```http
POST /predict/session/{session_id}/process_all?max_steps=100
```

**Success (200 OK):**
```json
{
  "session_id": "session_0_123456789",
  "steps_executed": 15,
  "results": [
    {
      "success": true,
      "element": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
      "message": "Successfully executed action type0_left",
      "action_info": {
        "action_type": "type0_left",
        "reference_vertex_idx": 5,
        "new_coords": null,
        "is_valid": true,
        "validation_message": null
      }
    }
  ],
  "final_status": {
    "current_step": 15,
    "boundary_size": 4,
    "generated_elements_count": 14,
    "is_completed": true
  },
  "success": true
}
```

---

## Data Models

### Prediction Session Status

```typescript
interface SessionStatus {
  current_step: number;
  boundary_size: number;
  generated_elements_count: number;
  is_completed: boolean;
  active_predictor: string;
  can_undo: boolean;
  total_steps_possible: number;
}
```

### Step Result

```typescript
interface StepResult {
  success: boolean;
  element: [number, number][] | null;
  message: string;
  action_info: {
    action_type: "type0_left" | "type0_right" | "type1";
    reference_vertex_idx: number;
    new_coords: [number, number][] | null;
    is_valid: boolean;
    validation_message: string | null;
  };
}
```

### Model Info

```typescript
interface ModelInfo {
  name: string;
  path: string;
  size: number;
  description: string;
}
```

---

## Error Handling

| HTTP Code | Description | Cause |
|-----------|-------------|-------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Missing required fields or invalid parameters |
| 404 | Not Found | Session or resource not found |
| 500 | Internal Server Error | Server-side error during processing |

---

## Frontend Integration Guide

### Prediction Session Manager

```javascript
class PredictionSessionManager {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.sessions = new Map();
  }

  async getComponents() {
    const response = await fetch(`${this.apiBaseUrl}/predict/components`);
    return response.json();
  }

  async createSession(config) {
    const response = await fetch(`${this.apiBaseUrl}/predict/session/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    const data = await response.json();
    if (data.success) {
      this.sessions.set(data.session_id, data);
    }
    return data;
  }

  async executeStep(sessionId) {
    const response = await fetch(
      `${this.apiBaseUrl}/predict/session/${sessionId}/next`,
      { method: 'POST' }
    );
    return response.json();
  }

  async undoStep(sessionId) {
    const response = await fetch(
      `${this.apiBaseUrl}/predict/session/${sessionId}/prev`,
      { method: 'POST' }
    );
    return response.json();
  }

  async processAllSteps(sessionId, maxSteps = 100) {
    const response = await fetch(
      `${this.apiBaseUrl}/predict/session/${sessionId}/process_all?max_steps=${maxSteps}`,
      { method: 'POST' }
    );
    return response.json();
  }

  async getSessionStatus(sessionId) {
    const response = await fetch(
      `${this.apiBaseUrl}/predict/session/${sessionId}/status`
    );
    return response.json();
  }

  async deleteSession(sessionId) {
    const response = await fetch(
      `${this.apiBaseUrl}/predict/session/${sessionId}`,
      { method: 'DELETE' }
    );
    
    if (response.ok) {
      this.sessions.delete(sessionId);
    }
    return response.json();
  }
}
```

### React Prediction Component

```jsx
import { useState, useEffect, useCallback } from 'react';

export function PredictionDashboard() {
  const [sessionManager] = useState(() => new PredictionSessionManager());
  const [components, setComponents] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);
  const [generatedElements, setGeneratedElements] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    loadComponents();
  }, []);

  const loadComponents = async () => {
    try {
      const data = await sessionManager.getComponents();
      setComponents(data);
    } catch (error) {
      console.error('Failed to load components:', error);
    }
  };

  const createSession = async (config) => {
    try {
      const session = await sessionManager.createSession(config);
      if (session.success) {
        setCurrentSession(session);
        setGeneratedElements([]);
      }
      return session;
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const executeNextStep = useCallback(async () => {
    if (!currentSession || isProcessing) return;

    setIsProcessing(true);
    try {
      const result = await sessionManager.executeStep(currentSession.session_id);
      
      if (result.success && result.step_result.success) {
        setGeneratedElements(prev => [...prev, result.step_result.element]);
        
        // Update session status
        const status = await sessionManager.getSessionStatus(currentSession.session_id);
        setCurrentSession(prev => ({ ...prev, ...status }));
      }
      
      return result;
    } catch (error) {
      console.error('Failed to execute step:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [currentSession, isProcessing]);

  const processAllSteps = async () => {
    if (!currentSession || isProcessing) return;

    setIsProcessing(true);
    try {
      const result = await sessionManager.processAllSteps(currentSession.session_id);
      
      if (result.success) {
        const validElements = result.results
          .filter(r => r.success && r.element)
          .map(r => r.element);
        
        setGeneratedElements(validElements);
        
        // Update session status
        const status = await sessionManager.getSessionStatus(currentSession.session_id);
        setCurrentSession(prev => ({ ...prev, ...status }));
      }
      
      return result;
    } catch (error) {
      console.error('Failed to process all steps:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const undoStep = async () => {
    if (!currentSession || isProcessing) return;

    setIsProcessing(true);
    try {
      const result = await sessionManager.undoStep(currentSession.session_id);
      
      if (result.success && result.undo_result.success) {
        setGeneratedElements(prev => prev.slice(0, -1));
        
        // Update session status
        const status = await sessionManager.getSessionStatus(currentSession.session_id);
        setCurrentSession(prev => ({ ...prev, ...status }));
      }
      
      return result;
    } catch (error) {
      console.error('Failed to undo step:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="prediction-dashboard">
      <h1>Mesh Generation Prediction</h1>
      
      {!currentSession ? (
        <div className="session-creator">
          <h2>Create Prediction Session</h2>
          {components && (
            <SessionCreator 
              components={components}
              onCreateSession={createSession}
            />
          )}
        </div>
      ) : (
        <div className="session-controls">
          <h2>Session: {currentSession.session_id}</h2>
          <div className="status">
            <p>Step: {currentSession.status?.current_step || 0}</p>
            <p>Elements: {generatedElements.length}</p>
            <p>Boundary Size: {currentSession.status?.boundary_size}</p>
            <p>Completed: {currentSession.status?.is_completed ? 'Yes' : 'No'}</p>
          </div>
          
          <div className="controls">
            <button 
              onClick={executeNextStep}
              disabled={isProcessing || currentSession.status?.is_completed}
            >
              Next Step
            </button>
            <button 
              onClick={undoStep}
              disabled={isProcessing || !currentSession.status?.can_undo}
            >
              Undo
            </button>
            <button 
              onClick={processAllSteps}
              disabled={isProcessing || currentSession.status?.is_completed}
            >
              Process All
            </button>
          </div>
          
          <MeshVisualization 
            elements={generatedElements}
            boundary={currentSession.initialBoundary}
          />
        </div>
      )}
    </div>
  );
}
```

### Best Practices

1. **Session Management**: Always clean up sessions when done
2. **Error Handling**: Handle prediction failures gracefully
3. **UI Feedback**: Show loading states during processing
4. **Step Validation**: Check if steps can be executed/undone
5. **Real-time Updates**: Update UI immediately after each step

### Configuration Examples

**Basic RL Predictor:**
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
