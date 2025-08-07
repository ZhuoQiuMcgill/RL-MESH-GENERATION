# Training Management API

> **Status**: `Official`  
> **Version**: v2.0.0  
> **Maintainer**: @ZhuoQiuMcgill  
> **Last Updated**: 2025-01-07  
> **Blueprint**: `training`  
> **URL Prefix**: `/training`

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [Start Training](#start-training)
  - [Stop Training](#stop-training)
  - [Get Training Status](#get-training-status)
  - [Health Check](#health-check)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Frontend Integration Guide](#frontend-integration-guide)

## Overview

The Training Management API provides comprehensive control over reinforcement learning training sessions for mesh generation. It supports starting/stopping training, real-time status monitoring, and checkpoint-based resumption.

**Key Features:**
- **Session Management**: Start and stop training sessions
- **Real-time Monitoring**: Live training statistics and progress tracking
- **Checkpoint Support**: Resume training from saved checkpoints
- **Flexible Configuration**: Customizable training parameters

## Base URL

```
http://127.0.0.1:5000/training
```

---

## Endpoints

### Start Training

Start a new training session with specified parameters.

#### Request

```http
POST /training/start
Content-Type: application/json
```

**Body Parameters:**
```json
{
  "mesh_name": "simple_square",
  "subfolder": "mesh",
  "max_timesteps": 1000000,
  "max_steps": 1000,
  "description": "Training on simple square mesh",
  "checkpoint_name": "checkpoint1",
  "from_checkpoint": false
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| **mesh_name** | string | No | null | Name of mesh file to use (without .txt) |
| **subfolder** | string | No | "mesh" | Subfolder containing mesh file |
| **max_timesteps** | integer | No | null | Maximum training timesteps |
| **max_steps** | integer | No | null | Maximum steps per episode |
| **description** | string | No | null | Training session description |
| **checkpoint_name** | string | No | null | Checkpoint name if resuming |
| **from_checkpoint** | boolean | No | false | Whether to resume from checkpoint |

#### Response

**Success (200 OK):**
```json
{
  "message": "training_started",
  "success": true,
  "config": {
    "mesh_name": "simple_square",
    "subfolder": "mesh",
    "max_timesteps": 1000000,
    "max_steps": 1000,
    "description": "Training on simple square mesh"
  }
}
```

**Error (400 Bad Request):**
```json
{
  "error": "Training already running",
  "success": false
}
```

---

### Stop Training

Stop the currently running training session.

#### Request

```http
POST /training/stop
```

#### Response

**Success (200 OK):**
```json
{
  "message": "stop_requested",
  "success": true
}
```

**Error (500 Internal Server Error):**
```json
{
  "error": "Failed to stop training: Connection error",
  "success": false
}
```

---

### Get Training Status

Retrieve comprehensive real-time training status.

#### Request

```http
GET /training/status
```

#### Response

**Training Active (200 OK):**
```json
{
  "running": true,
  "status": "running",
  "stats": {
    "episode": 150,
    "total_steps": 15000,
    "episode_reward": 0.842,
    "average_reward": 0.756,
    "episode_length": 98,
    "boundary_vertices": 12,
    "buffer_size": 5000,
    "training_id": "train_20250107_143022_simple_square",
    "online_learning_mode": false,
    "recent_actor_loss": 0.003421,
    "recent_critic_loss": 0.005123,
    "current_alpha": 0.2
  },
  "progress": {
    "current_episode": 150,
    "total_steps": 15000,
    "latest_reward": 0.842,
    "average_reward": 0.756,
    "buffer_utilization": 5000
  },
  "timestamp": 1641888000.123
}
```

**Training Idle (200 OK):**
```json
{
  "running": false,
  "status": "idle",
  "stats": null,
  "progress": null,
  "timestamp": 1641888000.123
}
```

---

### Health Check

Check the health status of the training service.

#### Request

```http
GET /training/health
```

#### Response

**Success (200 OK):**
```json
{
  "status": "healthy",
  "service": "training-api",
  "manager_running": false,
  "timestamp": 1641888000.123
}
```

---

## Data Models

### Training Status Model

```typescript
interface TrainingStatus {
  running: boolean;
  status: "running" | "idle" | "error";
  stats: TrainingStats | null;
  progress: TrainingProgress | null;
  timestamp: number;
  error?: string;
}
```

### Training Stats Model

```typescript
interface TrainingStats {
  episode: number;
  total_steps: number;
  episode_reward: number;
  average_reward: number;
  episode_length: number;
  boundary_vertices: number;
  buffer_size: number;
  training_id: string;
  online_learning_mode: boolean;
  recent_actor_loss?: number;
  recent_critic_loss?: number;
  current_alpha?: number;
}
```

### Training Progress Model

```typescript
interface TrainingProgress {
  current_episode: number;
  total_steps: number;
  latest_reward: number;
  average_reward: number;
  buffer_utilization: number;
}
```

---

## Error Handling

| HTTP Code | Description | Cause |
|-----------|-------------|-------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid parameters or training already running |
| 500 | Internal Server Error | Server-side error during processing |

### Common Error Response Format

```json
{
  "error": "Error description",
  "success": false
}
```

---

## Frontend Integration Guide

### Real-time Status Polling

```javascript
class TrainingMonitor {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.pollingInterval = null;
  }

  async startTraining(config) {
    const response = await fetch(`${this.apiBaseUrl}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  }

  async getStatus() {
    const response = await fetch(`${this.apiBaseUrl}/training/status`);
    return response.json();
  }

  startPolling(callback, intervalMs = 10000) {
    this.pollingInterval = setInterval(async () => {
      try {
        const status = await this.getStatus();
        callback(status);
      } catch (error) {
        console.error('Status polling failed:', error);
      }
    }, intervalMs);
  }

  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }
}
```

### React Component Example

```jsx
import { useState, useEffect, useCallback } from 'react';

export function TrainingDashboard() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/training/status');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Poll every 10 seconds
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const startTraining = async (config) => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const result = await response.json();
      if (result.success) {
        fetchStatus(); // Refresh status
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="training-dashboard">
      <h1>Training Dashboard</h1>
      {status?.running ? (
        <div className="training-active">
          <h2>Training Active</h2>
          <p>Episode: {status.stats?.episode}</p>
          <p>Reward: {status.stats?.episode_reward?.toFixed(3)}</p>
          <p>Steps: {status.stats?.total_steps}</p>
        </div>
      ) : (
        <div className="training-idle">
          <h2>Training Idle</h2>
          <button onClick={() => startTraining({mesh_name: 'simple_square'})}>
            Start Training
          </button>
        </div>
      )}
    </div>
  );
}
```

### Best Practices

1. **Polling Frequency**: Poll status every 10-30 seconds during active training
2. **Error Handling**: Always handle network errors gracefully
3. **Loading States**: Show loading indicators during API calls
4. **Real-time Updates**: Use WebSockets for more responsive updates if needed
5. **Memory Management**: Clear intervals when components unmount

### Configuration Examples

**Basic Training:**
```json
{
  "mesh_name": "simple_square",
  "max_timesteps": 100000,
  "description": "Basic training session"
}
```

**Advanced Training:**
```json
{
  "mesh_name": "complex_mesh",
  "subfolder": "custom",
  "max_timesteps": 1000000,
  "max_steps": 2000,
  "description": "Advanced training with custom parameters"
}
```

**Checkpoint Resume:**
```json
{
  "checkpoint_name": "checkpoint1",
  "from_checkpoint": true,
  "description": "Resume from saved checkpoint"
}
```
