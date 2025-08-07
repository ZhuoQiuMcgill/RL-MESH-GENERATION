# RL Mesh Generation API Documentation

> **Project**: RL-MESH-GENERATION  
> **Version**: v2.0.0  
> **Last Updated**: 2025-01-07  
> **Maintainer**: @ZhuoQiuMcgill

## Overview

This documentation provides comprehensive API references and module guides for the RL Mesh Generation system. The documentation is organized into two main sections: **Frontend APIs** for client-side integration and **Backend Modules** for server-side development.

## Documentation Structure

```
data/docs/
├── README.md                           # This file
├── frontend/                           # Frontend API Documentation
│   ├── training-api.md                 # Training Management API
│   ├── mesh-api.md                     # Mesh Management API
│   ├── predict-api.md                  # Prediction API
│   ├── training-history-api.md         # Training History API
│   └── quality-action-apis.md          # Quality & Action APIs
└── backend/                            # Backend Module Documentation
    └── training-manager.md             # Training Manager Module
```

---

## Frontend API Documentation

These documents are designed for frontend developers building user interfaces and client applications that interact with the RL Mesh Generation system.

### 🚀 [Training Management API](./frontend/training-api.md)
**Blueprint:** `training` | **URL Prefix:** `/training`

Comprehensive control over reinforcement learning training sessions. Supports starting/stopping training, real-time status monitoring, and checkpoint-based resumption.

**Key Features:**
- Session Management (start/stop training)
- Real-time Monitoring (live statistics and progress)
- Checkpoint Support (resume from saved models)
- Flexible Configuration (customizable parameters)

**Quick Start:**
```javascript
const response = await fetch('http://127.0.0.1:5000/training/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    mesh_name: 'simple_square',
    max_timesteps: 100000,
    description: 'Basic training session'
  })
});
```

---

### 📐 [Mesh Management API](./frontend/mesh-api.md)
**Blueprint:** `mesh` | **URL Prefix:** `/mesh`

Access to mesh files and boundary data for training and visualization. Handles mesh file discovery, metadata retrieval, and boundary vertex extraction.

**Key Features:**
- Mesh Discovery (list available files)
- Metadata Access (file info and statistics)
- Boundary Extraction (vertex coordinates)
- File Validation (existence and validity checks)

**Quick Start:**
```javascript
// List available meshes
const meshes = await fetch('http://127.0.0.1:5000/mesh/list').then(r => r.json());

// Get boundary vertices for visualization
const boundary = await fetch('http://127.0.0.1:5000/mesh/boundary/simple_square')
  .then(r => r.json());
```

---

### 🎯 [Prediction API](./frontend/predict-api.md)
**Blueprint:** `predict` | **URL Prefix:** `/predict`

Mesh generation prediction using trained reinforcement learning models. Supports session-based prediction with step-by-step mesh generation for interactive visualization.

**Key Features:**
- Session-based Prediction (manage multiple sessions)
- Step-by-step Generation (individual mesh generation steps)
- RL Model Integration (trained SAC models)
- Interactive Visualization (real-time generation display)

**Quick Start:**
```javascript
// Create prediction session
const session = await fetch('http://127.0.0.1:5000/predict/session/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    mesh_name: 'basic1.txt',
    predictor_type: 'RL',
    predictor_config: {
      model_path: 'data/models/basic1-reward68.026.zip'
    }
  })
});

// Execute next step
const step = await fetch(`http://127.0.0.1:5000/predict/session/${sessionId}/next`, {
  method: 'POST'
});
```

---

### 📊 [Training History API](./frontend/training-history-api.md)
**Blueprint:** `training_history` | **URL Prefix:** `/training/history`

Access to historical training session data. Allows querying training session metadata, episode details, and performance statistics for analysis and visualization.

**Key Features:**
- Session Discovery (list available sessions)
- Episode Retrieval (detailed episode data)
- Performance Analysis (statistics and metrics)
- Historical Insights (compare runs and track progress)

**Quick Start:**
```javascript
// Get all training sessions
const sessions = await fetch('http://127.0.0.1:5000/training/history/list')
  .then(r => r.json());

// Get best episode from a session
const info = await fetch(`http://127.0.0.1:5000/training/history/info/${trainingId}`, {
  method: 'POST'
}).then(r => r.json());

const episode = await fetch(
  `http://127.0.0.1:5000/training/history/episode/${trainingId}/${info.best_episode}`,
  { method: 'POST' }
).then(r => r.json());
```

---

### ⚙️ [Quality & Action APIs](./frontend/quality-action-apis.md)
**Blueprints:** `quality`, `action`, `geometry`

Essential APIs for mesh quality analysis, action testing, and coordinate processing in the mesh generation system.

**Key Features:**
- Quality Calculation (multiple quality metrics)
- Action Testing (validate mesh generation actions)
- Coordinate Processing (normalize and transform coordinates)
- Interactive Testing (step-by-step validation)

**Quick Start:**
```javascript
// Calculate element quality
const quality = await fetch('http://127.0.0.1:5000/quality/calculate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    vertices: [[0, 0], [1, 0], [1, 1], [0, 1]],
    method: 'robust'
  })
});

// Test action validity
const actionResult = await fetch('http://127.0.0.1:5000/action/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    mesh_name: 'simple_square',
    action_type: 'type1',
    reference_point_index: 10,
    clicked_point: [0.5, 0.5]
  })
});
```

---

## Backend Module Documentation

These documents are designed for backend developers working on the server-side implementation and extending the system functionality.

### 🏗️ [Training Manager Module](./backend/training-manager.md)
**Module:** `src.ui.training_manager`

High-level abstraction for managing reinforcement learning training sessions. Handles the lifecycle of training processes, from initialization and execution to monitoring and cleanup.

**Key Responsibilities:**
- Process Management (start/stop training processes)
- Status Monitoring (real-time training status tracking)
- Configuration Management (training parameters and settings)
- Resource Management (process cleanup and resource allocation)
- Error Recovery (handle training failures and edge cases)

**Quick Start:**
```python
from src.ui.training_manager import get_training_manager

# Get singleton manager instance
manager = get_training_manager()

# Start training
config = {
    "mesh_name": "simple_square",
    "max_timesteps": 100000,
    "description": "Test training session"
}
result = manager.start_training(config)

# Monitor status
status = manager.get_status()
print(f"Training running: {status['running']}")
```

---

## API Reference Summary

### Base URL
```
http://127.0.0.1:5000
```

### Available Endpoints

| Blueprint | Prefix | Description | Documentation |
|-----------|--------|-------------|---------------|
| `training` | `/training` | Training session management | [📖](./frontend/training-api.md) |
| `mesh` | `/mesh` | Mesh file operations | [📖](./frontend/mesh-api.md) |
| `predict` | `/predict` | Mesh generation prediction | [📖](./frontend/predict-api.md) |
| `training_history` | `/training/history` | Historical training data | [📖](./frontend/training-history-api.md) |
| `quality` | `/quality` | Quality calculation methods | [📖](./frontend/quality-action-apis.md) |
| `action` | `/action` | Action testing and validation | [📖](./frontend/quality-action-apis.md) |
| `geometry` | `/geometry` | Coordinate processing | [📖](./frontend/quality-action-apis.md) |
| `checkpoint` | `/checkpoint` | Model checkpoint management | *Legacy endpoints* |

---

## Common Data Models

### Vertex Coordinate
```typescript
type Vertex = [number, number]; // [x, y]
```

### Mesh Element
```typescript
type MeshElement = [number, number][]; // Array of 3-4 vertices forming a polygon
```

### Training Configuration
```typescript
interface TrainingConfig {
  mesh_name?: string;
  subfolder?: string;
  max_timesteps?: number;
  max_steps?: number;
  description?: string;
  checkpoint_name?: string;
  from_checkpoint?: boolean;
}
```

### API Response Format
```typescript
interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp?: number;
}
```

---

## Getting Started

### For Frontend Developers

1. **Choose your API**: Start with the [Training API](./frontend/training-api.md) for basic training operations
2. **Set up your client**: Use the provided JavaScript examples and TypeScript interfaces
3. **Handle errors**: Follow the error handling patterns shown in each API document
4. **Implement real-time updates**: Use polling or WebSockets for live data
5. **Test thoroughly**: Use the health check endpoints to verify connectivity

### For Backend Developers

1. **Understand the architecture**: Start with the [Training Manager](./backend/training-manager.md) overview
2. **Study existing patterns**: Follow the established patterns for new modules
3. **Implement proper error handling**: Use the exception hierarchy and recovery strategies
4. **Write tests**: Follow the testing examples for comprehensive coverage
5. **Document your changes**: Update relevant documentation when adding features

---

## Best Practices

### API Usage
- Always check the `success` field in API responses
- Implement proper error handling for network failures
- Use appropriate HTTP methods (GET for queries, POST for actions)
- Include `Content-Type: application/json` for POST requests
- Cache frequently accessed data to reduce API calls

### Development
- Follow the established code patterns and conventions
- Use type hints and proper documentation strings
- Implement comprehensive error handling and logging
- Write unit tests for new functionality
- Update documentation when adding or modifying features

### Performance
- Use batch operations when available
- Implement caching for expensive operations
- Monitor resource usage during training
- Set appropriate timeouts for long-running operations
- Clean up resources properly to prevent memory leaks

---

## Support and Contributing

### Getting Help
- Check the relevant API documentation for detailed examples
- Review the error codes and common solutions
- Examine the integration examples and best practices
- Test with the health check endpoints to verify system status

### Contributing
- Follow the existing code style and documentation patterns
- Add comprehensive tests for new features
- Update documentation to reflect changes
- Use clear commit messages and pull request descriptions
- Ensure backward compatibility when possible

---

## Version History

- **v2.0.0** (2025-01-07): Complete documentation reorganization
  - Separated frontend API docs from backend module docs
  - Added comprehensive integration examples
  - Improved TypeScript interface definitions
  - Enhanced error handling documentation

---

**📝 Note**: This documentation reflects the current state of the RL-MESH-GENERATION project as of January 2025. For the most up-to-date information, please refer to the source code and latest commits.
