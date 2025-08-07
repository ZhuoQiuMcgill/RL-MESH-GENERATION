# API Provider Documentation

> **Status**: `Complete`  
> **Version**: v1.0.0  
> **Last Updated**: 2025-01-07  
> **File**: `frontend/src/context/ApiProvider.jsx`

## Table of Contents

- [Overview](#overview)
- [Singleton Behavior](#singleton-behavior)
- [Error Handling Strategy](#error-handling-strategy)
- [Retry Strategy](#retry-strategy)
- [Timeout Configuration](#timeout-configuration)
- [API Methods Reference](#api-methods-reference)
- [Training Monitor Method Mapping](#training-monitor-method-mapping)
- [Hook Usage Patterns](#hook-usage-patterns)
- [Configuration Constants](#configuration-constants)

## Overview

The `ApiProvider` is a React context that implements a singleton pattern to provide a centralized API client for the entire application. It wraps the `ApiClient` class with enhanced error handling, retry mechanisms, and React-specific hooks for seamless integration.

**Key Features:**
- Singleton pattern ensures one API instance across the application
- Enhanced error handling with user-friendly messages
- Exponential backoff retry strategy for network failures
- React hooks for API calls (`useApi`) and polling (`usePolling`)
- Comprehensive timeout management
- Full backwards compatibility with existing API methods

## Singleton Behavior

### Implementation
```javascript
class ApiClient {
  constructor() {
    if (ApiClient.instance) {
      return ApiClient.instance;
    }
    this.baseUrl = CONSTANTS.API_BASE_URL;
    ApiClient.instance = this;
  }
}
```

### Benefits
- **Consistency**: Single configuration and connection pool
- **Memory Efficiency**: One instance handles all requests
- **State Management**: Centralized API state across components
- **Performance**: Reduces overhead of multiple client instantiation

### Context Integration
The singleton is wrapped in React Context and exposed via `useApi()` hook, ensuring all components access the same enhanced instance.

## Error Handling Strategy

### Enhanced Error Messages
The API provider transforms low-level network errors into user-friendly messages:

| Original Error | Enhanced Message |
|---|---|
| `timeout` | "Request timed out, please check network connection" |
| `Failed to fetch` | "Network connection failed, please check server status" |
| Server errors | Original server error message (preserved) |

### Error Wrapping Function
```javascript
function withErrorHandling(apiMethod) {
  return async function (...args) {
    try {
      return await apiMethod.apply(this, args);
    } catch (error) {
      console.error('API call failed:', error);
      
      if (error.message.includes('timeout')) {
        throw new Error('Request timed out, please check network connection');
      } else if (error.message.includes('Failed to fetch')) {
        throw new Error('Network connection failed, please check server status');
      } else {
        throw error; // Preserve original server errors
      }
    }
  };
}
```

## Retry Strategy

### Exponential Backoff Algorithm
```javascript
async function withRetry(apiCall, maxRetries = 1, retryDelay = 3000) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall();
    } catch (error) {
      if (attempt < maxRetries) {
        await delay(retryDelay * Math.pow(2, attempt)); // Exponential backoff
      }
    }
  }
  throw lastError;
}
```

### Retry Configuration
- **Default Max Retries**: 1 attempt
- **Default Initial Delay**: 3000ms (3 seconds)  
- **Backoff Factor**: 2x (exponential)
- **Retry Schedule**: 3s, 6s, 12s, etc.

### Retry Conditions
- **Retries On**: Network timeouts, connection failures
- **No Retries On**: Server errors (4xx, 5xx status codes)
- **Logging**: Each retry attempt is logged with attempt count

## Timeout Configuration

### Timeout Types and Values

| Operation | Timeout | Constant | Description |
|---|---|---|---|
| **Default Connection** | 10,000ms | `CONNECTION_TIMEOUT` | Standard API requests |
| **Training Stop** | 30,000ms | `TRAINING_STOP_TIMEOUT` | Training termination (longer due to cleanup) |
| **Polling Default** | 2,000ms | `DEFAULT_POLLING_INTERVAL` | usePolling hook interval |

### AbortController Implementation
```javascript
async request(endpoint, options = {}, customTimeout = null) {
  const timeout = customTimeout || CONSTANTS.CONNECTION_TIMEOUT;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  const response = await fetch(url, {
    ...requestOptions,
    signal: controller.signal
  });
  
  clearTimeout(timeoutId);
  return response;
}
```

### Custom Timeout Usage
Methods can specify custom timeouts:
```javascript
// Uses 30-second timeout instead of default 10-second
async stopTraining() {
  return await this.request('/training/stop', {
    method: 'POST'
  }, CONSTANTS.TRAINING_STOP_TIMEOUT);
}
```

## API Methods Reference

### Training APIs

#### `getTrainingStatus()`
- **HTTP Method**: GET
- **Path**: `/training/status`  
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    running: boolean,
    status: string,
    stats: object,
    progress: object,
    timestamp: number
  }
  ```
- **Error Cases**: Network timeout, connection failure, server errors

#### `startTraining(config)`
- **HTTP Method**: POST
- **Path**: `/training/start`
- **Arguments**: 
  ```javascript
  {
    mesh_name?: string,
    subfolder?: string,
    max_timesteps?: number,
    max_steps?: number,
    description?: string,
    checkpoint_name?: string,
    from_checkpoint?: boolean
  }
  ```
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    message: string,
    config: object
  }
  ```
- **Error Cases**: Training already running, invalid config, mesh not found

#### `stopTraining()`
- **HTTP Method**: POST  
- **Path**: `/training/stop`
- **Arguments**: None
- **Timeout**: 30s (extended)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    message: string
  }
  ```
- **Error Cases**: No training running, stop timeout, server error

#### `checkTrainingHealth()`
- **HTTP Method**: GET
- **Path**: `/training/health`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    status: "healthy|unhealthy",
    service: "training-api",
    manager_running: boolean,
    timestamp: number
  }
  ```
- **Error Cases**: Service unavailable

#### `getTrainingReferencePoint(data)`
- **HTTP Method**: POST
- **Path**: `/training/reference-point`
- **Arguments**: 
  ```javascript
  {
    mesh?: string,
    // Additional reference point parameters
  }
  ```
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    reference_point: {
      x: number,
      y: number,
      // Additional point data
    }
  }
  ```
- **Error Cases**: Invalid mesh, calculation failure

### Mesh APIs

#### `getMeshList(subfolder = 'mesh')`
- **HTTP Method**: GET
- **Path**: `/mesh/list?subfolder={subfolder}`
- **Arguments**: `subfolder: string` (optional, default: 'mesh')
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    meshes: string[],
    count: number
  }
  ```
- **Error Cases**: Directory not found, read permission error

#### `getMeshInfo(meshName, subfolder = 'mesh')`
- **HTTP Method**: GET
- **Path**: `/mesh/info/{meshName}?subfolder={subfolder}`
- **Arguments**: `meshName: string`, `subfolder: string` (optional)
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    name: string,
    exists: boolean,
    vertices?: number,
    size?: number
  }
  ```
- **Error Cases**: Mesh not found, invalid mesh format

#### `getMeshBoundary(meshName, subfolder = 'mesh')`
- **HTTP Method**: GET
- **Path**: `/mesh/boundary/{meshName}?subfolder={subfolder}`
- **Arguments**: `meshName: string`, `subfolder: string` (optional)
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    mesh_name: string,
    subfolder: string,
    boundary_vertices: number[][],
    vertex_count: number
  }
  ```
- **Error Cases**: Mesh not found, boundary extraction failure

#### `getMeshData(meshName)`
- **HTTP Method**: GET
- **Path**: `/mesh/data/{meshName}`
- **Arguments**: `meshName: string`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    mesh_data: {
      vertices: number[][],
      elements: number[][],
      // Additional mesh structure data
    }
  }
  ```
- **Error Cases**: Mesh not found, data parsing failure

#### `checkMeshHealth()`
- **HTTP Method**: GET
- **Path**: `/mesh/health`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    status: "healthy|unhealthy",
    service: "mesh-api",
    timestamp: number
  }
  ```
- **Error Cases**: Service unavailable

### Checkpoint APIs

#### `getCheckpointList()`
- **HTTP Method**: GET
- **Path**: `/checkpoint/list`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    checkpoints: string[],
    count: number
  }
  ```
- **Error Cases**: Directory access error

#### `getCheckpointInfo(checkpointName)`
- **HTTP Method**: GET
- **Path**: `/checkpoint/info/{checkpointName}`
- **Arguments**: `checkpointName: string`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    name: string,
    size: number,
    created: string,
    valid: boolean
  }
  ```
- **Error Cases**: Checkpoint not found, corrupted checkpoint

#### `validateCheckpoint(checkpointName)`
- **HTTP Method**: GET
- **Path**: `/checkpoint/validate/{checkpointName}`
- **Arguments**: `checkpointName: string`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    valid: boolean,
    message: string
  }
  ```
- **Error Cases**: Checkpoint not found, validation failure

#### `deleteCheckpoint(checkpointName)`
- **HTTP Method**: DELETE
- **Path**: `/checkpoint/delete/{checkpointName}`
- **Arguments**: `checkpointName: string`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    message: string
  }
  ```
- **Error Cases**: Checkpoint not found, permission error

#### `copyCheckpointFromHistory(trainingId, checkpointName?)`
- **HTTP Method**: POST
- **Path**: `/checkpoint/copy`
- **Arguments**: 
  ```javascript
  {
    training_id: string,
    checkpoint_name?: string
  }
  ```
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    copied_name: string,
    message: string
  }
  ```
- **Error Cases**: Training history not found, copy failure

#### `checkCheckpointHealth()`
- **HTTP Method**: GET
- **Path**: `/checkpoint/health`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    status: "healthy|unhealthy",
    service: "checkpoint-api",
    timestamp: number
  }
  ```
- **Error Cases**: Service unavailable

### Action APIs

#### `findReferencePoint(meshName)`
- **HTTP Method**: GET
- **Path**: `/action/find-ref-point/{meshName}`
- **Arguments**: `meshName: string`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    reference_point: {
      x: number,
      y: number,
      quality_score: number
    }
  }
  ```
- **Error Cases**: Mesh not found, algorithm failure

#### `executeAction(actionData)`
- **HTTP Method**: POST
- **Path**: `/action/execute`
- **Arguments**: 
  ```javascript
  {
    action_type: string,
    parameters: object,
    mesh_context?: object
  }
  ```
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    success: boolean,
    result: object,
    execution_time: number
  }
  ```
- **Error Cases**: Invalid action, execution failure

#### `validateAction(actionType, actionData)`
- **HTTP Method**: POST
- **Path**: `/action/validate/{actionType}`
- **Arguments**: `actionType: string`, `actionData: object`
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    valid: boolean,
    errors?: string[],
    warnings?: string[]
  }
  ```
- **Error Cases**: Unknown action type, validation failure

#### `getActionInfo()`
- **HTTP Method**: GET
- **Path**: `/action/info`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    available_actions: string[],
    action_descriptions: object
  }
  ```
- **Error Cases**: Service configuration error

#### `checkActionHealth()`
- **HTTP Method**: GET
- **Path**: `/action/health`
- **Arguments**: None
- **Timeout**: 10s (default)
- **Return Shape**: 
  ```javascript
  {
    status: "healthy|unhealthy",
    service: "action-api",
    timestamp: number
  }
  ```
- **Error Cases**: Service unavailable

## Training Monitor Method Mapping

The `TrainingMonitor` component references the following API methods. All methods **exist and are implemented** in the `ApiProvider`:

### ✅ Required Methods (All Present)

| TrainingMonitor Usage | ApiProvider Method | Status |
|---|---|---|
| `api.getMeshBoundary(meshName)` | `getMeshBoundary(meshName, subfolder = 'mesh')` | ✅ **EXISTS** |
| `api.getMeshData(meshName)` | `getMeshData(meshName)` | ✅ **EXISTS** |
| `api.getTrainingReferencePoint(data)` | `getTrainingReferencePoint(data)` | ✅ **EXISTS** |
| `api.getTrainingStatus()` | `getTrainingStatus()` | ✅ **EXISTS** |
| `api.startTraining(config)` | `startTraining(config)` | ✅ **EXISTS** |
| `api.stopTraining()` | `stopTraining()` | ✅ **EXISTS** |

### Method Signature Compatibility

All methods are **fully compatible** with TrainingMonitor usage patterns:

```javascript
// TrainingMonitor usage examples:
const data = await api.getMeshBoundary(meshName);        // ✅ Compatible
const meshData = await api.getMeshData(meshName);        // ✅ Compatible  
const refPoint = await api.getTrainingReferencePoint({   // ✅ Compatible
  mesh: selectedMesh 
});
const status = await api.getTrainingStatus();            // ✅ Compatible
const result = await api.startTraining(config);          // ✅ Compatible
const stopResult = await api.stopTraining();             // ✅ Compatible
```

### **No Implementation Gaps**

All methods referenced by TrainingMonitor are present and functional. No additional implementation is required.

## Hook Usage Patterns

### useApi() Hook

The `useApi()` hook provides access to the enhanced API client with error handling and retry built-in.

#### Basic Usage
```javascript
import { useApi } from '../context/ApiProvider';

const MyComponent = () => {
  const api = useApi();
  
  const handleLoad = async () => {
    try {
      const result = await api.getTrainingStatus();
      console.log('Status:', result);
    } catch (error) {
      console.error('Failed:', error.message);
    }
  };
  
  return <button onClick={handleLoad}>Load Status</button>;
};
```

#### Error Handling Pattern
```javascript
const api = useApi();

// All API calls automatically include:
// - User-friendly error messages
// - Retry mechanism with exponential backoff
// - Timeout handling
// - Network failure detection

const handleApiCall = async () => {
  try {
    const data = await api.getMeshBoundary('simple_square');
    // Success handling
  } catch (error) {
    // Error is already enhanced with user-friendly message
    showNotification(error.message, 'error');
  }
};
```

### usePolling() Hook

The `usePolling()` hook provides automatic polling with comprehensive configuration options.

#### Basic Polling Pattern
```javascript
import { usePolling } from '../context/ApiProvider';

const StatusMonitor = () => {
  const { data, error, isLoading } = usePolling('getTrainingStatus', 2000);
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <div>
      Status: {data?.status}
      Running: {data?.running ? 'Yes' : 'No'}
    </div>
  );
};
```

#### Advanced Polling with Dependencies
```javascript
const MeshBoundaryPolling = ({ selectedMesh, isActive }) => {
  const {
    data: boundaryData,
    error,
    isLoading,
    isPolling,
    refresh,
    startPolling,
    stopPolling
  } = usePolling('getMeshBoundary', 5000, {
    // Only poll when conditions are met
    enabled: selectedMesh && isActive,
    
    // Pass arguments to the API method
    methodArgs: [selectedMesh, 'mesh'],
    
    // Restart polling when dependencies change
    dependencies: [selectedMesh],
    
    // Success callback
    onSuccess: (data) => {
      console.log('Boundary updated for:', selectedMesh);
      updateVisualization(data.boundary_vertices);
    },
    
    // Error callback  
    onError: (error) => {
      console.error('Boundary polling failed:', error);
      showNotification(`Failed to load ${selectedMesh} boundary`);
    }
  });

  return (
    <div>
      <div>Status: {isPolling ? 'Polling' : 'Stopped'}</div>
      <button onClick={refresh}>Refresh Now</button>
      <button onClick={isPolling ? stopPolling : startPolling}>
        {isPolling ? 'Stop' : 'Start'} Polling
      </button>
      
      {isLoading && <div>Loading boundary...</div>}
      {error && <div>Error: {error.message}</div>}
      {boundaryData && (
        <div>
          Mesh: {boundaryData.mesh_name}
          Vertices: {boundaryData.vertex_count}
        </div>
      )}
    </div>
  );
};
```

#### Direct Endpoint Polling
```javascript
// Poll a direct API endpoint
const HealthMonitor = () => {
  const { data, error } = usePolling('/training/health', 10000, {
    onSuccess: (data) => {
      if (data.status === 'unhealthy') {
        showNotification('Training service is unhealthy!', 'warning');
      }
    }
  });
  
  return (
    <div className={data?.status === 'healthy' ? 'text-green-500' : 'text-red-500'}>
      Service: {data?.status || 'Unknown'}
    </div>
  );
};
```

#### Custom Function Polling  
```javascript
// Poll with custom function
const CustomPolling = () => {
  const { data } = usePolling(
    async () => {
      // Custom API call logic
      const status = await fetch('/custom/endpoint').then(r => r.json());
      const metrics = await fetch('/metrics/latest').then(r => r.json());
      
      return { status, metrics };
    },
    3000,
    {
      enabled: true,
      onError: (error) => console.error('Custom polling failed:', error)
    }
  );
  
  return <div>{JSON.stringify(data, null, 2)}</div>;
};
```

#### Polling Control Patterns
```javascript
const TrainingControlPanel = () => {
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [pollingInterval, setPollingInterval] = useState(2000);
  
  const {
    data: trainingStatus,
    isPolling,
    startPolling,
    stopPolling,
    refresh
  } = usePolling('getTrainingStatus', pollingInterval, {
    enabled: autoUpdate,
    onSuccess: (data) => {
      if (data.status === 'completed') {
        // Auto-stop polling when training completes
        setAutoUpdate(false);
        showNotification('Training completed!', 'success');
      }
    }
  });

  return (
    <div>
      <div>
        <label>
          <input
            type="checkbox"
            checked={autoUpdate}
            onChange={(e) => setAutoUpdate(e.target.checked)}
          />
          Auto Update
        </label>
      </div>
      
      <div>
        <label>Update Interval:</label>
        <select
          value={pollingInterval}
          onChange={(e) => setPollingInterval(Number(e.target.value))}
        >
          <option value={1000}>1 second</option>
          <option value={2000}>2 seconds</option>
          <option value={5000}>5 seconds</option>
          <option value={10000}>10 seconds</option>
        </select>
      </div>
      
      <div>
        <button onClick={refresh} disabled={!trainingStatus}>
          Refresh Now
        </button>
        <button 
          onClick={isPolling ? stopPolling : startPolling}
          disabled={!autoUpdate}
        >
          {isPolling ? 'Stop' : 'Start'} Polling
        </button>
      </div>
      
      <div>
        Status: {trainingStatus?.status || 'Unknown'}
        {isPolling && <span> (Polling every {pollingInterval}ms)</span>}
      </div>
    </div>
  );
};
```

## Configuration Constants

### Default Configuration
```javascript
const CONSTANTS = {
  API_BASE_URL: 'http://localhost:8000',
  CONNECTION_TIMEOUT: 10000,          // 10 seconds
  TRAINING_STOP_TIMEOUT: 30000,       // 30 seconds  
  DEFAULT_RETRY_COUNT: 1,              // 1 retry attempt
  DEFAULT_RETRY_DELAY: 3000,           // 3 second initial delay
  DEFAULT_POLLING_INTERVAL: 2000       // 2 second polling
};
```

### Configuration Usage
- **API_BASE_URL**: Base server URL for all requests
- **CONNECTION_TIMEOUT**: Default timeout for all API calls
- **TRAINING_STOP_TIMEOUT**: Extended timeout for training termination
- **DEFAULT_RETRY_COUNT**: Maximum retry attempts on failure
- **DEFAULT_RETRY_DELAY**: Initial delay before first retry
- **DEFAULT_POLLING_INTERVAL**: Default interval for usePolling hook

### Environment Customization
Constants can be overridden via environment variables or configuration files for different deployment environments (development, staging, production).
