# API Provider Context & Hooks

This document describes the new singleton API context system that provides enhanced API client functionality with error handling, retry mechanisms, and polling capabilities.

## Overview

The `ApiProvider` is a React context that wraps the existing `ApiClient` class into a singleton pattern, providing two main hooks:

- `useApi()` - Returns an enhanced API client with error handling and retry baked in
- `usePolling(endpoint, interval, options)` - Generic hook for live updates and polling

## Features

### ✅ Singleton Pattern
- Single instance of `ApiClient` shared across the entire application
- Consistent configuration and state management

### ✅ Enhanced Error Handling
- Automatic error handling with user-friendly error messages
- Network timeout detection
- Connection failure detection
- Preserves original exception messages

### ✅ Retry Mechanism
- Exponential backoff retry strategy
- Configurable retry count and delay
- Automatic retry on network failures

### ✅ Polling Support
- Generic polling hook for any API endpoint
- Configurable polling intervals
- Dependency-based re-polling
- Manual control (start/stop/refresh)

### ✅ Complete API Coverage
- All existing API endpoints preserved
- Training APIs (start, stop, status, health)
- Mesh APIs (list, info, boundary, data)
- Checkpoint APIs (list, info, validate, delete, copy)
- Action APIs (execute, validate, find reference points)

## Usage

### Setup

Wrap your app with the `ApiProvider`:

```jsx
import { ApiProvider } from './context/ApiProvider';

function App() {
  return (
    <ApiProvider>
      {/* Your app components */}
    </ApiProvider>
  );
}
```

### useApi() Hook

The `useApi()` hook returns an enhanced API client with all methods wrapped in error handling and retry logic:

```jsx
import { useApi } from '../context/ApiProvider';

const MyComponent = () => {
  const api = useApi();

  const handleLoadData = async () => {
    try {
      // All API methods have error handling and retry built-in
      const status = await api.getTrainingStatus();
      const meshes = await api.getMeshList();
      const checkpoints = await api.getCheckpointList();
      
      console.log('Data loaded successfully');
    } catch (error) {
      // Error handling and retry already attempted
      console.error('Failed to load data:', error.message);
    }
  };

  return (
    <button onClick={handleLoadData}>
      Load Data
    </button>
  );
};
```

### usePolling() Hook

The `usePolling()` hook provides automatic polling with comprehensive options:

#### Basic Usage

```jsx
import { usePolling } from '../context/ApiProvider';

const StatusMonitor = () => {
  // Poll training status every 2 seconds
  const { data, error, isLoading } = usePolling('getTrainingStatus', 2000);

  return (
    <div>
      {isLoading && <span>Loading...</span>}
      {error && <span>Error: {error.message}</span>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
};
```

#### Advanced Usage with Options

```jsx
const MeshMonitor = ({ selectedMesh }) => {
  const {
    data: meshData,
    error,
    isLoading,
    isPolling,
    refresh,
    startPolling,
    stopPolling
  } = usePolling('getMeshBoundary', 5000, {
    // Only poll when mesh is selected and component is active
    enabled: selectedMesh && isActive,
    
    // Pass arguments to the API method
    methodArgs: [selectedMesh, 'mesh'],
    
    // Re-start polling when dependencies change
    dependencies: [selectedMesh],
    
    // Success callback
    onSuccess: (data) => {
      console.log('Mesh data updated:', data);
    },
    
    // Error callback
    onError: (error) => {
      console.error('Polling error:', error);
    }
  });

  return (
    <div>
      <div>Status: {isPolling ? 'Polling' : 'Stopped'}</div>
      <button onClick={refresh}>Refresh Now</button>
      <button onClick={isPolling ? stopPolling : startPolling}>
        {isPolling ? 'Stop' : 'Start'} Polling
      </button>
      
      {/* Display data... */}
    </div>
  );
};
```

#### Polling Direct Endpoints

```jsx
// Poll a direct API endpoint
const { data } = usePolling('/training/health', 10000);

// Poll with custom function
const { data } = usePolling(
  () => fetch('/custom/endpoint').then(r => r.json()),
  3000
);
```

## API Methods

All original `ApiClient` methods are available through `useApi()`:

### Training APIs
- `getTrainingStatus()` - Get current training status
- `startTraining(config)` - Start training with configuration
- `stopTraining()` - Stop current training
- `checkTrainingHealth()` - Check training API health
- `getTrainingReferencePoint(data)` - Get reference point for training

### Mesh APIs
- `getMeshList(subfolder?)` - Get available mesh list
- `getMeshInfo(meshName, subfolder?)` - Get mesh information
- `getMeshBoundary(meshName, subfolder?)` - Get mesh boundary data
- `getMeshData(meshName)` - Get mesh data for visualization
- `checkMeshHealth()` - Check mesh API health

### Checkpoint APIs
- `getCheckpointList()` - Get available checkpoints
- `getCheckpointInfo(checkpointName)` - Get checkpoint information
- `validateCheckpoint(checkpointName)` - Validate checkpoint
- `deleteCheckpoint(checkpointName)` - Delete checkpoint
- `copyCheckpointFromHistory(trainingId, checkpointName?)` - Copy from history
- `checkCheckpointHealth()` - Check checkpoint API health

### Action APIs
- `findReferencePoint(meshName)` - Find reference point for mesh
- `executeAction(actionData)` - Execute and validate action
- `validateAction(actionType, actionData)` - Validate specific action
- `getActionInfo()` - Get available actions info
- `checkActionHealth()` - Check action API health

## Configuration

### Default Constants

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

### Error Handling

The system provides enhanced error messages:

- Network timeouts: "Request timed out, please check network connection"
- Connection failures: "Network connection failed, please check server status"
- Server errors: Uses server-provided error messages
- All original exception messages are preserved

### Retry Strategy

- Exponential backoff: delay × 2^attempt
- Maximum configurable retry attempts
- Only retries on network/timeout errors
- Server errors (4xx, 5xx) are not retried

## Migration Guide

### From Direct fetch() Calls

**Before:**
```jsx
const response = await fetch('/api/training/status');
const data = await response.json();
```

**After:**
```jsx
const api = useApi();
const data = await api.getTrainingStatus();
```

### From Manual Polling

**Before:**
```jsx
useEffect(() => {
  const interval = setInterval(async () => {
    try {
      const response = await fetch('/api/training/status');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error(error);
    }
  }, 2000);
  
  return () => clearInterval(interval);
}, []);
```

**After:**
```jsx
const { data: status } = usePolling('getTrainingStatus', 2000);
```

## Examples

See `frontend/src/components/examples/ApiHookExamples.jsx` for comprehensive usage examples including:

- Basic API calls with `useApi()`
- Polling with method names
- Polling with parameters and dependencies
- Polling direct endpoints
- Manual control of polling
- Error handling demonstrations

## Benefits

1. **Consistency** - Single API interface across the entire application
2. **Reliability** - Built-in error handling and retry mechanisms
3. **Performance** - Efficient polling with automatic cleanup
4. **Developer Experience** - Simple hooks with comprehensive options
5. **Maintainability** - Centralized API logic and configuration
6. **Backward Compatibility** - All existing endpoints preserved

## Architecture

```
ApiProvider (Singleton Context)
├── ApiClient (Singleton Instance)
├── Enhanced Methods (Error Handling + Retry)
├── useApi() Hook
└── usePolling() Hook
    ├── Automatic Polling
    ├── Manual Controls
    ├── Error Handling
    └── Cleanup
```

The system maintains full compatibility with the existing `ApiClient` while adding the benefits of React context, hooks, and enhanced error handling.
