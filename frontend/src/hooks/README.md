# Training Hooks - Business Logic Extraction

This directory contains custom React hooks that extract business logic from the TrainingMonitor component, following the container/presentational component pattern.

## Architecture Overview

The refactoring separates concerns as follows:

### 📋 Custom Hooks (Business Logic)

**Location**: `src/hooks/useTrainingHooks.js`

#### `useMeshBoundary(mesh)`
- Manages mesh boundary data loading and state
- Handles boundary preview functionality
- Returns: `{ boundaryData, isLoading, error, loadBoundary, clearBoundary }`

#### `useMeshData(mesh)`
- Manages mesh visualization data loading
- Handles full mesh data for training scenes
- Returns: `{ meshData, isLoading, error, loadMeshData, clearMeshData }`

#### `useReferencePoint(mesh)`
- Manages reference point finding and state
- Handles training reference point logic
- Returns: `{ refPointInfo, isLoading, error, findReferencePoint, clearReferencePoint }`

#### `useTrainingStatus({ polling, interval, onStatusChange })`
- Manages training status with optional polling
- Handles training start/stop operations
- Manages training configuration
- Returns: `{ trainingStatus, trainingConfig, isPolling, getStatus, startTraining, stopTraining, refreshStatus, updateConfig, startPolling, stopPolling }`

### 🗂️ Container Component

**Location**: `src/components/TrainingMonitor.jsx`

The TrainingMonitor now acts as a smart container that:
- ✅ Composes multiple business logic hooks
- ✅ Coordinates data flow between hooks
- ✅ Manages canvas visualization updates
- ✅ Handles user interactions and state changes
- ✅ Passes data to presentational components (MeshCanvas)

### 🎨 Presentational Components

#### MeshCanvas
**Location**: `src/components/MeshCanvas.jsx`

Already properly designed as a dumb view component:
- ✅ Well-defined props interface
- ✅ Imperative API via forwardRef
- ✅ Encapsulated CanvasRenderer
- ✅ No business logic or API calls
- ✅ Pure rendering functionality

## Benefits Achieved

### 🔄 Separation of Concerns
- **Business Logic**: Isolated in reusable hooks
- **UI Logic**: Container coordinates data flow
- **Rendering**: Pure presentational components

### ♻️ Reusability
- Hooks can be used across different components
- MeshCanvas can be reused anywhere visualization is needed
- Clear separation enables easier testing

### 🛠️ Maintainability
- Business logic changes isolated to hooks
- UI changes isolated to components
- Easier to debug and modify specific functionality

### 🧪 Testability
- Hooks can be tested independently
- Container logic is simpler to test
- Presentational components are pure

## Usage Examples

### Basic Hook Usage
```jsx
import { useMeshBoundary, useMeshData } from '../hooks/useTrainingHooks';

const MyComponent = ({ selectedMesh }) => {
  const boundary = useMeshBoundary(selectedMesh);
  const meshData = useMeshData(selectedMesh);
  
  // Use boundary.loadBoundary(), meshData.loadMeshData(), etc.
  // Access boundary.boundaryData, meshData.meshData, etc.
};
```

### Container Pattern
```jsx
import { useTrainingStatus, useMeshBoundary } from '../hooks/useTrainingHooks';

const TrainingContainer = () => {
  // Compose multiple hooks
  const training = useTrainingStatus({ polling: true });
  const boundary = useMeshBoundary(selectedMesh);
  
  // Coordinate between hooks
  const handleStart = () => {
    training.startTraining();
    // Other coordination logic...
  };
  
  return (
    <PresentationalComponent 
      trainingStatus={training.trainingStatus}
      boundaryData={boundary.boundaryData}
      onStartTraining={handleStart}
    />
  );
};
```

## Implementation Details

### State Management
- Each hook manages its own related state
- Container coordinates shared state between hooks
- No prop drilling - clean data flow

### Error Handling
- Each hook handles its own errors
- Consistent error interface across hooks
- Container can aggregate errors if needed

### Loading States
- Individual loading states per hook
- Container can derive combined loading state
- Granular loading feedback possible

### API Integration
- Hooks use the existing `useApi()` hook
- All API calls abstracted from UI components
- Consistent error handling and retry logic

## Future Enhancements

### Additional Hooks
- `useTrainingMetrics()` - For detailed training metrics
- `useMeshList()` - For mesh selection management
- `useCanvasControls()` - For canvas interaction state

### Presentational Components
- `TrainingControls` - Pure training control UI
- `TrainingStatus` - Pure status display
- `MeshSelector` - Pure mesh selection UI

### Advanced Patterns
- Hook composition for complex workflows
- Context-based hook configuration
- Optimistic updates and caching

This refactoring successfully extracts business logic into reusable hooks while maintaining a clean separation between container and presentational components, making the codebase more maintainable and testable.
