# TrainingMonitor Component

## Overview
A comprehensive training monitoring interface that provides mesh visualization, training controls, and status monitoring for reinforcement learning mesh generation. Serves as an example component demonstrating MeshCanvas integration.

## File Location
`frontend/src/components/TrainingMonitor.jsx`

## Props
This component does not accept any props - it's a self-contained monitoring interface.

## State Usage
| State Variable | Type | Default | Purpose |
|---------------|------|---------|---------|
| `selectedMesh` | string | `''` | Currently selected mesh name |
| `meshData` | Object\|null | `null` | Full mesh data for visualization |
| `boundaryData` | Object\|null | `null` | Boundary vertices data |
| `refPointInfo` | Object\|null | `null` | Reference point information |
| `clickCoordinates` | Object\|null | `null` | Canvas click coordinates (world + screen) |
| `isLoading` | boolean | `false` | Loading state for async operations |

## Dependencies

### React Dependencies
- `useRef` - For accessing MeshCanvas imperative methods
- `useEffect` - For component initialization
- `useState` - For state management
- `useCallback` - For memoized callback functions

### Internal Dependencies
- `MeshCanvas` from `'./MeshCanvas'` - Canvas rendering component
- `useApi` from `'../context/ApiProvider'` - API client hook
- `usePolling` from `'../context/ApiProvider'` - Polling utilities hook

### External Dependencies
- None

## Side Effects

### API Calls
- **Mesh Boundary Loading**: Calls `api.getMeshBoundary(meshName)` when mesh is selected
- **Mesh Data Loading**: Calls `api.getMeshData(meshName)` for full mesh visualization
- **Reference Point Finding**: Calls `api.getTrainingReferencePoint()` for training setup

### Canvas Manipulation
- **Boundary Preview**: Calls `canvasRef.current.renderBoundaryPreview()` for mesh boundary visualization
- **Scene Rendering**: Calls `canvasRef.current.renderScene()` for full mesh rendering
- **Canvas Clearing**: Calls `canvasRef.current.clearCanvas()` when clearing state

### Logging
- Console logging of canvas click events and API responses

## Features

### Mesh Selection and Visualization
- **Dropdown Selection**: Choose from predefined mesh options (simple_square, complex_polygon, curved_boundary)
- **Boundary Preview**: Automatic boundary visualization when mesh is selected
- **Full Mesh Loading**: Load complete mesh data with visualization
- **Canvas Clearing**: Reset visualization state

### Training Controls
- **Start Training**: Button for initiating training (placeholder)
- **Stop Training**: Button for stopping training (placeholder)
- **Reference Point Finding**: Locate optimal reference points for training

### Interactive Canvas
- **Click Handling**: Capture and display canvas click coordinates
- **World Coordinates**: Convert screen coordinates to mesh world coordinates
- **Visual Feedback**: Crosshair cursor when mesh is selected

### Status Monitoring
- **Real-time Status**: Display current state of mesh, boundary, and reference point
- **Loading States**: Visual feedback during async operations
- **Click Coordinates**: Show last clicked position in world coordinates

## API Integration

### Error Handling
- Try-catch blocks around all API calls
- Console error logging for debugging
- Graceful fallback when API calls fail

### API Methods Used
- `api.getMeshBoundary(meshName)` - Get boundary vertices
- `api.getMeshData(meshName)` - Get full mesh data
- `api.getTrainingReferencePoint({ mesh })` - Find reference point

## Canvas Integration

### Imperative Methods
Uses MeshCanvas ref to call:
- `renderBoundaryPreview(vertices, meshName)` - Preview mesh boundary
- `renderScene(meshData, boundaryData, refPointInfo)` - Full scene rendering
- `clearCanvas()` - Clear visualization

### Event Handling
- Handles canvas click events via `onCanvasClick` prop
- Converts click coordinates to world space
- Updates state with click information

## Visual Layout

### Grid Layout
- 3-column layout on large screens
- Responsive design with proper breakpoints
- Card-based UI with consistent borders and spacing

### Status Cards
- **Mesh Selection**: Dropdown and action buttons
- **Training Controls**: Start/stop/clear buttons
- **Status Info**: Real-time status indicators

### Canvas Container
- Fixed height (600px) canvas container
- Loading overlay during operations
- Gradient background for visual appeal

## CSS Classes Used
- `max-w-6xl mx-auto p-6` - Main container
- `grid grid-cols-1 lg:grid-cols-3 gap-6` - Responsive grid
- `bg-card border border-border-custom rounded-lg` - Card styling
- `relative` with fixed height for canvas container

## Known Issues
1. **Placeholder Controls**: Start/Stop training buttons are not implemented
2. **Hard-coded Mesh Options**: Mesh selection is limited to three hard-coded options
3. **No Error UI**: API errors only logged to console, no user feedback
4. **Memory Leaks**: Canvas state not properly cleaned up on unmount
5. **No Validation**: No validation of mesh selection before operations

## Usage Example
```jsx
import TrainingMonitor from './components/TrainingMonitor'

function TrainingPage() {
  return (
    <div>
      <TrainingMonitor />
    </div>
  )
}
```

## Performance Considerations
1. **Callback Memoization**: Uses `useCallback` to prevent unnecessary re-renders
2. **Conditional Rendering**: Only renders canvas when mesh is selected
3. **Loading States**: Prevents multiple simultaneous API calls

## Potential Improvements
1. **Dynamic Mesh Loading**: Load available meshes from API instead of hard-coding
2. **Error Boundaries**: Add error boundary for graceful error handling
3. **Training Integration**: Implement actual training start/stop functionality
4. **State Persistence**: Save selected mesh across sessions
5. **Canvas Controls**: Add zoom, pan, and view controls
6. **Export Features**: Add ability to export mesh data or visualizations
7. **Accessibility**: Add proper ARIA labels and keyboard navigation

## Related Components
- **Requires**: MeshCanvas for visualization
- **Uses**: ApiProvider context for data fetching
- **Used by**: Train page as main training interface
