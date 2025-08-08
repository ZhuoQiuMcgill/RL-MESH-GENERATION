# Action (ActionTester) Page

## Overview
A comprehensive interactive interface for testing reinforcement learning actions on meshes. Provides a step-by-step workflow for mesh selection, reference point finding, action selection, and execution with real-time visualization.

## File Location
`frontend/src/pages/Action.jsx`

## Props
This page component does not accept any props.

## State Usage
| State Variable | Type | Default | Purpose |
|---------------|------|---------|---------|
| `meshList` | Array | `[]` | Available meshes from API |
| `selectedMesh` | string | `''` | Currently selected mesh name |
| `meshInfo` | Object\|null | `null` | Detailed mesh information |
| `referencePoint` | Object\|null | `null` | Found reference point data |
| `selectedAction` | string\|null | `null` | Selected action type |
| `clickCoordinates` | Array\|null | `null` | Canvas click coordinates for Type1 actions |
| `executionResult` | Object\|null | `null` | Action execution results |
| `isLoading` | boolean | `false` | Loading state |
| `error` | string\|null | `null` | Error messages |
| `log` | Array | `[]` | Operation log entries |
| `isWaitingForClick` | boolean | `false` | Waiting for canvas click state |
| `status` | string | `'Ready'` | Current operation status |

## Dependencies

### React Dependencies
- `useState`, `useEffect`, `useRef`, `useCallback` - Standard React hooks

### Internal Dependencies
- `NavHeader`, `MeshCanvas` from `'../components'` 
- UI Components: `Button`, `FormSelect`, `LoadingOverlay`, `CompactStatusBar`, `PanelCard`, `EmptyState`
- `useApi` from `'../context/ApiProvider'` - API client

### External Dependencies
- None

## Side Effects

### API Integration
- **Mesh List Loading**: `api.getMeshList()` - Get available meshes
- **Mesh Info**: `api.getMeshInfo(meshName)` - Get mesh details
- **Boundary Loading**: `api.getMeshBoundary(meshName)` - Load mesh boundaries for visualization
- **Reference Point Finding**: `api.findReferencePoint(meshName)` - Find optimal reference point
- **Action Execution**: `api.executeAction(actionData)` - Execute selected action

### Canvas Integration
- **Boundary Rendering**: Display mesh boundaries when mesh selected
- **Click Handling**: Capture canvas clicks for Type1 actions
- **Coordinate Conversion**: Convert screen clicks to world coordinates
- **Visual Feedback**: Update cursor based on interaction state

## Features

### Step-by-Step Workflow
1. **Step 1: Mesh Selection** - Choose from available meshes
2. **Step 2: Reference Point** - Find optimal reference point for actions
3. **Step 3: Action Selection** - Choose action type (Type0-left, Type0-right, Type1)
4. **Step 4: Execution** - Execute the selected action

### Action Types
- **Type0-left**: Left-side action (no click required)
- **Type0-right**: Right-side action (no click required) 
- **Type1**: Interactive action requiring canvas click for vertex placement

### Real-time Status System
- **Status Bar**: Compact status indicator at top
- **Color-coded States**: Different colors for different operation states
- **Dynamic Updates**: Status updates throughout workflow

### Interactive Canvas
- **Click to Draw**: Type1 actions require canvas interaction
- **World Coordinates**: Display clicked coordinates in world space
- **Visual Feedback**: Crosshair cursor when waiting for click
- **Empty State**: Helpful message when no mesh selected

## Complex Layout

### Multi-Panel Design
- **Left Panel (320px)**: Step-by-step controls and configuration
- **Main Area**: Split between canvas and information panel
- **Canvas Area**: MeshCanvas with responsive sizing
- **Right Panel (320px)**: Status, mesh info, and detailed results
- **Bottom Log**: Action log area with auto-scroll option

### Responsive Features
- **Flexible Canvas**: Canvas area adjusts to available space
- **Minimum Widths**: Panels maintain usability on smaller screens
- **Status Indicators**: Multiple status displays for different contexts

## Status Management

### Status Colors
```jsx
const getStatusColor = () => {
  if (status.includes('Error')) return 'danger';
  if (status.includes('Loading') || status.includes('Finding') || status.includes('Executing')) return 'warning';
  if (status.includes('Found') || status.includes('Valid')) return 'success';
  return 'primary';
};
```

### Status Updates
- **Ready** → **Loading Mesh** → **Mesh Loaded**
- **Finding Reference Point** → **Reference Point Found** 
- **Waiting for Canvas Click** → **Click Recorded**
- **Executing Action** → **Action Valid/Invalid**

## Canvas Integration

### Click Handling
```jsx
const handleCanvasClick = (worldCoords, event) => {
  if (!isWaitingForClick) return;
  
  if (worldCoords) {
    setClickCoordinates(worldCoords);
    setIsWaitingForClick(false);
    setStatus('Click Recorded');
    // ... logging and state updates
  }
};
```

### Visualization States
- **Empty State**: When no mesh selected
- **Boundary Preview**: When mesh loaded
- **Interactive Mode**: When waiting for Type1 click
- **Result Display**: After action execution

## Information Panels

### Status Overview
- Mesh selection status
- Boundary vertex count
- Reference point status
- Current action type

### Detailed Cards
- **Mesh Information**: Vertices, file size
- **Reference Point**: Index, coordinates, interior angle
- **Action Details**: Type, clicked coordinates
- **Execution Results**: Validation status, polar coordinates

## Action Log System
- **Timestamped Entries**: All operations logged with precise timestamps
- **Categorized Logging**: Info, success, warning, error categories
- **Auto-scroll Option**: Checkbox to enable automatic log scrolling
- **Color-coded Messages**: Visual distinction for different message types
- **Clearable History**: User can clear log at any time

## Known Issues
1. **API Dependencies**: Some API methods may not be fully implemented
2. **Canvas State**: Canvas visualization updates commented out pending backend
3. **Error Recovery**: Limited error recovery mechanisms
4. **Mobile Support**: Touch interactions not optimized for mobile
5. **Memory Usage**: No cleanup of large mesh data structures

## Usage Example
```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Action from './pages/Action'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/action" element={<Action />} />
      </Routes>
    </Router>
  )
}
```

## Validation Logic
```jsx
const canExecute = () => {
  if (!selectedMesh || !referencePoint || !selectedAction) return false;
  if (selectedAction === 'type1' && !clickCoordinates) return false;
  return true;
};
```

## Performance Considerations
1. **Callback Memoization**: Uses `useCallback` for performance
2. **Conditional Rendering**: Only renders canvas when needed
3. **Loading States**: Prevents multiple simultaneous operations
4. **Log Management**: Could benefit from log entry limits

## Potential Improvements
1. **Keyboard Shortcuts**: Add keyboard shortcuts for common actions
2. **Action History**: Save and replay previous action sequences
3. **Batch Operations**: Execute multiple actions in sequence
4. **Export Results**: Export action results and mesh data
5. **Advanced Visualization**: Add zoom, pan, and measurement tools
6. **Undo/Redo**: Action undo/redo functionality
7. **Templates**: Save common action configurations as templates

## Related Components
- **Requires**: Multiple UI components and MeshCanvas
- **Uses**: ApiProvider for mesh operations
- **Complex Integration**: Most sophisticated page in terms of component integration
