# MeshCanvas and CanvasRenderer Technical Reference

## Overview

This technical reference documents the `MeshCanvas` React component and its underlying `CanvasRenderer` class, which provide a complete solution for visualizing mesh data, boundary previews, and training interactions in the RL Mesh Generation application.

## Architecture

```
MeshCanvas (React Component)
    │
    ├── Props API (className, style, onCanvasClick, ...canvasProps)
    │
    ├── useRef Management
    │   ├── canvasRef - DOM canvas element
    │   ├── rendererRef - CanvasRenderer instance
    │   └── resizeCleanupRef - Cleanup function for resize events
    │
    ├── Event Handling
    │   ├── Canvas click events with coordinate conversion
    │   ├── Window resize events (debounced)
    │   └── Component lifecycle cleanup
    │
    └── Imperative API (via forwardRef)
        ├── Core Rendering Methods
        ├── Coordinate Transforms
        └── Canvas Control Methods
            │
            └── CanvasRenderer (Core Engine)
                ├── Canvas Setup & Configuration
                ├── Coordinate System Management
                ├── Rendering Pipeline
                ├── Event Management
                └── Memory & Cleanup
```

---

## MeshCanvas API Reference

### Initialization and Props

#### Constructor/Initialization
```jsx
import MeshCanvas from './components/MeshCanvas';

// Basic usage
<MeshCanvas
  ref={canvasRef}
  onCanvasClick={handleCanvasClick}
  className="my-canvas"
  style={{ width: '800px', height: '600px' }}
/>
```

#### Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | string | `''` | CSS classes applied to canvas element |
| `style` | object | `{}` | Inline styles for canvas element |
| `onCanvasClick` | function | `null` | Callback for canvas click events |
| `...canvasProps` | object | | Additional HTML5 canvas props |

#### onCanvasClick Callback Signature
```javascript
onCanvasClick(worldCoords, event)
```
- `worldCoords`: `[x, y]` array in world coordinates, or `null` if no valid transform
- `event`: Native MouseEvent object from the click

---

### Exposed Imperative Methods

The MeshCanvas component exposes the following methods via `forwardRef`:

#### Core Rendering Methods

##### `clearCanvas()`
Clears the entire canvas and resets to empty state with grid background.

```javascript
// Clear the canvas completely
canvasRef.current.clearCanvas();
```

**Behavior:**
- Removes all rendered content
- Draws background grid
- Shows "waiting for data" text
- Resets internal transform state
- Clears cached render data

##### `renderBoundaryPreview(boundaryVertices, meshName?)`
Renders a preview of mesh boundary vertices.

```javascript
// Render boundary preview
canvasRef.current.renderBoundaryPreview(
  [[0, 0], [100, 0], [100, 100], [0, 100]], // boundary vertices
  'Square Mesh' // optional mesh name
);
```

**Parameters:**
- `boundaryVertices`: Array of `[x, y]` coordinate pairs
- `meshName`: Optional string for display title

**Behavior:**
- Calculates optimal view transform for boundary data
- Renders boundary as connected line segments (red)
- Highlights boundary vertices as circles
- Caches data for resize operations
- Updates internal transform state

##### `renderScene(meshData, boundaryVertices, refPointInfo?)`
Renders complete scene with mesh data, boundary, and reference points.

```javascript
// Render full scene
canvasRef.current.renderScene(
  meshData,           // mesh connectivity data
  boundaryVertices,   // boundary vertex array  
  refPointInfo        // reference point information
);
```

**Parameters:**
- `meshData`: Object with vertex coordinates as keys and adjacent vertices as values
- `boundaryVertices`: Array of `[x, y]` boundary coordinates
- `refPointInfo`: Object containing reference point data (optional)

**Behavior:**
- Collects all vertices for optimal transform calculation
- Renders mesh edges and vertices (blue)
- Renders boundary lines and vertices (red)
- Renders reference points and clicked points (green/pink)
- Applies adaptive sizing based on data density
- Caches complete scene for resize operations

#### Coordinate Transformation Methods

##### `getCurrentTransform()`
Returns current coordinate transformation parameters.

```javascript
const transform = canvasRef.current.getCurrentTransform();
// Returns: { scale, offsetX, offsetY } or null
```

**Returns:**
- `transform.scale`: Scale factor from world to screen coordinates
- `transform.offsetX`: X-axis offset in screen pixels
- `transform.offsetY`: Y-axis offset in screen pixels
- `null`: If no valid transform is available

##### `screenToWorld(screenX, screenY)`
Converts screen coordinates to world coordinates.

```javascript
const [worldX, worldY] = canvasRef.current.screenToWorld(100, 150);
```

**Parameters:**
- `screenX`: X coordinate in screen/canvas pixels
- `screenY`: Y coordinate in screen/canvas pixels

**Returns:**
- `[worldX, worldY]`: Array with world coordinates
- `[0, 0]`: If no valid transform is available

##### `worldToScreen(worldCoords)`
Converts world coordinates to screen coordinates.

```javascript
const [screenX, screenY] = canvasRef.current.worldToScreen([10.5, 20.3]);
```

**Parameters:**
- `worldCoords`: `[x, y]` array with world coordinates

**Returns:**
- `[screenX, screenY]`: Array with screen pixel coordinates
- `[0, 0]`: If no valid transform is available

---

### Resize Handling

The MeshCanvas component automatically handles resize events through multiple mechanisms:

#### Window Resize Events
- Bound through `CanvasRenderer.bindResizeEvent()`
- Debounced with 150ms delay for performance
- Monitors `devicePixelRatio` changes for high-DPI displays
- Automatically re-renders cached content after resize

#### Container Resize Events  
- Uses `ResizeObserver` to detect container size changes
- Observes canvas parent element
- Triggers canvas recalculation when container dimensions change

#### Manual Resize Trigger
```javascript
// Manually trigger resize recalculation
canvasRef.current.onResize();
```

#### Resize Behavior
1. **Canvas Dimension Update**: Recalculates canvas pixel size and display size
2. **Device Pixel Ratio**: Accounts for high-DPI displays automatically
3. **Transform Recalculation**: Updates coordinate transformation for new dimensions
4. **Content Re-render**: Re-renders cached content with new transform
5. **Adaptive Sizing**: Adjusts vertex/line sizes based on new scale factors

---

### Event Handling

#### Canvas Click Events
The MeshCanvas component provides sophisticated click event handling:

```javascript
const handleCanvasClick = (worldCoords, event) => {
  if (worldCoords) {
    console.log('Clicked at world coordinates:', worldCoords);
    console.log('Screen coordinates:', event.clientX, event.clientY);
    
    // Use world coordinates for application logic
    sendActionToTrainingSystem(worldCoords);
  } else {
    console.log('Click occurred but no valid coordinate transform available');
  }
};

<MeshCanvas 
  ref={canvasRef}
  onCanvasClick={handleCanvasClick}
/>
```

**Event Processing:**
1. **Coordinate Calculation**: Automatically calculates mouse position relative to canvas
2. **Transform Application**: Converts screen coordinates to world coordinates using current transform
3. **Validation**: Only calls callback with valid coordinates if transform is available
4. **Event Pass-through**: Provides original MouseEvent for additional processing

---

## CanvasRenderer Technical Reference

### Class Structure and Initialization

#### Constructor
```javascript
const renderer = new CanvasRenderer(canvasElement);
```

**Parameters:**
- `canvasElement`: HTML5 Canvas DOM element

**Initialization Process:**
1. **Canvas Context Setup**: Gets 2D rendering context and stores references
2. **State Initialization**: Sets up transform cache, resize flags, and adaptive sizing
3. **Canvas Configuration**: Calls `setupCanvas()` for initial sizing and clearing
4. **Performance Optimization**: Initializes debounce timers for resize handling

#### Internal State Properties
```javascript
{
  canvas: HTMLCanvasElement,           // Canvas DOM element
  ctx: CanvasRenderingContext2D,       // 2D rendering context
  currentTransform: Object|null,       // Current coordinate transform
  isResizing: boolean,                 // Resize operation flag
  resizeDebounceTimer: number|null,    // Debounce timer ID
  lastRenderData: Object|null,         // Cached render data
  adaptiveSizes: Object                // Dynamic sizing parameters
}
```

---

### Rendering Entry Points

The CanvasRenderer provides three main rendering entry points:

#### `clearCanvas()`
**Purpose**: Complete canvas reset and cleanup
**Usage**: `renderer.clearCanvas()`

**Implementation Details:**
- Calculates logical canvas dimensions (accounting for device pixel ratio)
- Clears entire canvas area with `clearRect()`
- Draws subtle background grid for visual reference
- Renders centered "waiting for data" text
- Resets `currentTransform` to `null`
- Clears `lastRenderData` cache

**Performance Notes:**
- Optimized for different device pixel ratios
- Grid drawing skipped on very small canvas sizes

#### `renderBoundaryPreview(boundaryVertices, meshName)`
**Purpose**: Fast preview rendering for boundary data only
**Usage**: `renderer.renderBoundaryPreview(vertices, "Mesh Name")`

**Implementation Process:**
1. **Validation**: Checks for valid vertex array input
2. **Data Caching**: Stores boundary data and preview flag in `lastRenderData`
3. **Transform Calculation**: Calls `calculateTransform()` for optimal viewport
4. **Adaptive Sizing**: Calculates appropriate line widths and vertex sizes
5. **Boundary Rendering**: Draws connected boundary lines (red) and vertices
6. **Title Rendering**: Displays mesh name and vertex count

**Optimization Features:**
- Skips expensive mesh data processing
- Uses streamlined rendering pipeline
- Cached for instant re-rendering on resize

#### `renderScene(meshData, boundaryVertices, refPointInfo)`
**Purpose**: Complete scene rendering with all data types
**Usage**: `renderer.renderScene(meshData, boundaryData, refPoints)`

**Implementation Process:**
1. **Data Caching**: Stores complete scene data for resize operations
2. **Data Parsing**: Processes backend data formats with `parseBackendData()`
3. **Vertex Collection**: Combines all vertices from mesh and boundary data
4. **Transform Calculation**: Computes optimal viewport for all data
5. **Adaptive Sizing**: Calculates sizes based on total data density
6. **Layered Rendering**:
   - Mesh edges and vertices (blue)
   - Boundary lines and vertices (red)
   - Reference points and interactions (green/pink)

**Layering Strategy:**
- Mesh data rendered first (background layer)
- Boundary data rendered second (structural layer)
- Reference points rendered last (interaction layer)

---

### Coordinate Transform System

The coordinate transform system handles conversion between world coordinates (mesh data space) and screen coordinates (canvas pixels).

#### Transform Calculation (`calculateTransform`)

**Input**: Array of vertices in world coordinates
**Output**: Transform object `{ scale, offsetX, offsetY }`

**Algorithm:**
1. **Bounds Calculation**: Find min/max X and Y coordinates
2. **Data Dimensions**: Calculate world space width and height
3. **Canvas Dimensions**: Get logical canvas size (accounting for device pixel ratio)
4. **Scale Calculation**: Determine uniform scale factor to fit data with padding
5. **Centering**: Calculate offsets to center data in canvas

**Mathematical Implementation:**
```javascript
// Calculate data bounds
const minX = Math.min(...vertices.map(v => v[0]));
const maxX = Math.max(...vertices.map(v => v[0]));
const dataWidth = maxX - minX;

// Calculate scale (uniform for both axes)
const scaleX = (canvasWidth - 2 * padding) / (dataWidth || 1);
const scaleY = (canvasHeight - 2 * padding) / (dataHeight || 1);  
const scale = Math.min(scaleX, scaleY);

// Calculate centering offsets
const offsetX = (canvasWidth - dataWidth * scale) / 2 - minX * scale;
const offsetY = (canvasHeight - dataHeight * scale) / 2 - minY * scale;
```

#### Coordinate Conversion Methods

##### `worldToScreen(worldCoords, transform)`
Converts world coordinates to screen pixel coordinates.

**Formula:**
```javascript
screenX = worldX * transform.scale + transform.offsetX
screenY = worldY * transform.scale + transform.offsetY
```

##### `screenToWorld(screenX, screenY, transform)`
Converts screen pixel coordinates to world coordinates.

**Formula:**
```javascript
worldX = (screenX - transform.offsetX) / transform.scale
worldY = (screenY - transform.offsetY) / transform.scale
```

#### Transform Caching and Management

**State Management:**
- `currentTransform` stores active transform parameters
- Updated whenever new data is rendered
- Reset to `null` when canvas is cleared
- Used by coordinate conversion methods

**Thread Safety:**
- Transform calculations are synchronous
- No race conditions between render and coordinate conversion
- Guaranteed consistency within single render cycle

---

## CanvasRenderer Assumptions and Behavior

### Expected Input Formats

#### Mesh Data Format
```javascript
const meshData = {
  "[10.5, 20.3]": [[15.2, 18.7], [8.1, 25.4]],  // vertex -> adjacent vertices
  "[15.2, 18.7]": [[10.5, 20.3], [22.0, 16.1]],
  // ... more vertices
};
```

**Assumptions:**
- Keys are JSON-stringified coordinate arrays
- Values are arrays of adjacent vertex coordinates
- All coordinates are numeric `[x, y]` pairs
- Invalid entries are skipped with warnings

#### Boundary Vertices Format
```javascript
const boundaryVertices = [
  [0, 0], [100, 0], [100, 100], [0, 100]  // Connected boundary vertices
];
```

**Assumptions:**
- Array of `[x, y]` coordinate pairs
- Vertices form a closed boundary (last connects to first)
- Coordinates are in consistent world coordinate system

#### Reference Point Info Format
```javascript
const refPointInfo = {
  ref_vertex: [x, y],                    // Reference vertex coordinates
  clicked_point: [x, y],                 // User clicked coordinates (Type1 actions)
  local_env_vertices: [[x1, y1], ...],   // Local environment vertices
  new_element: [[x1, y1], [x2, y2], [x3, y3]]  // Generated triangular element
};
```

**Assumptions:**
- All coordinate fields are optional
- Invalid coordinates are silently skipped
- Reference points rendered with distinctive styling

### Rendering Behavior

#### Adaptive Sizing System
The renderer automatically adjusts visual element sizes based on:

**Data Density Factors:**
- Total vertex count (more vertices = smaller elements)
- Transform scale factor (higher zoom = larger elements)
- Canvas dimensions (larger canvas allows larger elements)

**Size Categories:**
```javascript
// Size multipliers based on vertex count
if (totalVertexCount > 500) sizeMultiplier = 0.4;
else if (totalVertexCount > 200) sizeMultiplier = 0.6;
else if (totalVertexCount > 100) sizeMultiplier = 0.8;
else if (totalVertexCount < 20) sizeMultiplier = 1.3;

// Scale factor adjustment
const scaleFactor = Math.max(0.5, Math.min(2.0, scale / 100));
```

**Applied Element Sizes:**
- `vertexRadius`: Mesh vertex circles (1.0 - 12px)
- `boundaryVertexRadius`: Boundary vertex circles (1.0 - 8px)  
- `boundaryLineWidth`: Boundary line thickness (0.5 - 8px)
- `meshVertexLineWidth`: Mesh edge thickness (0.3 - 4px)
- `referencePointRadius`: Reference point circles (2 - 16px)

#### Color Scheme and Visual Hierarchy
```javascript
// Mesh elements (background layer)
meshEdges: '#6366F1' (blue)
meshVertices: '#3B82F6' (blue), stroke '#1E40AF' (dark blue)

// Boundary elements (structural layer)  
boundaryLines: '#EF4444' (red)
boundaryVertices: '#DC2626' (dark red)

// Reference elements (interaction layer)
referencePoint: '#10B981' (green), stroke '#FFFFFF' (white)
clickedPoint: '#FF6B6B' (pink), stroke '#FFFFFF' (white)

// Background elements
grid: 'rgba(255, 255, 255, 0.08)' (subtle white)
waitingText: '#a0aec0' (gray)
```

#### Performance Optimizations

**Vertex Deduplication:**
- Uses `Set` to track already-drawn vertices
- Prevents duplicate rendering of shared vertices
- Significantly improves performance for dense meshes

**Resize Debouncing:**
- 150ms debounce delay for window resize events
- Prevents excessive re-rendering during window dragging
- Maintains cached data during debounce period

**High-DPI Display Support:**
- Automatic device pixel ratio detection
- Canvas resolution matches actual display resolution
- Maintains sharp rendering on Retina/4K displays

**Memory Management:**
- Render data cached only for resize operations
- Automatic cleanup in `destroy()` method
- Proper event listener cleanup

### Error Handling and Resilience

#### Data Validation
- Invalid coordinates are filtered out with `isValidCoordinate()`
- Malformed JSON keys are caught and logged as warnings
- Empty or null data sets handled gracefully

#### Rendering Errors
- Individual render operations wrapped in try-catch blocks  
- Failed vertex parsing logged as warnings, continues rendering
- Canvas operation failures logged as errors, attempt graceful recovery

#### Resource Cleanup
- Timer cleanup in `destroy()` method
- Event listener removal handled automatically
- Memory leak prevention through proper state reset

---

## Usage Examples and Best Practices

### Basic Usage Pattern
```javascript
import React, { useRef, useEffect } from 'react';
import MeshCanvas from './components/MeshCanvas';

const MyMeshViewer = () => {
  const canvasRef = useRef(null);
  
  const handleCanvasClick = (worldCoords, event) => {
    if (worldCoords) {
      console.log('World coordinates:', worldCoords);
      // Handle interaction logic here
    }
  };

  useEffect(() => {
    // Load and render initial data
    if (canvasRef.current) {
      canvasRef.current.renderBoundaryPreview(boundaryData, 'My Mesh');
    }
  }, []);

  return (
    <div style={{ width: '800px', height: '600px' }}>
      <MeshCanvas
        ref={canvasRef}
        onCanvasClick={handleCanvasClick}
        className="mesh-viewer"
      />
    </div>
  );
};
```

### Advanced Usage with API Integration
```javascript
const TrainingInterface = () => {
  const canvasRef = useRef(null);
  const [meshData, setMeshData] = useState(null);
  const [refPointInfo, setRefPointInfo] = useState(null);

  const executeTrainingAction = async (worldCoords) => {
    try {
      // Send coordinates to training API
      const response = await api.executeAction({
        coordinates: worldCoords,
        action_type: 'type1'
      });
      
      // Update reference point info with result
      setRefPointInfo(response.reference_point_info);
      
      // Re-render with new reference point data
      if (canvasRef.current) {
        canvasRef.current.renderScene(
          meshData,
          boundaryData, 
          response.reference_point_info
        );
      }
    } catch (error) {
      console.error('Training action failed:', error);
    }
  };

  return (
    <MeshCanvas
      ref={canvasRef}
      onCanvasClick={(coords) => coords && executeTrainingAction(coords)}
      style={{ cursor: 'crosshair' }}
    />
  );
};
```

### Performance Best Practices

#### Canvas Container Sizing
```css
.canvas-container {
  width: 100%;
  height: 600px; /* Explicit height required */
  position: relative;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}
```

#### Optimize Re-rendering
```javascript
// Use memoization for expensive data processing
const processedMeshData = useMemo(() => {
  return processMeshData(rawMeshData);
}, [rawMeshData]);

// Batch multiple rendering operations
useEffect(() => {
  if (canvasRef.current && processedMeshData && boundaryData) {
    // Single renderScene call instead of multiple operations
    canvasRef.current.renderScene(processedMeshData, boundaryData, refPointInfo);
  }
}, [processedMeshData, boundaryData, refPointInfo]);
```

#### Memory Management
```javascript
useEffect(() => {
  // Cleanup function to prevent memory leaks
  return () => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
  };
}, []);
```

---

## Error Handling and Debugging

### Common Issues and Solutions

#### Canvas Not Rendering
**Symptoms**: Blank canvas or "waiting for data" message
**Causes**: 
- Container has zero dimensions
- Invalid data format passed to render methods
- Canvas context initialization failed

**Solutions**:
- Ensure parent container has explicit width/height
- Validate data format before passing to render methods
- Check browser console for initialization errors

#### Coordinate Conversion Issues  
**Symptoms**: Click coordinates don't match visual positions
**Causes**:
- Canvas not properly sized relative to display
- Device pixel ratio not handled correctly
- Transform calculated before canvas resize complete

**Solutions**:
- Wait for component mount before coordinate operations
- Trigger manual resize after dynamic sizing: `canvasRef.current?.onResize()`
- Validate transform exists before coordinate conversion

#### Performance Issues
**Symptoms**: Slow rendering, browser freezing during resize
**Causes**:
- Very large datasets overwhelming rendering pipeline
- Excessive resize events without proper debouncing
- Memory leaks from improper cleanup

**Solutions**:
- Implement data pagination for large meshes
- Verify resize debouncing is working (150ms delay)
- Ensure proper component cleanup in `useEffect` return functions

### Debug Logging
```javascript
// Enable debug logging for coordinate transforms
const debugCoordinateTransform = (canvasRef) => {
  const transform = canvasRef.current?.getCurrentTransform();
  console.log('Current transform:', transform);
  
  if (transform) {
    const testWorld = [10, 20];
    const screen = canvasRef.current.worldToScreen(testWorld);
    const backToWorld = canvasRef.current.screenToWorld(screen[0], screen[1]);
    
    console.log('World -> Screen -> World test:');
    console.log('Original:', testWorld);
    console.log('Screen:', screen);  
    console.log('Back to world:', backToWorld);
    console.log('Accuracy error:', Math.abs(testWorld[0] - backToWorld[0]));
  }
};
```

This comprehensive technical reference provides complete coverage of the MeshCanvas and CanvasRenderer APIs, their behavior patterns, expected inputs, and implementation details necessary for effective integration and debugging.
