# MeshCanvas Component

## Overview
A React wrapper component around the CanvasRenderer class that provides mesh visualization capabilities. Uses forwardRef to expose imperative methods for parent components to control rendering operations directly.

## File Location
`frontend/src/components/MeshCanvas.jsx`

## Props
| Prop Name | Type | Default | Required | Purpose |
|-----------|------|---------|----------|---------|
| `className` | string | `''` | No | Additional CSS classes for the canvas element |
| `style` | Object | `{}` | No | Inline styles for the canvas element |
| `onCanvasClick` | Function | `null` | No | Callback function for canvas click events |
| `...canvasProps` | any | - | No | Additional props passed to the canvas element |

## State Usage
This component does not manage local React state - it uses refs to maintain canvas renderer state.

## Dependencies

### React Dependencies
- `useRef` - For canvas element and renderer references
- `useEffect` - For initialization and cleanup
- `forwardRef` - To expose imperative methods to parent components
- `useImperativeHandle` - To define the imperative API

### Internal Dependencies
- `CanvasRenderer` from `'../utils/CanvasRenderer.js'` - Core canvas rendering engine

### External Dependencies
- None (canvas element is native HTML5)

## Side Effects

### Canvas Initialization
- **Renderer Creation**: Instantiates CanvasRenderer with the canvas element
- **Event Binding**: Binds window resize events for responsive canvas
- **Click Handler**: Attaches click event listener if `onCanvasClick` is provided

### Coordinate Transformation
- **Screen to World**: Converts mouse click coordinates to mesh world coordinates
- **World to Screen**: Converts mesh coordinates to screen pixel coordinates

### Cleanup Operations
- **Renderer Destruction**: Calls `renderer.destroy()` on unmount
- **Event Cleanup**: Removes resize and click event listeners
- **Reference Cleanup**: Clears renderer and cleanup references

## Imperative API

### Core Rendering Methods
| Method | Parameters | Purpose |
|--------|------------|---------|
| `clearCanvas()` | None | Clear the canvas completely |
| `renderBoundaryPreview(vertices, meshName)` | `vertices`: Array, `meshName`: string | Render mesh boundary preview |
| `renderScene(meshData, boundaryVertices, refPointInfo)` | `meshData`: Object, `boundaryVertices`: Array, `refPointInfo`: Object | Render complete scene |

### Coordinate Transformation Methods
| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `getCurrentTransform()` | None | Transform object | Get current canvas transform |
| `screenToWorld(screenX, screenY)` | `screenX`: number, `screenY`: number | `[x, y]` | Convert screen to world coordinates |
| `worldToScreen(worldCoords)` | `worldCoords`: `[x, y]` | `[x, y]` | Convert world to screen coordinates |

### Utility Methods
| Method | Parameters | Purpose |
|--------|------------|---------|
| `onResize()` | None | Manually trigger canvas resize |
| `getRenderer()` | None | Get underlying CanvasRenderer instance |
| `getCanvas()` | None | Get canvas DOM element |

## Event Handling

### Canvas Click Events
- **Event Processing**: Captures click events and converts to world coordinates
- **Boundary Checking**: Validates click coordinates before calling callback
- **Error Handling**: Gracefully handles coordinate transformation failures

### Resize Handling
- **Automatic Resize**: Uses ResizeObserver to detect canvas container size changes
- **Manual Resize**: Exposes `onResize()` method for manual triggering
- **Responsive Design**: Maintains proper canvas dimensions on window resize

## Canvas Integration

### CanvasRenderer Interface
Wraps the CanvasRenderer class methods:
- `renderBoundaryPreview()` - For mesh boundary visualization
- `renderScene()` - For complete scene rendering
- `clearCanvas()` - For canvas clearing
- `getCurrentTransform()` - For coordinate system access

### Error Handling
- **Initialization Errors**: Catches and logs CanvasRenderer creation failures
- **Runtime Errors**: Graceful handling of rendering operation failures
- **Null Checks**: Validates renderer existence before method calls

## CSS and Styling

### Default Styles
```css
display: block;
max-width: 100%;
max-height: 100%;
cursor: crosshair; /* when onCanvasClick is provided */
```

### Custom Styling
- **className**: Merged with default `mesh-canvas` class
- **style**: Merged with default styles, allows overrides
- **Dynamic Cursor**: Changes to crosshair when click handler is provided

## Usage Patterns

### Basic Usage
```jsx
const MyComponent = () => {
  const canvasRef = useRef(null);

  const loadMesh = () => {
    if (canvasRef.current) {
      canvasRef.current.renderBoundaryPreview(vertices, 'mesh1');
    }
  };

  return (
    <MeshCanvas 
      ref={canvasRef}
      onCanvasClick={(worldCoords) => console.log(worldCoords)}
    />
  );
};
```

### Advanced Usage with Controls
```jsx
const AdvancedMeshViewer = () => {
  const canvasRef = useRef(null);

  const clearCanvas = () => canvasRef.current?.clearCanvas();
  const renderFullScene = () => {
    canvasRef.current?.renderScene(meshData, boundaries, refPoint);
  };

  return (
    <div>
      <div>
        <button onClick={clearCanvas}>Clear</button>
        <button onClick={renderFullScene}>Render</button>
      </div>
      <MeshCanvas 
        ref={canvasRef}
        className="w-full h-96 border"
        onCanvasClick={handleCanvasClick}
      />
    </div>
  );
};
```

## Known Issues
1. **Memory Management**: Canvas resources not fully cleaned up in some scenarios
2. **Performance**: No throttling for resize events could cause performance issues
3. **Error Boundaries**: Rendering errors not caught by React error boundaries
4. **Mobile Support**: Touch events not explicitly handled for mobile interaction
5. **Accessibility**: No ARIA labels or keyboard navigation support

## Performance Considerations
1. **Ref Stability**: Uses stable refs to prevent unnecessary re-renders
2. **Event Listener Management**: Properly adds and removes event listeners
3. **Render Throttling**: No built-in throttling for frequent render calls
4. **Memory Usage**: Canvas contexts can consume significant memory

## Potential Improvements
1. **TypeScript**: Add TypeScript definitions for better type safety
2. **Error Boundaries**: Implement proper error boundary integration
3. **Touch Support**: Add mobile touch event handling
4. **Performance**: Add render call throttling and optimization
5. **Accessibility**: Add ARIA labels and keyboard interaction
6. **WebGL**: Consider WebGL backend for better performance
7. **Animation**: Add animation frame handling for smooth updates

## Related Components
- **Used by**: TrainingMonitor, History page, Action page, Generator page
- **Depends on**: CanvasRenderer utility class
- **Integration**: Core visualization component for all mesh-related features
