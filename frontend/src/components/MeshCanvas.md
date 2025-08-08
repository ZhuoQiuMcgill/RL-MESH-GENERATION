# MeshCanvas React Component - Enhanced

## Overview

`MeshCanvas` is an enhanced React wrapper component that provides high-fidelity, responsive mesh visualization with comprehensive interaction support. This component features improved high-DPI scaling, zoom/pan capabilities, overlay annotations, and extensive customization options for professional mesh generation applications.

## Enhanced Features

### Core Capabilities
- **High-DPI Scaling**: Automatic device pixel ratio detection and compensation for crisp rendering on all displays
- **Responsive Design**: Proper ResizeObserver integration with contentBoxSize support for accurate measurements
- **React Integration**: Uses modern React hooks with proper lifecycle management and cleanup
- **Memory Management**: Comprehensive cleanup of event listeners, observers, and canvas resources

### Interaction & Navigation
- **Zoom Control**: Mouse wheel zoom with configurable min/max limits
- **Pan Navigation**: Click-and-drag panning with visual feedback
- **Click Interaction**: Enhanced click handling with zoom/pan coordinate compensation
- **Touch Support**: Touch-action CSS properties for mobile device compatibility

### Visual Enhancements
- **Customizable Background**: Configurable background colors with smart grid opacity
- **Grid Overlay**: Optional grid with high-DPI scaling and adaptive opacity
- **Annotation Layer**: HTML overlay system for UI annotations and markers
- **Visual Indicators**: Zoom level display and interactive state feedback

### Performance & Quality
- **Adaptive Sizing**: Dynamic element sizing based on data density and zoom level
- **Render Caching**: Intelligent caching of render data for efficient resize operations
- **High-Quality Rendering**: Enhanced canvas context settings for crisp lines and shapes
- **Debounced Operations**: Optimized resize and interaction handling

## Basic Usage

```jsx
import React, { useRef, useState } from 'react';
import MeshCanvas from './components/MeshCanvas';

const MyComponent = () => {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1.0);
  const [annotations, setAnnotations] = useState([]);

  const handleCanvasClick = (worldCoords, event) => {
    if (worldCoords) {
      console.log('Clicked at world coordinates:', worldCoords);
      
      // Add annotation at click point
      setAnnotations(prev => [...prev, {
        position: worldCoords,
        content: `<div class="annotation-marker">Point ${prev.length + 1}</div>`,
        type: 'marker',
        interactive: true,
        onClick: (annotation) => console.log('Annotation clicked:', annotation)
      }]);
    }
  };

  const loadMeshData = () => {
    // Example boundary vertices
    const boundaryVertices = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100]
    ];

    canvasRef.current?.renderBoundaryPreview(boundaryVertices, 'Square Mesh');
  };

  return (
    <div>
      <div className="controls">
        <button onClick={loadMeshData}>Load Mesh</button>
        <button onClick={() => canvasRef.current?.resetView()}>Reset View</button>
        <span>Zoom: {(zoom * 100).toFixed(0)}%</span>
      </div>
      
      <div style={{ width: '800px', height: '600px' }}>
        <MeshCanvas
          ref={canvasRef}
          onCanvasClick={handleCanvasClick}
          onZoomChange={setZoom}
          backgroundColor="#1a1a2e"
          showGrid={true}
          enableZoom={true}
          enablePan={true}
          minZoom={0.1}
          maxZoom={10.0}
          showOverlay={true}
          annotations={annotations}
          className="enhanced-mesh-canvas"
        />
      </div>
    </div>
  );
};
```

## Enhanced Props

### Core Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | string | `''` | CSS classes for the container element |
| `style` | object | `{}` | Inline styles for the container |
| `onCanvasClick` | function | `null` | Click handler with world coordinates |

### Visual & Styling Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `backgroundColor` | string | `'transparent'` | Canvas background color |
| `showGrid` | boolean | `true` | Show/hide grid overlay |
| `devicePixelRatio` | number | `null` | Override device pixel ratio (auto-detected if null) |

### Interaction Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `enableZoom` | boolean | `true` | Enable mouse wheel zoom |
| `enablePan` | boolean | `true` | Enable click-drag panning |
| `minZoom` | number | `0.1` | Minimum zoom level (10%) |
| `maxZoom` | number | `5.0` | Maximum zoom level (500%) |
| `onZoomChange` | function | `null` | Callback for zoom level changes |
| `onPanChange` | function | `null` | Callback for pan offset changes |

### Overlay Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `showOverlay` | boolean | `false` | Enable annotation overlay layer |
| `annotations` | array | `[]` | Array of annotation objects |

### Legacy Props
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `...canvasProps` | object | | Additional props passed to canvas element |

### Enhanced Callback Functions

#### onCanvasClick Callback
```javascript
onCanvasClick(worldCoords, event) {
  // worldCoords: [x, y] array in world coordinates with zoom/pan compensation
  // event: Original mouse click event
  // Note: worldCoords is null if no transform available or during interactions
}
```

#### onZoomChange Callback
```javascript
onZoomChange(zoomLevel) {
  // zoomLevel: Current zoom level (1.0 = 100%)
  // Called when zoom changes via mouse wheel or programmatically
}
```

#### onPanChange Callback
```javascript
onPanChange(panOffset) {
  // panOffset: { x: number, y: number } - pixel offset from center
  // Called during pan operations
}
```

## Annotation System

The enhanced MeshCanvas includes a powerful annotation overlay system for adding UI elements that follow world coordinates.

### Annotation Object Format
```javascript
const annotation = {
  position: [x, y],              // World coordinates
  content: '<div>Label</div>',   // HTML string or DOM element
  type: 'marker',               // CSS class suffix: mesh-canvas-annotation-marker
  interactive: true,            // Enable pointer events
  zIndex: 10,                   // Stacking order
  style: 'color: red;',         // Additional CSS styles
  onClick: (annotation, event) => {}, // Click handler
};
```

### Common Annotation Examples
```jsx
// Reference point marker
const refPointAnnotation = {
  position: [50, 30],
  content: '<div class="ref-point">🎯</div>',
  type: 'reference',
  interactive: false,
  style: 'font-size: 20px;'
};

// Interactive label
const labelAnnotation = {
  position: [0, 0],
  content: '<div class="vertex-label">Origin</div>',
  type: 'label',
  interactive: true,
  onClick: (ann) => console.log('Label clicked')
};

// Data visualization
const dataAnnotation = {
  position: [75, 45],
  content: `<div class="data-popup">
    <strong>Vertex Info</strong><br>
    Quality: 0.95<br>
    Angle: 87°
  </div>`,
  type: 'popup',
  zIndex: 20
};
```

## Enhanced Imperative API

The component exposes an extensive API via ref for programmatic control:

### Core Rendering Methods

```javascript
// Clear the canvas
canvasRef.current.clearCanvas();

// Render boundary preview
canvasRef.current.renderBoundaryPreview(boundaryVertices, meshName);

// Render full scene with mesh data, boundary, and reference points
canvasRef.current.renderScene(meshData, boundaryVertices, refPointInfo);
```

### Enhanced Coordinate Transformation

```javascript
// Get current transformation parameters
const transform = canvasRef.current.getCurrentTransform();

// Convert screen coordinates to world coordinates (with zoom/pan compensation)
const worldCoords = canvasRef.current.screenToWorld(screenX, screenY);

// Convert world coordinates to screen coordinates (with zoom/pan compensation)
const screenCoords = canvasRef.current.worldToScreen([x, y]);

// Manual coordinate transformation (advanced usage)
const manualWorldCoords = canvasRef.current.screenToWorld(
  screenX, screenY, transform, { zoom: 1.5, pan: { x: 10, y: 20 } }
);
```

### Enhanced Canvas Control

```javascript
// Manually trigger resize (with optional dimensions)
canvasRef.current.onResize();
canvasRef.current.onResize({ width: 800, height: 600, devicePixelRatio: 2.0 });

// Zoom and pan controls
canvasRef.current.setZoom(1.5);              // Set zoom to 150%
canvasRef.current.setPan({ x: 50, y: -30 }); // Pan offset in pixels
canvasRef.current.resetView();               // Reset to default view

const currentZoom = canvasRef.current.getZoom();
const currentPan = canvasRef.current.getPan();

// Overlay management
canvasRef.current.updateOverlay();           // Force overlay refresh

// Access underlying elements
const renderer = canvasRef.current.getRenderer();
const canvas = canvasRef.current.getCanvas();
const overlay = canvasRef.current.getOverlay();
const container = canvasRef.current.getContainer();

// Get complete state information
const state = canvasRef.current.getState();
// Returns: { zoom, pan, isInteracting, showGrid, showOverlay }
```

## Data Formats

### Boundary Vertices
```javascript
const boundaryVertices = [
  [x1, y1],
  [x2, y2],
  [x3, y3],
  // ... more vertices
];
```

### Mesh Data
```javascript
const meshData = {
  "[x1,y1]": [[x2,y2], [x3,y3]], // Adjacent vertices
  "[x2,y2]": [[x1,y1], [x4,y4]],
  // ... more vertices and their connections
};
```

### Reference Point Info
```javascript
const refPointInfo = {
  ref_vertex: [x, y],           // Reference point coordinates
  clicked_point: [x, y],        // For Type1 actions
  local_env_vertices: [...],    // Local environment vertices
  new_element: [...]            // Generated mesh element
};
```

## Styling

The canvas automatically adapts to its container size. Ensure the parent container has explicit dimensions:

```css
.canvas-container {
  width: 800px;
  height: 600px;
  border: 1px solid #ccc;
  border-radius: 8px;
  overflow: hidden;
}
```

## Advanced Usage Example

```jsx
import React, { useRef, useEffect, useState } from 'react';
import MeshCanvas from './components/MeshCanvas';

const AdvancedMeshViewer = () => {
  const canvasRef = useRef(null);
  const [meshData, setMeshData] = useState(null);
  const [boundaryData, setBoundaryData] = useState(null);

  // Load data from API
  useEffect(() => {
    fetch('/api/mesh/data')
      .then(res => res.json())
      .then(data => {
        setMeshData(data.meshData);
        setBoundaryData(data.boundaryVertices);
      });
  }, []);

  // Auto-render when data changes
  useEffect(() => {
    if (canvasRef.current && meshData && boundaryData) {
      canvasRef.current.renderScene(meshData, boundaryData);
    }
  }, [meshData, boundaryData]);

  const handleClick = (worldCoords, event) => {
    // Handle training interactions
    if (worldCoords) {
      // Send coordinates to training system
      fetch('/api/training/action', {
        method: 'POST',
        body: JSON.stringify({ coordinates: worldCoords })
      });
    }
  };

  return (
    <div className="mesh-viewer">
      <div className="canvas-container">
        <MeshCanvas
          ref={canvasRef}
          onCanvasClick={handleClick}
          className="training-canvas"
          style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)' }}
        />
      </div>
    </div>
  );
};
```

## Error Handling

The component includes built-in error handling for:

- Canvas initialization failures
- Invalid coordinate data
- Resize operation errors
- Event listener cleanup

All errors are logged to the console and don't crash the component.

## Performance Considerations

- The canvas automatically handles high-DPI displays
- Resize operations are debounced (150ms) for performance
- Large datasets are rendered with adaptive sizing to maintain performance
- Memory is properly cleaned up on component unmount

## Browser Compatibility

- Requires modern browsers with Canvas 2D API support
- Uses ResizeObserver (polyfill may be needed for older browsers)
- Supports touch devices with proper event handling
