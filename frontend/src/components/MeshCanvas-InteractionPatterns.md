# MeshCanvas Interaction Patterns

## Overview

This document describes the comprehensive interaction patterns supported by the enhanced MeshCanvas component, including navigation, annotation management, and integration with mesh generation workflows.

## Navigation Interactions

### Mouse Interactions

#### Zoom Control
- **Mouse Wheel**: Scroll up/down to zoom in/out
- **Zoom Limits**: Configurable min/max zoom levels (default: 10% - 500%)
- **Zoom Center**: Zooms relative to current canvas center
- **Smooth Scaling**: Zoom increment of ±10% per wheel step

```jsx
<MeshCanvas
  enableZoom={true}
  minZoom={0.05}      // 5% minimum
  maxZoom={20.0}      // 2000% maximum
  onZoomChange={(zoom) => console.log(`Zoom: ${(zoom * 100).toFixed(0)}%`)}
/>
```

#### Pan Navigation
- **Left Click + Drag**: Pan the canvas view
- **Visual Feedback**: Cursor changes to indicate pan mode
- **Smooth Movement**: Real-time pan updates during drag
- **Boundary Handling**: No artificial boundaries, infinite pan space

```jsx
<MeshCanvas
  enablePan={true}
  onPanChange={(offset) => console.log(`Pan: ${offset.x}, ${offset.y}`)}
/>
```

#### Click Interactions
- **Single Click**: Get world coordinates at click position
- **Interaction State**: Clicks ignored during pan/zoom operations
- **Coordinate Precision**: High-precision coordinate calculation with DPI compensation
- **Event Bubbling**: Click events properly handled with overlay annotations

### Keyboard Shortcuts (Future Enhancement)

The following keyboard shortcuts are planned for future implementation:

- **Space + Click**: Alternative pan mode
- **Ctrl + Mouse Wheel**: Faster zoom (20% increments)
- **R**: Reset view to default zoom/pan
- **F**: Fit content to view
- **G**: Toggle grid visibility

## Annotation Interactions

### Annotation Types

#### Static Markers
Non-interactive visual markers for reference points, measurements, etc.

```javascript
const staticMarker = {
  position: [x, y],
  content: '<div class="static-marker">📍</div>',
  interactive: false,
  type: 'static'
};
```

#### Interactive Labels
Clickable labels with hover effects and custom actions.

```javascript
const interactiveLabel = {
  position: [x, y],
  content: '<div class="interactive-label">Click me</div>',
  interactive: true,
  type: 'label',
  onClick: (annotation, event) => {
    console.log('Label clicked:', annotation.position);
    event.stopPropagation(); // Prevent canvas click
  }
};
```

#### Data Popups
Rich information displays with formatted content.

```javascript
const dataPopup = {
  position: [x, y],
  content: `
    <div class="data-popup">
      <h4>Vertex Information</h4>
      <p>Quality: <strong>0.95</strong></p>
      <p>Angle: <strong>87°</strong></p>
      <button onclick="editVertex()">Edit</button>
    </div>
  `,
  interactive: true,
  type: 'popup',
  zIndex: 100
};
```

### Annotation Management Patterns

#### Dynamic Annotation Updates

```jsx
const MeshViewer = () => {
  const [annotations, setAnnotations] = useState([]);
  const canvasRef = useRef();

  const addAnnotation = (worldCoords, content) => {
    const newAnnotation = {
      id: Date.now(),
      position: worldCoords,
      content,
      interactive: true,
      onClick: (ann) => removeAnnotation(ann.id)
    };
    
    setAnnotations(prev => [...prev, newAnnotation]);
  };

  const removeAnnotation = (id) => {
    setAnnotations(prev => prev.filter(ann => ann.id !== id));
  };

  return (
    <MeshCanvas
      ref={canvasRef}
      showOverlay={true}
      annotations={annotations}
      onCanvasClick={(coords) => coords && addAnnotation(coords, '📍')}
    />
  );
};
```

#### Annotation Filtering and Grouping

```jsx
const AdvancedAnnotationViewer = () => {
  const [allAnnotations, setAllAnnotations] = useState([]);
  const [visibleTypes, setVisibleTypes] = useState(['marker', 'label']);

  const filteredAnnotations = useMemo(() =>
    allAnnotations.filter(ann => visibleTypes.includes(ann.type)),
    [allAnnotations, visibleTypes]
  );

  return (
    <div>
      <AnnotationControls 
        types={['marker', 'label', 'popup', 'measurement']}
        visible={visibleTypes}
        onChange={setVisibleTypes}
      />
      <MeshCanvas
        showOverlay={true}
        annotations={filteredAnnotations}
      />
    </div>
  );
};
```

## Mesh Generation Workflow Patterns

### Reference Point Selection

```jsx
const ReferencePointSelector = () => {
  const [referencePoint, setReferencePoint] = useState(null);
  const [boundaryVertices, setBoundaryVertices] = useState([]);

  const handleClick = (worldCoords) => {
    if (worldCoords) {
      setReferencePoint(worldCoords);
    }
  };

  const annotations = useMemo(() => {
    const anns = [];
    
    // Add reference point annotation
    if (referencePoint) {
      anns.push({
        position: referencePoint,
        content: '<div class="reference-point">🎯 Reference</div>',
        type: 'reference'
      });
    }
    
    return anns;
  }, [referencePoint]);

  return (
    <MeshCanvas
      onCanvasClick={handleClick}
      showOverlay={true}
      annotations={annotations}
    />
  );
};
```

### Interactive Mesh Editing

```jsx
const InteractiveMeshEditor = () => {
  const [meshData, setMeshData] = useState({});
  const [selectedVertex, setSelectedVertex] = useState(null);
  const canvasRef = useRef();

  const handleVertexClick = (worldCoords) => {
    // Find closest vertex to click
    const closestVertex = findClosestVertex(worldCoords, meshData);
    setSelectedVertex(closestVertex);
  };

  const moveVertex = (oldPos, newPos) => {
    const updatedMesh = { ...meshData };
    // Update mesh data with new vertex position
    // ... mesh update logic
    setMeshData(updatedMesh);
    
    // Re-render scene
    canvasRef.current?.renderScene(updatedMesh, boundaryVertices);
  };

  return (
    <MeshCanvas
      ref={canvasRef}
      onCanvasClick={handleVertexClick}
      enablePan={true}
      enableZoom={true}
    />
  );
};
```

## Performance Optimization Patterns

### Lazy Annotation Rendering

For large numbers of annotations, implement viewport culling:

```jsx
const OptimizedAnnotationRenderer = () => {
  const [allAnnotations, setAllAnnotations] = useState([]);
  const [viewBounds, setViewBounds] = useState(null);

  const visibleAnnotations = useMemo(() => {
    if (!viewBounds) return allAnnotations;
    
    return allAnnotations.filter(annotation => 
      isInViewport(annotation.position, viewBounds)
    );
  }, [allAnnotations, viewBounds]);

  const handleViewChange = (zoom, pan) => {
    // Calculate current viewport bounds
    const bounds = calculateViewBounds(zoom, pan);
    setViewBounds(bounds);
  };

  return (
    <MeshCanvas
      annotations={visibleAnnotations}
      onZoomChange={handleViewChange}
      onPanChange={handleViewChange}
    />
  );
};
```

### Debounced Updates

For real-time data updates, use debouncing:

```jsx
const RealTimeMeshViewer = () => {
  const [meshData, setMeshData] = useState(null);
  const canvasRef = useRef();

  const debouncedRender = useMemo(
    () => debounce((data) => {
      canvasRef.current?.renderScene(data.meshData, data.boundaryVertices);
    }, 100),
    []
  );

  useEffect(() => {
    const interval = setInterval(() => {
      fetchMeshData().then(debouncedRender);
    }, 50); // High frequency updates

    return () => clearInterval(interval);
  }, [debouncedRender]);

  return <MeshCanvas ref={canvasRef} />;
};
```

## Integration Patterns

### API Integration

```jsx
const APIIntegratedMeshCanvas = () => {
  const canvasRef = useRef();
  const [apiClient] = useState(() => new MeshAPIClient());

  const handleAction = async (worldCoords) => {
    try {
      const result = await apiClient.executeAction({
        coordinates: worldCoords,
        action: 'add_vertex'
      });

      // Update visualization with API response
      if (result.success) {
        canvasRef.current?.renderScene(
          result.updatedMesh, 
          result.boundaryVertices,
          result.referencePoint
        );
      }
    } catch (error) {
      console.error('API action failed:', error);
    }
  };

  return (
    <MeshCanvas
      ref={canvasRef}
      onCanvasClick={handleAction}
      enableZoom={true}
      enablePan={true}
    />
  );
};
```

### State Management Integration

```jsx
// With Redux/Context
const ReduxConnectedMeshCanvas = () => {
  const dispatch = useDispatch();
  const { meshData, boundaryData, annotations } = useSelector(selectMeshState);
  const canvasRef = useRef();

  const handleCanvasClick = (worldCoords) => {
    dispatch(addMeshPoint(worldCoords));
  };

  const handleZoomChange = (zoom) => {
    dispatch(setViewportZoom(zoom));
  };

  useEffect(() => {
    if (meshData && boundaryData) {
      canvasRef.current?.renderScene(meshData, boundaryData);
    }
  }, [meshData, boundaryData]);

  return (
    <MeshCanvas
      ref={canvasRef}
      onCanvasClick={handleCanvasClick}
      onZoomChange={handleZoomChange}
      annotations={annotations}
      showOverlay={annotations.length > 0}
    />
  );
};
```

## Accessibility Considerations

### Keyboard Navigation
- Implement tab navigation for annotations
- Provide keyboard shortcuts for common actions
- Support screen reader announcements for state changes

### Visual Indicators
- High contrast mode support
- Configurable color schemes
- Clear visual feedback for all interactions

### Touch Support
- Touch-friendly annotation sizes
- Gesture support for pan/zoom
- Haptic feedback on mobile devices

## Testing Patterns

### Unit Testing Interactions

```javascript
describe('MeshCanvas Interactions', () => {
  test('handles zoom interaction', () => {
    const onZoomChange = jest.fn();
    const { container } = render(
      <MeshCanvas enableZoom={true} onZoomChange={onZoomChange} />
    );
    
    const canvas = container.querySelector('canvas');
    fireEvent.wheel(canvas, { deltaY: -100 });
    
    expect(onZoomChange).toHaveBeenCalledWith(expect.any(Number));
  });

  test('handles annotation clicks', () => {
    const onAnnotationClick = jest.fn();
    const annotations = [{
      position: [50, 50],
      content: '<div>Test</div>',
      onClick: onAnnotationClick,
      interactive: true
    }];
    
    render(<MeshCanvas showOverlay={true} annotations={annotations} />);
    
    // Test annotation click handling
    // ... test implementation
  });
});
```

### Integration Testing

```javascript
describe('MeshCanvas Integration', () => {
  test('coordinates properly with API calls', async () => {
    const apiMock = new MockAPIClient();
    const component = render(<APIIntegratedMeshCanvas api={apiMock} />);
    
    // Simulate click
    fireEvent.click(component.canvas, { clientX: 100, clientY: 100 });
    
    // Verify API was called with correct coordinates
    await waitFor(() => {
      expect(apiMock.executeAction).toHaveBeenCalledWith(
        expect.objectContaining({
          coordinates: expect.any(Array)
        })
      );
    });
  });
});
```

This comprehensive interaction pattern documentation ensures developers can effectively implement complex mesh generation workflows with the enhanced MeshCanvas component.
