# Canvas Module

## Overview

The canvas module provides advanced 3D visualization capabilities for mesh rendering, interactive canvas controls, and visual analysis tools. It handles all aspects of mesh visualization and user interaction with 3D content.

## Public Surface

### Pages
- `pages/Canvas.jsx` - Main canvas interface with 3D mesh visualization and controls

### Components
- `components/MeshCanvas.jsx` - Core 3D mesh rendering component
- `components/MeshCanvasTest.jsx` - Testing component for canvas functionality
- `components/CanvasControls.jsx` - Interactive controls for canvas manipulation
- `components/ViewportControls.jsx` - Camera and viewport management controls

### Hooks
- `hooks/useMeshGenerator.js` - Mesh generation and rendering logic
- `hooks/useCanvasControls.js` - Canvas interaction and control state
- `hooks/useViewport.js` - Viewport and camera management

### Services
- `services/canvasRenderer.js` - Core rendering engine and utilities
- `services/meshLoader.js` - Mesh data loading and processing
- `services/canvasExporter.js` - Canvas content export functionality

## Module Interface

### Exports
```javascript
// Pages
export { default as CanvasPage } from './pages/Canvas'

// Components
export { MeshCanvas } from './components/MeshCanvas'

// Hooks
export { useMeshGenerator } from './hooks/useMeshGenerator'
export { useCanvasControls } from './hooks/useCanvasControls'

// Services (if needed by other modules)
export { canvasRenderer } from './services/canvasRenderer'
export { meshLoader } from './services/meshLoader'
```

### Key Features
- High-performance 3D mesh rendering
- Interactive canvas controls (zoom, pan, rotate)
- Multiple visualization modes
- Real-time mesh generation display
- Canvas content export
- Integration with quality analysis
- Performance optimization for large meshes

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Geometry module for geometric data
- Quality module for quality visualization
- WebGL/Three.js for 3D rendering

### Data Flow
1. Canvas page initializes 3D rendering context
2. useMeshGenerator hook manages mesh data and generation
3. MeshCanvas component renders 3D content
4. Interactive controls update viewport and rendering
5. Quality analysis data overlays on visualization
