# Geometry Module

## Overview

The geometry module handles geometric operations, shape management, and geometric analysis for mesh generation. It provides tools for working with geometric primitives, transformations, and spatial calculations.

## Public Surface

### Pages
- `pages/Geometry.jsx` - Main geometry management interface with shape tools and properties

### Components
- `components/GeometryTools.jsx` - Tool palette for geometric operations
- `components/ShapeProperties.jsx` - Properties panel for geometric shapes
- `components/GeometryViewer.jsx` - 3D viewer for geometric shapes
- `components/TransformControls.jsx` - Controls for geometric transformations

### Hooks
- `hooks/useGeometry.js` - Core geometry operations and state management
- `hooks/useShapeManagement.js` - Shape creation, editing, and deletion
- `hooks/useGeometricCalculations.js` - Geometric calculations and analysis

### Services
- `services/geometryApi.js` - Geometry data API integration
- `services/geometryEngine.js` - Core geometric calculations and operations
- `services/shapeValidator.js` - Geometric validation and error checking

## Module Interface

### Exports
```javascript
// Pages
export { default as GeometryPage } from './pages/Geometry'

// Hooks
export { useGeometry } from './hooks/useGeometry'
export { useShapeManagement } from './hooks/useShapeManagement'
export { useGeometricCalculations } from './hooks/useGeometricCalculations'

// Services (if needed by other modules)
export { geometryEngine } from './services/geometryEngine'
export { shapeValidator } from './services/shapeValidator'
```

### Key Features
- Geometric shape creation and editing
- Shape transformation and manipulation
- Geometric property calculations
- Shape validation and error detection
- Import/export of geometric data
- Integration with mesh generation

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Canvas module for geometric visualization
- Quality module for geometric validation

### Data Flow
1. Geometry page provides shape management interface
2. useGeometry hook manages geometric state
3. Geometric operations are performed via geometryEngine
4. Shape validation ensures data integrity
5. Results are visualized and can be used for mesh generation
