# Angle Module

## Overview

The angle module specializes in angle analysis and measurement within mesh structures. It provides tools for analyzing mesh angles, detecting angle-based quality issues, and visualizing angular properties.

## Public Surface

### Pages
- `pages/Angle.jsx` - Main angle analysis interface with measurement tools and visualizations

### Components
- `components/AngleAnalyzer.jsx` - Angle measurement and analysis tools
- `components/AngleVisualization.jsx` - Visual representation of angle data
- `components/AngleMetrics.jsx` - Angle-based quality metrics display
- `components/AngleHistogram.jsx` - Histogram visualization of angle distributions

### Hooks
- `hooks/useAngleAnalysis.js` - Core angle analysis functionality
- `hooks/useAngleMeasurement.js` - Interactive angle measurement tools
- `hooks/useAngleQuality.js` - Angle-based quality assessment

### Services
- `services/angleApi.js` - Angle analysis API integration
- `services/angleCalculator.js` - Angle calculation algorithms
- `services/angleValidator.js` - Angle quality validation

## Module Interface

### Exports
```javascript
// Pages
export { default as AnglePage } from './pages/Angle'

// Hooks
export { useAngleAnalysis } from './hooks/useAngleAnalysis'
export { useAngleMeasurement } from './hooks/useAngleMeasurement'

// Services (if needed by other modules)
export { angleCalculator } from './services/angleCalculator'
export { angleValidator } from './services/angleValidator'
```

### Key Features
- Comprehensive angle analysis and measurement
- Angle quality assessment and validation
- Visual angle distribution analysis
- Interactive angle measurement tools
- Angle-based mesh quality metrics
- Integration with quality module

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Canvas module for angle visualization
- Geometry module for geometric calculations
- Quality module for quality integration

### Data Flow
1. Angle page loads mesh data for analysis
2. useAngleAnalysis hook processes angle calculations
3. Angle measurements and quality metrics are computed
4. Visualization components display angle data
5. Results integrate with overall quality assessment
