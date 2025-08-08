# Quality Module

## Overview

The quality module provides comprehensive mesh quality analysis, including geometric validation, quality metrics calculation, and visualization of mesh quality issues and improvements.

## Public Surface

### Pages
- `pages/Quality.jsx` - Main quality analysis interface with metrics and visualizations

### Components
- `components/QualityMetrics.jsx` - Quality metrics display and charts
- `components/QualityAnalysis.jsx` - Detailed quality analysis results
- `components/QualityVisualization.jsx` - Visual representation of quality issues
- `components/QualityComparison.jsx` - Side-by-side quality comparisons

### Hooks
- `hooks/useQualityAnalysis.js` - Quality analysis logic and calculations
- `hooks/useQualityMetrics.js` - Quality metrics data fetching
- `hooks/useQualityComparison.js` - Quality comparison functionality

### Services
- `services/qualityApi.js` - Quality analysis API integration
- `services/qualityCalculator.js` - Client-side quality metric calculations
- `services/qualityReports.js` - Quality report generation

## Module Interface

### Exports
```javascript
// Pages
export { default as QualityPage } from './pages/Quality'

// Hooks
export { useQualityAnalysis } from './hooks/useQualityAnalysis'
export { useQualityMetrics } from './hooks/useQualityMetrics'

// Services (if needed by other modules)
export { qualityCalculator } from './services/qualityCalculator'
export { qualityReports } from './services/qualityReports'
```

### Key Features
- Comprehensive mesh quality analysis
- Multiple quality metrics (aspect ratio, skewness, etc.)
- Quality visualization and highlighting
- Quality comparison between meshes
- Quality improvement suggestions
- Quality reports and exports

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Canvas module for quality visualization
- Geometry module for geometric calculations

### Data Flow
1. Quality page receives mesh data for analysis
2. useQualityAnalysis hook processes quality calculations
3. Quality metrics are computed and displayed
4. Visualization components highlight quality issues
5. Results can be compared and exported
