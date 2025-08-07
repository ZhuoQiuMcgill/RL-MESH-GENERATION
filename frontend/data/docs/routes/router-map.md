# Router Configuration and Navigation Map

## Overview

This document provides a comprehensive overview of the router configuration, navigation structure, and relationships between pages in the RL Mesh Generation application.

## Router Configuration

### Main Router Setup (`src/App.jsx`)

The application uses React Router v6 with the following configuration:

- **Router Type**: `BrowserRouter` (aliased as `Router`)
- **Lazy Loading**: None - All components are statically imported
- **Base Structure**: Routes are wrapped in `ApiProvider` context

```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
```

### Route Definitions

All routes are defined in `src/App.jsx` with static imports (no lazy loading):

| Path | Component | Import Status | Description |
|------|-----------|---------------|-------------|
| `/` | Dashboard | Static | Main dashboard/home page |
| `/train` | Train | Static | Training interface |
| `/history` | History | Static | Training history view |
| `/quality` | Quality | Static | Quality assessment |
| `/geometry` | Geometry | Static | Geometry configuration |
| `/canvas` | Canvas | Static | 3D mesh canvas |
| `/angle` | Angle | Static | Angle configuration |
| `/action` | Action | Static | Action management |
| `/generator` | Generator | Static | Mesh generation tools |

### Lazy Loading Status

**Current Status**: No lazy loading implemented
- All page components are imported statically at the top of `App.jsx`
- All routes load immediately when the application starts
- No code splitting or dynamic imports are currently in use

**Potential Optimization**: Consider implementing lazy loading for better performance:
```jsx
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Train = lazy(() => import('./pages/Train'))
// ... other components
```

## Navigation Structure

### Navigation Header (`src/components/NavHeader.jsx`)

The main navigation is defined in the `NavHeader` component with the following items:

```jsx
const navItems = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/train', label: 'Train', icon: '🚂' },
  { path: '/history', label: 'History', icon: '📋' },
  { path: '/quality', label: 'Quality', icon: '⭐' },
  { path: '/geometry', label: 'Geometry', icon: '📐' },
  { path: '/canvas', label: 'Canvas', icon: '🎨' },
  { path: '/angle', label: 'Angle', icon: '📐' },
  { path: '/action', label: 'Action', icon: '⚡' },
  { path: '/generator', label: 'Generator', icon: '🔧' }
]
```

### Navigation Item to Route Mapping

| Navigation Label | Route Path | Icon | Purpose |
|------------------|------------|------|---------|
| Dashboard | `/` | 📊 | Main overview and statistics |
| Train | `/train` | 🚂 | Model training interface |
| History | `/history` | 📋 | Training history and logs |
| Quality | `/quality` | ⭐ | Quality metrics and assessment |
| Geometry | `/geometry` | 📐 | Geometric configuration |
| Canvas | `/canvas` | 🎨 | 3D visualization and mesh canvas |
| Angle | `/angle` | 📐 | Angle-related configurations |
| Action | `/action` | ⚡ | Action space management |
| Generator | `/generator` | 🔧 | Mesh generation utilities |

## Breadcrumb Configuration

### Breadcrumb Component (`src/components/Breadcrumb.jsx`)

The breadcrumb system automatically generates navigation breadcrumbs based on the current route:

```jsx
const breadcrumbLabels = {
  '': 'Dashboard',
  'train': 'Train',
  'history': 'History',
  'quality': 'Quality',
  'geometry': 'Geometry',
  'canvas': 'Canvas',
  'angle': 'Angle',
  'action': 'Action',
  'generator': 'Generator'
}
```

### Breadcrumb Generation Logic

1. **Path Parsing**: Current pathname is split into segments
2. **Label Mapping**: Each segment is mapped to a human-readable label
3. **Navigation Links**: All breadcrumb items except the last are clickable links
4. **Home Link**: Always shows a home icon (🏠) linking to Dashboard

## Page Relationships and Flow

### Primary Navigation Flow

```
Dashboard (/)
├── Train (/train)
│   └── Related to: History, Quality
├── History (/history)
│   └── Related to: Train, Quality
├── Quality (/quality)
│   └── Related to: Train, History
├── Geometry (/geometry)
│   └── Related to: Canvas, Angle
├── Canvas (/canvas)
│   └── Related to: Geometry, Generator
├── Angle (/angle)
│   └── Related to: Geometry, Action
├── Action (/action)
│   └── Related to: Angle, Train
└── Generator (/generator)
    └── Related to: Canvas, Quality
```

### Functional Groupings

#### Training Workflow
- **Dashboard** → **Train** → **History** → **Quality**
- Main flow for training models and reviewing results

#### Geometric Configuration
- **Geometry** → **Angle** → **Canvas**
- Configure geometric parameters and visualize results

#### Generation Pipeline
- **Generator** → **Canvas** → **Quality**
- Generate meshes, visualize, and assess quality

#### Action Management
- **Action** → **Train** → **History**
- Configure action space and monitor training

### Page Dependencies

| Page | Dependencies | Purpose |
|------|-------------|---------|
| Dashboard | - | Entry point, overview |
| Train | Action, Geometry | Requires configured actions and geometry |
| History | Train | Shows training session results |
| Quality | Train, Generator | Evaluates training or generation output |
| Geometry | - | Base configuration for spatial parameters |
| Canvas | Geometry, Generator | Visualizes geometric or generated content |
| Angle | Geometry | Specialized geometric parameter |
| Action | - | Base configuration for action space |
| Generator | Geometry | Uses geometric parameters for generation |

## Technical Implementation Notes

### Router Features Used
- ✅ `BrowserRouter` for HTML5 history API
- ✅ `Routes` and `Route` components
- ✅ `Link` components for navigation
- ✅ `useLocation` hook for current path detection
- ❌ Lazy loading / code splitting
- ❌ Route guards or authentication
- ❌ Nested routes
- ❌ Route parameters or query strings

### Current Limitations
1. **No Lazy Loading**: All components load upfront
2. **Flat Route Structure**: No nested or protected routes
3. **Static Navigation**: Navigation items are hardcoded
4. **No Route Parameters**: All routes are static paths

### Recommended Improvements
1. Implement lazy loading for better performance
2. Add route parameters for dynamic content (e.g., `/train/:sessionId`)
3. Consider nested routes for complex page sections
4. Add route guards for feature access control
