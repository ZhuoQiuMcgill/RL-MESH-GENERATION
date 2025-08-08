# Component Map and Relationships

**Status**: Approved  
**Last Updated**: 2024-01-20  
**Owner**: Frontend Team  
**Reviewers**: Development Team  

## Overview

This document provides a comprehensive map of all components in the RL Mesh Generation frontend application, their relationships, and architectural patterns.

## Component Hierarchy

### Application Root
```
App (src/App.jsx)
├── ApiProvider (src/context/ApiProvider.jsx)
│   └── Router
│       └── Routes
│           ├── Dashboard (src/pages/Dashboard.jsx)
│           ├── Train (src/pages/Train.jsx)
│           ├── TrainingMonitor (src/pages/TrainingMonitor.jsx)
│           ├── Canvas (src/pages/Canvas.jsx)
│           ├── History (src/pages/History.jsx)
│           ├── Quality (src/pages/Quality.jsx)
│           ├── Geometry (src/pages/Geometry.jsx)
│           ├── Angle (src/pages/Angle.jsx)
│           ├── Action (src/pages/Action.jsx)
│           └── Generator (src/pages/Generator.jsx)
```

## Module-by-Module Component Map

### Shared Layout Components

#### AppShell (src/shared/layout/AppShell.jsx)
```
AppShell
├── NavHeader
│   ├── Logo
│   ├── NavigationMenu
│   └── ThemeToggle
├── Breadcrumb
│   └── BreadcrumbItem[]
└── MainContent
    └── <Outlet /> (Router content)
```

**Dependencies:**
- `NavHeader` → Navigation state
- `Breadcrumb` → Route context
- React Router `Outlet`

**Usage Pattern:**
```jsx
<AppShell>
  {/* Page content rendered via routing */}
</AppShell>
```

#### NavHeader (src/components/NavHeader.jsx)
```
NavHeader
├── Brand/Logo
├── NavigationLinks[]
│   ├── DashboardLink
│   ├── TrainLink
│   ├── CanvasLink
│   └── [other navigation items]
└── UserActions
    └── ThemeToggle
```

**Dependencies:**
- Route configuration
- Theme context
- Navigation state

### Shared UI Components

#### Button System (src/shared/ui/Button.jsx)
```
Button (Base)
├── PrimaryButton
├── SecondaryButton  
├── AccentButton
├── DangerButton
└── GhostButton

ButtonGroup
├── Button[]
└── Separator?
```

**Variants:**
- `primary` - Main actions (gradient background)
- `secondary` - Secondary actions (card background)
- `accent` - Special emphasis (accent color)
- `danger` - Destructive actions (red)
- `ghost` - Subtle actions (transparent)

**Usage Pattern:**
```jsx
<Button variant="primary" size="md" loading={isLoading}>
  Start Training
</Button>
```

#### Card System (src/shared/ui/Card.jsx)
```
Card (Base Component)
├── CardHeader?
│   ├── CardTitle
│   └── CardActions?
├── CardContent
└── CardFooter?
    └── CardActions

PanelCard (Enhanced Card)
├── StatusIndicator?
├── CardHeader
├── CardContent
└── CardActions
```

**Dependencies:**
- Design token system
- Theme context

#### Form Components
```
FormInput (src/shared/ui/FormInput.jsx)
├── Label
├── Input
├── ValidationMessage?
└── HelpText?

FormSelect (src/shared/ui/FormSelect.jsx)  
├── Label
├── Select
│   └── Option[]
├── ValidationMessage?
└── HelpText?
```

**Dependencies:**
- Form validation system
- Error boundary context

#### Status & Feedback Components
```
CompactStatusBar (src/shared/ui/CompactStatusBar.jsx)
├── StatusIcon
├── StatusText
└── ActionButtons?

LoadingOverlay (src/shared/ui/LoadingOverlay.jsx)
├── Backdrop
├── Spinner
└── LoadingMessage?

EmptyState (src/shared/ui/EmptyState.jsx)
├── IllustrationIcon
├── EmptyTitle
├── EmptyDescription
└── ActionButton?
```

### Page-Level Components

#### Dashboard (src/pages/Dashboard.jsx)
```
Dashboard
├── PageHeader
│   ├── Title
│   └── QuickActions
├── StatusCards[]
│   ├── TrainingStatusCard
│   ├── MeshStatsCard
│   ├── SystemHealthCard
│   └── RecentActivityCard
└── DashboardCharts?
    ├── TrainingProgressChart
    └── MeshQualityChart
```

**Dependencies:**
- `useApi()` hook for data fetching
- Status components
- Chart libraries (future)

#### Training Pages
```
Train (src/pages/Train.jsx)
├── TrainingConfigForm
│   ├── FormInput (learning rate, episodes, etc.)
│   ├── FormSelect (algorithm selection)
│   └── SubmitButton
└── TrainingHistory?
    └── HistoryTable

TrainingMonitor (src/pages/TrainingMonitor.jsx)
├── TrainingControls
│   ├── StartButton
│   ├── StopButton
│   └── PauseButton
├── RealTimeMetrics
│   ├── ProgressBar
│   ├── MetricsDisplay
│   └── LiveChart
└── LogOutput
    └── LogViewer
```

**Dependencies:**
- `useTrainingHooks()` for training state
- `usePolling()` for real-time updates
- Form validation

#### Canvas (src/pages/Canvas.jsx)
```
Canvas
├── CanvasToolbar
│   ├── ViewControls
│   ├── RenderOptions
│   └── ExportButtons
├── MeshCanvas
│   ├── WebGLRenderer
│   ├── InteractionHandlers
│   └── MeshDisplay
└── CanvasSidebar?
    ├── MeshProperties
    ├── LayerControls
    └── QualityMetrics
```

**Dependencies:**
- WebGL context
- Mesh data from API
- Canvas utilities

#### MeshCanvas (src/components/MeshCanvas.jsx)
```
MeshCanvas
├── CanvasElement
├── WebGLContext
├── RenderLoop
├── InteractionSystem
│   ├── MouseHandlers
│   ├── TouchHandlers
│   └── KeyboardHandlers
└── PerformanceMonitor
```

**Dependencies:**
- `CanvasRenderer.js` utility
- Mesh data structures
- Performance optimization hooks

## Component Communication Patterns

### 1. Props Down, Events Up
```
Parent Component
    │ (props)
    ▼
Child Component
    │ (events/callbacks)
    ▲
Parent Component
```

### 2. Context-Based Global State
```
Provider (Context)
    │ (value)
    ├── Component A
    ├── Component B
    └── Component C
```

### 3. Custom Hook Data Flow
```
Custom Hook
    │ (data, actions)
    ├── Component X
    ├── Component Y
    └── Component Z
```

## Component Dependencies Graph

### High-Level Dependencies
```
API Layer
    │
    ├── useApi() hook
    ├── usePolling() hook
    └── Custom Module Hooks
            │
            ├── Page Components
            ├── Feature Components
            └── UI Components
                    │
                    └── Base UI Components
```

### Specific Dependency Examples

#### Training Module Dependencies
```
ApiProvider
    │
    ├── useApi()
    └── usePolling()
            │
            ├── useTrainingHooks()
            │       │
            │       ├── Train.jsx
            │       └── TrainingMonitor.jsx
            └── TrainingMonitor.jsx (direct polling)
```

#### UI Component Dependencies
```
Design Tokens (CSS Custom Properties)
    │
    ├── Button.jsx
    ├── Card.jsx
    ├── FormInput.jsx
    └── All UI Components
            │
            ├── Page Components
            └── Feature Components
```

## Component Lifecycle Patterns

### 1. Standard React Lifecycle
```
Mount → Render → Update → Unmount
    │       │        │        │
    ▼       ▼        ▼        ▼
useEffect for data fetching, cleanup
```

### 2. API Integration Lifecycle
```
Component Mount
    │
    ├── API Call (useApi)
    ├── Loading State
    ├── Success/Error State
    └── Component Update
            │
            └── Cleanup on Unmount
```

### 3. Real-time Data Lifecycle
```
Component Mount
    │
    ├── Start Polling (usePolling)
    ├── Receive Updates
    ├── Update State
    └── Stop Polling on Unmount
```

## Testing Patterns

### Component Testing Structure
```
Component.test.jsx
├── Render Tests
│   ├── Default render
│   ├── Props variations
│   └── State variations
├── Interaction Tests
│   ├── User events
│   ├── Form submissions
│   └── Button clicks
├── API Integration Tests
│   ├── Loading states
│   ├── Success responses
│   └── Error handling
└── Accessibility Tests
    ├── Keyboard navigation
    ├── Screen reader support
    └── Focus management
```

## Performance Optimization Patterns

### 1. React Optimization
```jsx
// Memoization patterns
const MemoizedComponent = React.memo(Component);
const memoizedValue = useMemo(() => computation, [deps]);
const memoizedCallback = useCallback(handler, [deps]);
```

### 2. Bundle Optimization
```jsx
// Lazy loading
const LazyComponent = lazy(() => import('./Component'));

// Code splitting
<Suspense fallback={<LoadingSpinner />}>
  <LazyComponent />
</Suspense>
```

### 3. Data Fetching Optimization
```jsx
// Efficient polling
const { data, isLoading } = usePolling(endpoint, {
  enabled: shouldPoll,
  dependencies: [key],
});
```

## Component Style Patterns

### 1. Tailwind + Design Tokens
```jsx
<div className="bg-bg-primary text-text-primary p-xl rounded-lg">
  {/* Component content */}
</div>
```

### 2. Conditional Styling
```jsx
<button className={cn(
  'btn-base',
  variant === 'primary' && 'btn-primary',
  disabled && 'btn-disabled'
)}>
```

### 3. Responsive Design
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-lg">
  {/* Responsive grid */}
</div>
```

## Component Evolution Guidelines

### 1. Adding New Components
1. Follow existing patterns and naming conventions
2. Implement proper TypeScript-style JSDoc
3. Add comprehensive tests
4. Update this component map
5. Follow the established folder structure

### 2. Modifying Existing Components
1. Maintain backward compatibility
2. Update tests and documentation
3. Consider impact on dependent components
4. Follow the established migration patterns

### 3. Component Deprecation
1. Mark as deprecated in JSDoc
2. Provide migration path
3. Update documentation
4. Plan removal timeline

---

*This component map serves as a living document and should be updated as the component architecture evolves.*
