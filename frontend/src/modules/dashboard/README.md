# Dashboard Module

## Overview

The dashboard module provides a comprehensive overview of the mesh generation system, including system status, recent activity, performance metrics, and quick access to key features.

## Public Surface

### Pages
- `pages/Dashboard.jsx` - Main dashboard page with system overview and metrics

### Components
- `components/SystemStatus.jsx` - System health and status indicators
- `components/ActivityFeed.jsx` - Recent system activity and notifications
- `components/MetricsSummary.jsx` - Key performance metrics display
- `components/QuickActions.jsx` - Shortcut buttons to common actions

### Hooks
- `hooks/useDashboard.js` - Dashboard data aggregation and state management
- `hooks/useSystemMetrics.js` - System performance metrics fetching
- `hooks/useActivityFeed.js` - Activity feed data and real-time updates

### Services
- `services/dashboardApi.js` - Dashboard-specific API endpoints
- `services/metricsService.js` - Metrics aggregation and processing

## Module Interface

### Exports
```javascript
// Pages
export { default as DashboardPage } from './pages/Dashboard'

// Hooks
export { useDashboard } from './hooks/useDashboard'
export { useSystemMetrics } from './hooks/useSystemMetrics'

// Services (if needed by other modules)
export { metricsService } from './services/metricsService'
```

### Key Features
- System status monitoring
- Performance metrics visualization
- Activity feed with real-time updates
- Quick navigation to key features
- Alert and notification management

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Training module for training status
- Canvas module for visualization previews

### Data Flow
1. Dashboard page loads and initializes data fetching
2. useDashboard hook aggregates data from multiple sources
3. useSystemMetrics provides performance data
4. Components display organized dashboard sections
5. Real-time updates refresh metrics and activity feed
