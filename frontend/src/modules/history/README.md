# History Module

## Overview

The history module manages the tracking, storage, and visualization of mesh generation history, including past sessions, results, and user activities within the system.

## Public Surface

### Pages
- `pages/History.jsx` - Main history view with filterable list of past activities

### Components
- `components/HistoryList.jsx` - List component for displaying history entries
- `components/HistoryItem.jsx` - Individual history entry component
- `components/HistoryFilters.jsx` - Filtering and search controls
- `components/SessionDetails.jsx` - Detailed view of a specific session

### Hooks
- `hooks/useHistory.js` - History data fetching and management
- `hooks/useHistoryFilters.js` - History filtering and search logic
- `hooks/useSessionDetails.js` - Individual session data loading

### Services
- `services/historyApi.js` - History data API integration
- `services/historyStorage.js` - Local history data caching

## Module Interface

### Exports
```javascript
// Pages
export { default as HistoryPage } from './pages/History'

// Hooks
export { useHistory } from './hooks/useHistory'
export { useSessionDetails } from './hooks/useSessionDetails'

// Services (if needed by other modules)
export { historyApi } from './services/historyApi'
```

### Key Features
- Comprehensive activity history tracking
- Advanced filtering and search capabilities
- Session replay and analysis
- Export functionality for history data
- Integration with other modules for context

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Training module for training history
- Canvas module for mesh previews

### Data Flow
1. History page loads with recent activity
2. useHistory hook fetches and manages history data
3. Filters and search update the displayed results
4. Users can drill down into specific sessions
5. Session details are loaded on-demand
