# Action Module

## Overview

The action module manages user actions, system commands, and workflow orchestration within the mesh generation system. It provides action tracking, undo/redo functionality, and batch operation capabilities.

## Public Surface

### Pages
- `pages/Action.jsx` - Main action management interface with action history and controls

### Components
- `components/ActionPanel.jsx` - Action control panel and quick actions
- `components/ActionHistory.jsx` - History of user actions with undo/redo
- `components/BatchActions.jsx` - Batch operation controls and status
- `components/ActionQueue.jsx` - Action queue management and monitoring

### Hooks
- `hooks/useActions.js` - Core action management and dispatch
- `hooks/useActionHistory.js` - Action history and undo/redo functionality
- `hooks/useBatchActions.js` - Batch operation management

### Services
- `services/actionApi.js` - Action-related API integration
- `services/actionQueue.js` - Action queue processing and management
- `services/actionLogger.js` - Action logging and tracking

## Module Interface

### Exports
```javascript
// Pages
export { default as ActionPage } from './pages/Action'

// Hooks
export { useActions } from './hooks/useActions'
export { useActionHistory } from './hooks/useActionHistory'
export { useBatchActions } from './hooks/useBatchActions'

// Services (if needed by other modules)
export { actionQueue } from './services/actionQueue'
export { actionLogger } from './services/actionLogger'
```

### Key Features
- Comprehensive action management system
- Undo/redo functionality
- Batch operation support
- Action queue processing
- Action logging and audit trail
- Integration with all modules for action tracking

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- All other modules for action integration
- History module for action persistence

### Data Flow
1. Actions are dispatched through useActions hook
2. Action history is maintained for undo/redo
3. Batch actions are queued and processed
4. Action results are logged and tracked
5. All actions integrate with history module
