# History Page

## Overview
A sophisticated training history viewer that allows users to browse past training sessions, view episode-by-episode data, and visualize training progress with integrated mesh canvas functionality.

## File Location
`frontend/src/pages/History.jsx`

## Props
This page component does not accept any props.

## State Usage
| State Variable | Type | Default | Purpose |
|---------------|------|---------|---------|
| `trainingList` | Array | `[]` | List of available training sessions |
| `selectedTraining` | Object\|null | `null` | Currently selected training session |
| `currentEpisode` | number | `0` | Current episode being viewed |
| `episodeData` | Object\|null | `null` | Data for the current episode |
| `isLoading` | boolean | `false` | Loading state for async operations |
| `error` | string\|null | `null` | Error message for failed operations |
| `log` | Array | `[]` | Operation log entries with timestamps |
| `clickCoordinates` | string | `'Not clicked'` | Canvas click coordinates display |

## Dependencies

### React Dependencies
- `useState` - For component state management
- `useEffect` - For component initialization
- `useRef` - For MeshCanvas reference

### Internal Dependencies
- `NavHeader` from `'../components'` - Page header with navigation
- `MeshCanvas` from `'../components'` - Canvas visualization component
- Multiple UI components: `Button`, `FormInput`, `FormSelect`, `LoadingOverlay`
- `useApi` from `'../context/ApiProvider'` - API client access

### External Dependencies
- None

## Side Effects

### API Integration (Planned)
- **Training History Loading**: `api.getTrainingHistory()` (not implemented)
- **Episode Data Loading**: `api.getEpisodeData(trainingId, episodeIndex)` (not implemented)
- **Health Checking**: `api.checkTrainingHealth()` for service status

### Canvas Manipulation
- **Scene Rendering**: Updates canvas based on episode data
- **Click Handling**: Processes canvas clicks for coordinate display
- **State Synchronization**: Keeps canvas in sync with selected episode

### Logging System
- **Timestamped Logs**: All operations logged with timestamps
- **Categorized Messages**: Info, success, error, warning message types
- **User Feedback**: Real-time operation status through log display

## Features

### Training Session Management
- **Session List**: Browse available training sessions (currently empty/mock)
- **Session Selection**: Click to select and load training session details
- **Session Info**: Display training metadata and statistics

### Episode Navigation
- **Episode Scrubbing**: Navigate through training episodes with input field
- **Sequential Navigation**: Previous/Next episode buttons
- **Quick Access**: Jump to best episode or last episode
- **Episode Details**: View episode-specific metrics and data

### Visualization
- **Canvas Integration**: Full MeshCanvas integration for episode visualization
- **Interactive Canvas**: Click to get world coordinates
- **Loading States**: Visual feedback during data loading
- **Real-time Updates**: Canvas updates when episodes change

### Operation Logging
- **Activity Log**: Real-time log of all operations
- **Timestamped Entries**: Each log entry includes timestamp
- **Color-coded Messages**: Different colors for different message types
- **Clearable Log**: User can clear log history

## Complex Layout

### Three-Panel Layout
1. **Left Panel (320px)**: Training session list and controls
2. **Center Panel**: MeshCanvas visualization area
3. **Right Panel (320px)**: Episode details and operation log

### Responsive Design
- **Flexible Layout**: Main panels adjust to screen size
- **Minimum Widths**: Panels maintain minimum widths for usability
- **Vertical Stacking**: May stack on very small screens

## Canvas Integration

### Event Handling
```jsx
const handleCanvasClick = (worldCoords, event) => {
  if (worldCoords) {
    const coordsText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
    setClickCoordinates(coordsText);
    addLogEntry(`Canvas clicked at: ${coordsText}`, 'info');
  }
};
```

### Visualization Updates
- **Episode Rendering**: Canvas updates when episode changes
- **Data Synchronization**: Mesh data, boundaries, and reference points all synchronized
- **Error Handling**: Graceful handling when visualization data unavailable

## UI Components Used

### Custom UI Components
- `Button` - Various actions and navigation
- `FormInput` - Episode index input
- `FormSelect` - Dropdowns (not currently used)
- `LoadingOverlay` - Full-screen loading indicator
- `NavHeader` - Page header with breadcrumbs

### Component Configuration
```jsx
<NavHeader 
  title="History Viewer"
  breadcrumbs={[
    { label: 'Tools', href: '/' },
    { label: 'History Viewer', href: '/history' }
  ]}
/>
```

## CSS Classes Used
- `min-h-screen bg-bg-primary` - Full-height page container
- `flex min-h-[calc(100vh-var(--nav-header-height))]` - Main layout container
- `w-80 min-w-80` - Fixed-width panels
- `flex-1 flex flex-col min-w-0` - Flexible center panel
- `overflow-y-auto` - Scrollable panel content
- `bg-card border-r border-border-primary` - Panel styling

## Known Issues
1. **API Not Implemented**: Most API calls are commented out/mocked
2. **No Real Data**: Training list and episode data are not loaded from backend
3. **Canvas Updates**: Canvas rendering logic commented out pending API implementation
4. **Error States**: Limited error handling and user feedback
5. **Performance**: No virtualization for large training session lists

## Usage Example
```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import History from './pages/History'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/history" element={<History />} />
      </Routes>
    </Router>
  )
}
```

## Implementation Status
- **UI Components**: ✅ Fully implemented and styled
- **State Management**: ✅ Complete state handling
- **Canvas Integration**: ✅ MeshCanvas properly integrated
- **API Integration**: ❌ Commented out pending backend implementation
- **Data Loading**: ❌ Mock data only
- **Episode Navigation**: ✅ UI complete, needs data integration

## Potential Improvements
1. **API Implementation**: Complete backend API integration
2. **Real-time Updates**: WebSocket integration for live training monitoring
3. **Data Visualization**: Charts and graphs for training metrics
4. **Export Features**: Export episode data, training results, or visualizations
5. **Comparison Tools**: Compare multiple training sessions side-by-side
6. **Search and Filter**: Search training sessions by name, date, or performance
7. **Pagination**: Handle large numbers of training sessions efficiently
8. **Caching**: Cache episode data for faster navigation

## Related Components
- **Requires**: NavHeader, MeshCanvas, and UI components
- **Uses**: ApiProvider context for data access
- **Navigation**: Accessible from Dashboard and navigation header
