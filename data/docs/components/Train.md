# Train Page

## Overview
The training interface page that provides access to reinforcement learning training functionality through the TrainingMonitor component, along with additional training history and navigation.

## File Location
`frontend/src/pages/Train.jsx`

## Props
This page component does not accept any props.

## State Usage
This component does not manage local state - it delegates functionality to the TrainingMonitor component.

## Dependencies

### React Dependencies
- `Link` from `react-router-dom` - For navigation back to dashboard

### Internal Dependencies
- `TrainingMonitor` from `'../components/TrainingMonitor'` - Main training interface component

### External Dependencies
- None

## Side Effects
- **Navigation**: Provides back-to-dashboard navigation
- **Delegation**: All training functionality delegated to TrainingMonitor component

## Features

### Page Structure
- **Header Section**: Page title with emoji icon and description
- **Back Navigation**: Link to return to dashboard
- **Training Interface**: Full TrainingMonitor component integration
- **Training History**: Static table of recent training runs

### Training History Table
Displays recent training sessions with:
- **Model Name**: PPO v1.2, SAC v1.0
- **Progress**: Episodes completed (e.g., 850/1000)
- **Performance**: Best reward achieved
- **Duration**: Time spent training
- **Status**: Running, Complete with color-coded badges

### Navigation
- **Back Button**: Styled link to dashboard with arrow icon
- **Breadcrumb-style**: Shows hierarchical relationship

## Visual Design

### Layout Structure
- **Max Width Container**: `max-w-7xl mx-auto` for content constraints
- **Vertical Spacing**: Consistent margin bottom spacing between sections
- **Card-based Design**: Training history in card container

### Status Badges
- **Running**: Yellow background with yellow text
- **Complete**: Green background with green text
- **Consistent Styling**: `px-2 py-1 rounded text-sm`

### Table Styling
- **Responsive**: Horizontal scroll on smaller screens
- **Clean Headers**: Secondary text color for column headers
- **Row Separation**: Border between table rows
- **Text Hierarchy**: Primary text for data, secondary for headers

## CSS Classes Used
- `max-w-7xl mx-auto` - Main container
- `text-3xl font-bold text-text-primary` - Page title
- `text-4xl` - Large emoji icon
- `inline-flex items-center` - Back button layout
- `bg-card border border-border-custom rounded-xl` - Card styling
- `overflow-x-auto` - Table responsiveness
- `bg-yellow-500/20 text-yellow-400` - Status badge colors

## Static Data
Training history table uses static/mock data:
```jsx
// PPO v1.2 - Running (850/1000 episodes, 2h 15m)
// SAC v1.0 - Complete (1000/1000 episodes, 3h 42m)
```

## Known Issues
1. **Static Training History**: Table data is hard-coded, not fetched from API
2. **No Real Integration**: Training history not connected to actual training system
3. **Limited Functionality**: Page mainly serves as wrapper for TrainingMonitor
4. **No Error Handling**: No error boundaries or error states
5. **Missing Features**: No pagination, filtering, or search for training history

## Usage Example
```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Train from './pages/Train'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/train" element={<Train />} />
      </Routes>
    </Router>
  )
}
```

## Component Integration
The Train page primarily serves as a container that:
1. Provides page-level navigation and branding
2. Integrates the full TrainingMonitor component
3. Shows supplementary training history information
4. Maintains consistent page layout patterns

## Page Flow
1. **User Navigation**: User clicks "Training" from dashboard
2. **Page Load**: Train page renders with header and back navigation
3. **Training Interface**: TrainingMonitor component provides main functionality
4. **History Display**: Static training history table shows at bottom

## Potential Improvements
1. **Dynamic History**: Connect to real training history API
2. **Live Updates**: Real-time updates of training progress
3. **Pagination**: Handle large numbers of training sessions
4. **Filtering**: Filter by model type, status, date range
5. **Export**: Export training results or history
6. **Integration**: Better integration between history and current training
7. **Status Updates**: Real-time status updates for running training
8. **Model Management**: Ability to manage and compare different models

## Related Components
- **Contains**: TrainingMonitor as primary functionality
- **Navigation**: Links to/from Dashboard
- **Layout**: Follows standard page layout patterns
- **Styling**: Consistent with other page components
