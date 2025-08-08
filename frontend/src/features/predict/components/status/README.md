# StatusDisplay Components

The StatusDisplay component group provides real-time status monitoring for mesh generation prediction sessions. All components are completely decoupled and derive their data from `context.state` via the `usePredictSession` hook.

## Components

### 1. SessionStatusPanel
Displays current session information including:
- Current step and total steps with progress bar
- Boundary status (Active/Inactive)
- Session duration
- Completion status with visual indicators

### 2. ActionInfoPanel
Shows action-related information including:
- Action type (type0_left, type0_right, type1, etc.)
- Reference vertex index
- Validity status with color-coded indicators
- Coordinates (click points, decoded coords, generated elements)
- Polar coordinates when available

### 3. ReferencePointPanel
Displays reference point data including:
- Selector index
- Point coordinates
- Interior angle with quality assessment
- Neighbor vertices
- Visual angle indicator with quality color coding

### 4. QualityPanel
Calculates and shows mesh quality metrics including:
- Average quality score with progress bar
- Min/Max quality values
- Median quality
- Element count and quality data coverage
- Quality distribution (Excellent/Good/Fair/Poor)
- Quality trend visualization
- Letter grade (A+ to F)

## Usage

### Individual Components
```jsx
import { SessionStatusPanel, ActionInfoPanel, ReferencePointPanel, QualityPanel } from '../components';

function MyComponent() {
  return (
    <div className="space-y-4">
      <SessionStatusPanel />
      <ActionInfoPanel />
      <ReferencePointPanel />
      <QualityPanel />
    </div>
  );
}
```

### Combined StatusDisplay
```jsx
import { StatusDisplay } from '../components';

function MyComponent() {
  return (
    <div>
      {/* Grid layout (default) */}
      <StatusDisplay />
      
      {/* Column layout */}
      <StatusDisplay layout="column" />
      
      {/* Row layout */}
      <StatusDisplay layout="row" />
      
      {/* Custom styling */}
      <StatusDisplay className="my-4 p-6 bg-gray-50" />
    </div>
  );
}
```

### With PredictSessionProvider
```jsx
import { PredictSessionProvider } from '../contexts/PredictSessionContext';
import { StatusDisplay } from '../components';

function PredictPage() {
  return (
    <PredictSessionProvider>
      <div className="container mx-auto p-6">
        <StatusDisplay />
      </div>
    </PredictSessionProvider>
  );
}
```

## Data Sources

All components automatically extract data from the PredictSession context:

- **Session Status**: `currentStep`, `totalSteps`, `progress`, `status`, `startTime`, `endTime`
- **Action Info**: Parsed from `logs` and `meshData.lastAction`
- **Reference Point**: `refPoint` and related log entries
- **Quality Metrics**: Calculated from `meshData` elements and quality logs

## Features

### Real-time Updates
Components automatically re-render when context state changes.

### No Coupling
Each component is independent and can be used separately without dependencies on others.

### Responsive Design
All components use Tailwind CSS classes and are fully responsive.

### Visual Indicators
- Color-coded status badges
- Progress bars and trend charts
- Quality grading system
- Interactive angle visualization

### Error Handling
Components gracefully handle missing or invalid data with appropriate fallbacks.

### Performance
Uses `useMemo` for expensive calculations like quality metrics computation.

## Styling

Components use consistent Tailwind CSS styling:
- White background with subtle shadows
- Gray color palette for text hierarchy
- Color-coded status indicators (green/blue/yellow/red)
- Smooth transitions and animations
- Responsive grid layouts

## Data Flow

```
PredictSessionContext (state)
         ↓
   usePredictSession()
         ↓
  StatusDisplay Components
         ↓
    Visual Rendering
```

The components are completely stateless and reactive - they automatically update when the session context changes without requiring manual data passing or prop drilling.
