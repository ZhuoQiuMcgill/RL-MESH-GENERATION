# Error Handling & Toast System

This directory contains a comprehensive error handling and toast notification system for the RL Mesh Generation application.

## Components

### 1. ToastContext.jsx
The main context provider that manages toast state and provides methods for creating toasts.

**Features:**
- Multiple toast types (info, success, warning, error)
- Configurable positioning
- Auto-dismiss with customizable duration  
- Pause on hover
- Promise-based toasts for async operations
- Maximum toast limits
- Update and remove individual toasts

### 2. Toast.jsx
Individual toast component with animations and interactions.

**Features:**
- Smooth slide-in/out animations
- Progress bar for timed toasts
- Hover to pause auto-dismiss
- Action buttons with retry functionality
- Customizable icons and styling
- Accessible markup

### 3. ToastContainer.jsx
Container component that renders all toasts with proper positioning.

**Features:**
- Portal-based rendering to avoid z-index issues
- Configurable positioning (top/bottom, left/center/right)
- Proper animation ordering
- Screen reader friendly

### 4. ErrorBoundary.jsx
React error boundary component for catching and displaying component errors.

**Features:**
- Catches JavaScript errors in component tree
- Shows user-friendly error messages
- Retry functionality
- Development vs production error details
- Customizable fallback UI
- Error reporting integration

## Usage

### Basic Toast Usage

```jsx
import { useToast } from '../shared/ui'

function MyComponent() {
  const toast = useToast()

  const handleSuccess = () => {
    toast.success('Operation completed successfully!')
  }

  const handleError = () => {
    toast.error('Something went wrong!', {
      action: {
        label: 'Retry',
        onClick: () => {
          // Retry logic
        }
      }
    })
  }

  return (
    <div>
      <button onClick={handleSuccess}>Success</button>
      <button onClick={handleError}>Error</button>
    </div>
  )
}
```

### Promise-based Toasts

```jsx
const handleAsyncOperation = async () => {
  const promise = fetch('/api/data')
  
  toast.promise(promise, {
    loading: 'Fetching data...',
    success: 'Data loaded successfully!',
    error: 'Failed to load data',
    retry: () => handleAsyncOperation()
  })
}
```

### API Integration with Enhanced Client

```jsx
import { useApiWithToast } from '../core/hooks/useApiWithToast'

function TrainingComponent() {
  const api = useApiWithToast()

  const startTraining = async () => {
    try {
      await api.startTraining(config)
      // Success toast automatically shown
    } catch (error) {
      // Error toast automatically shown with retry option
    }
  }

  return <button onClick={startTraining}>Start Training</button>
}
```

### Error Boundary Usage

```jsx
import { ErrorBoundary } from '../shared/ui'

function App() {
  return (
    <ErrorBoundary
      errorMessage="Application encountered an error"
      showReload={true}
      onError={(error, errorInfo) => {
        // Report to error service
        console.error('App Error:', error, errorInfo)
      }}
    >
      <MyComponent />
    </ErrorBoundary>
  )
}
```

### Wrapping Routes

```jsx
import { ErrorBoundary } from '../shared/ui'

function AppRoutes() {
  return (
    <Routes>
      {routes.map((route) => (
        <Route
          key={route.path}
          path={route.path}
          element={
            <ErrorBoundary errorMessage={`Error loading ${route.title}`}>
              <route.component />
            </ErrorBoundary>
          }
        />
      ))}
    </Routes>
  )
}
```

## Configuration

### Toast Provider Setup

```jsx
import { ToastProvider, ToastContainer } from '../shared/ui'

function App() {
  return (
    <ToastProvider
      config={{
        duration: 5000,
        maxToasts: 5,
        position: 'top-right',
        pauseOnHover: true
      }}
    >
      <YourApp />
      <ToastContainer />
    </ToastProvider>
  )
}
```

### Toast Options

```jsx
toast.error('Error message', {
  title: 'Error Title',
  description: 'Additional details',
  duration: 7000,
  action: {
    label: 'Retry',
    onClick: retryFunction,
    closeOnClick: true
  },
  icon: <CustomIcon />, // or false to hide
  showCloseButton: true,
  pauseOnHover: true
})
```

## API Enhanced Client

The enhanced API client provides automatic toast integration for all API operations:

### Features
- Automatic loading toasts
- Success/error notifications
- Retry functionality with exponential backoff
- Network error detection
- Timeout handling
- Customizable messages per operation

### Usage with Hook

```jsx
import { useApiOperations } from '../core/hooks/useApiWithToast'

function DataComponent() {
  const { api, executeOperation, showSuccessToast } = useApiOperations()

  const loadData = async () => {
    try {
      const data = await executeOperation(
        () => api.getMeshList(),
        'loading mesh list',
        {
          loadingMessage: 'Fetching meshes...',
          successMessage: 'Meshes loaded successfully!'
        }
      )
      // Use data
    } catch (error) {
      // Error already handled with toast
    }
  }

  return <button onClick={loadData}>Load Data</button>
}
```

## Styling

The toast system uses Tailwind CSS classes and can be customized by modifying the variant styles in `Toast.jsx`. The components follow the application's design system with theme-aware colors.

## Accessibility

- All toasts include proper ARIA attributes
- Screen reader announcements via `aria-live`
- Keyboard navigation support
- Focus management
- High contrast support

## Browser Support

- Modern browsers with ES2018+ support
- React 16.8+ (hooks)
- Portal API support (for toast rendering)

## Performance

- Efficient re-renders with React Context optimization
- Automatic cleanup of expired toasts
- Memory leak prevention
- Minimal bundle impact with tree shaking
