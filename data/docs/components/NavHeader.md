# NavHeader Component

## Overview
A comprehensive navigation header component that provides site branding, navigation links, and dark mode toggle functionality for the RL Mesh Generation application.

## File Location
`frontend/src/components/NavHeader.jsx`

## Props
This component does not accept any props - it's a standalone navigation header.

## State Usage
| State Variable | Type | Default | Purpose |
|---------------|------|---------|---------|
| `isDark` | boolean | `true` | Controls dark mode state for the theme toggle |

## Dependencies

### React Dependencies
- `Link` from `react-router-dom` - For client-side navigation
- `useLocation` from `react-router-dom` - To determine current route for active link highlighting
- `useState` from `react` - For dark mode state management

### External Dependencies
- None

## Side Effects

### DOM Manipulation
- **Dark Mode Toggle**: Directly manipulates `document.documentElement.classList` to add/remove the `dark` class
- **Effect**: Changes the global theme by toggling CSS dark mode classes

### Navigation Effects
- **Route Navigation**: Uses React Router's Link components for client-side navigation
- **Active State**: Highlights current page based on `location.pathname`

## Features

### Navigation Items
Provides navigation to all major application sections:
- Dashboard (`/`) - Main overview page
- Train (`/train`) - Training interface
- History (`/history`) - Training history viewer
- Quality (`/quality`) - Quality analysis tools
- Geometry (`/geometry`) - Geometry manipulation
- Canvas (`/canvas`) - Interactive 3D canvas
- Angle (`/angle`) - Angle analysis
- Action (`/action`) - Action space testing
- Generator (`/generator`) - Mesh generation tools

### Visual Features
- **Gradient Background**: Uses `gradient-bg` class for visual appeal
- **Icon Support**: Each navigation item includes an emoji icon
- **Active State Highlighting**: Current page highlighted with `bg-white/20` background
- **Hover Effects**: Navigation items have hover states with `hover:bg-white/10`

## CSS Classes Used
- `gradient-bg` - Background gradient styling
- `text-4xl font-bold text-white` - Main title styling
- `text-white/80` - Secondary text with opacity
- `bg-white/20` - Active navigation item background
- `hover:text-white hover:bg-white/10` - Hover effects

## Known Issues
1. **Global DOM Manipulation**: The dark mode toggle directly manipulates the document element, which could conflict with other theme systems
2. **Hard-coded Navigation**: Navigation items are hard-coded in the component rather than being configurable
3. **No Accessibility**: Dark mode toggle lacks proper ARIA labels for screen readers
4. **State Persistence**: Dark mode state doesn't persist across page reloads

## Usage Example
```jsx
import NavHeader from './components/NavHeader'

function App() {
  return (
    <div>
      <NavHeader />
      {/* Rest of your application */}
    </div>
  )
}
```

## Potential Improvements
1. **Theme Context**: Integrate with a React context for theme management instead of direct DOM manipulation
2. **Configuration**: Make navigation items configurable via props
3. **Accessibility**: Add proper ARIA labels and keyboard navigation support
4. **Persistence**: Save dark mode preference to localStorage
5. **TypeScript**: Add TypeScript support for better type safety

## Related Components
- Used by: All page components that need navigation
- Related to: Breadcrumb component for secondary navigation
