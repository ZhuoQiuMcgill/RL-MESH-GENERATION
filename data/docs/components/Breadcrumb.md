# Breadcrumb Component

## Overview
A secondary navigation component that displays the current page path as clickable breadcrumb links, helping users understand their location within the application hierarchy.

## File Location
`frontend/src/components/Breadcrumb.jsx`

## Props
This component does not accept any props - it automatically determines breadcrumbs from the current route.

## State Usage
This component does not manage any local state.

## Dependencies

### React Dependencies
- `Link` from `react-router-dom` - For breadcrumb navigation links
- `useLocation` from `react-router-dom` - To get current URL path for breadcrumb generation

### External Dependencies
- None

## Side Effects
- **Route Reading**: Reads current location from React Router to generate breadcrumbs
- **No DOM manipulation**: Pure component with no side effects

## Features

### Automatic Breadcrumb Generation
- Parses current URL path segments to create breadcrumb trail
- Filters out empty segments for clean breadcrumb display
- Maps path segments to human-readable labels

### Navigation Support
- Home icon (🏠) always links to dashboard (`/`)
- Each intermediate breadcrumb segment is clickable
- Last segment (current page) is highlighted but not clickable

### Label Mapping
Converts URL segments to readable labels:
- `''` (root) → 'Dashboard'
- `'train'` → 'Train'
- `'history'` → 'History'
- `'quality'` → 'Quality'
- `'geometry'` → 'Geometry'
- `'canvas'` → 'Canvas'
- `'angle'` → 'Angle'
- `'action'` → 'Action'
- `'generator'` → 'Generator'

## Visual Design
- **Separator**: Uses arrow (`→`) to separate breadcrumb items
- **Color Scheme**: Secondary text color for inactive items, primary for current page
- **Typography**: Small text (`text-sm`) for compact display
- **Spacing**: Consistent gap between items (`gap-2`)

## CSS Classes Used
- `flex items-center gap-2 text-sm` - Layout and spacing
- `text-text-secondary` - Color for inactive breadcrumbs
- `text-text-primary font-medium` - Color and weight for current page
- `hover:text-text-primary transition-colors` - Hover effects

## Known Issues
1. **Hard-coded Labels**: Breadcrumb labels are hard-coded in the component
2. **Single Level Only**: Only supports single-level paths (doesn't handle nested routes like `/train/session/123`)
3. **No Dynamic Labels**: Cannot display dynamic content like IDs or names in breadcrumbs
4. **Limited Customization**: No way to customize breadcrumb appearance or behavior

## Usage Example
```jsx
import Breadcrumb from './components/Breadcrumb'

function PageLayout() {
  return (
    <div>
      <Breadcrumb />
      {/* Page content */}
    </div>
  )
}
```

## Breadcrumb Examples
- On Dashboard (`/`): `🏠 Dashboard`
- On Train page (`/train`): `🏠 Dashboard → Train`
- On History page (`/history`): `🏠 Dashboard → History`

## Potential Improvements
1. **Nested Route Support**: Handle multi-level routes like `/train/session/123`
2. **Dynamic Labels**: Support for dynamic breadcrumb labels (e.g., showing session names)
3. **Configuration**: Make label mapping configurable via props or context
4. **Custom Separators**: Allow customization of separator characters
5. **Accessibility**: Add proper ARIA navigation landmarks
6. **Rich Content**: Support for icons or custom content in breadcrumbs

## Related Components
- Used alongside: NavHeader for primary navigation
- Complementary to: Page header components
- Integration: Could be integrated into a unified navigation system
