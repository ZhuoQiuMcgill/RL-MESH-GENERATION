# AppShell Layout Component

The `AppShell` component provides a consistent layout structure for the entire application with header, optional sidebar, and main content areas.

## Features

- **Responsive Header**: Contains app title, navigation, theme toggle, and optional sidebar toggle
- **Centralized Navigation**: Navigation items are managed within the AppShell component
- **Optional Sidebar**: Can be shown/hidden and collapsed/expanded
- **Theme Integration**: Works seamlessly with the centralized theme provider
- **Consistent Spacing**: Provides standardized margins and layout structure

## Usage

```jsx
import AppShell from '../shared/layout/AppShell'
import { useTheme } from '../app/providers'

function MyApp() {
  const { isDark, toggleTheme } = useTheme()

  return (
    <AppShell 
      isDark={isDark}
      onThemeToggle={toggleTheme}
      showSidebar={false} // or true to show sidebar
      sidebarContent={<MySidebarComponent />} // optional custom sidebar content
    >
      <YourMainContent />
    </AppShell>
  )
}
```

## Props

- `children`: Main content to be rendered
- `showSidebar`: Boolean to show/hide the sidebar (default: false)
- `sidebarContent`: JSX element to render in the sidebar
- `onThemeToggle`: Function to handle theme switching
- `isDark`: Current theme state (boolean)

## Layout Structure

```
AppShell
├── Header (gradient background)
│   ├── App title and subtitle
│   ├── Navigation menu
│   └── Controls (sidebar toggle, theme toggle)
├── Breadcrumb section
└── Main container
    ├── Optional Sidebar (collapsible)
    └── Main content area
```

## Integration with Theme Provider

The AppShell automatically integrates with the centralized theme provider through the `isDark` and `onThemeToggle` props, ensuring consistent theme management across the entire application.
