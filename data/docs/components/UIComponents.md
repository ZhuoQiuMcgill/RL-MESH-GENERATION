# UI Components Documentation

This document covers the reusable UI components used throughout the RL Mesh Generation application.

## Button Component

### File Location
`frontend/src/components/ui/Button.jsx`

### Overview
A flexible, customizable button component with multiple variants, sizes, and states.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `variant` | string | `'primary'` | Button style variant |
| `size` | string | `'default'` | Button size |
| `disabled` | boolean | `false` | Disabled state |
| `className` | string | `''` | Additional CSS classes |
| `children` | ReactNode | - | Button content |

### Variants
- `primary` - Main action button with card background
- `secondary` - Secondary action with subtle styling
- `danger` - Destructive actions (red background)
- `success` - Positive actions (green background)
- `outline` - Outlined button with border
- `ghost` - Minimal styling, no background

### Sizes
- `sm` - Small button (px-3 py-1.5 text-sm)
- `default` - Standard size (px-5 py-3)
- `lg` - Large button (px-6 py-4 text-lg)

### Dependencies
- `cn` from `'../../lib/utils'` - Class name utility
- `React.forwardRef` - For ref forwarding

---

## FormInput Component

### File Location
`frontend/src/components/ui/FormInput.jsx`

### Overview
A form input component with built-in label, error handling, and consistent styling.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `type` | string | `'text'` | Input type |
| `label` | string | - | Input label |
| `error` | string | - | Error message |
| `disabled` | boolean | `false` | Disabled state |
| `className` | string | `''` | Additional CSS classes |

### Features
- **Label Support**: Optional label with consistent styling
- **Error States**: Red border and error message display
- **Focus States**: Accent color ring on focus
- **Disabled States**: Visual feedback for disabled inputs

---

## FormSelect Component

### File Location
`frontend/src/components/ui/FormSelect.jsx`

### Overview
A select dropdown component with options support, placeholder text, and error handling.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `label` | string | - | Select label |
| `error` | string | - | Error message |
| `disabled` | boolean | `false` | Disabled state |
| `options` | Array | `[]` | Options array with {value, label} objects |
| `placeholder` | string | `"Select an option..."` | Placeholder text |
| `children` | ReactNode | - | Custom option elements |

### Features
- **Options Array**: Supports both options prop and children
- **Placeholder Support**: Disabled default option
- **Consistent Styling**: Matches FormInput styling
- **Error Handling**: Red styling for error states

---

## PanelCard Component

### File Location
`frontend/src/components/ui/PanelCard.jsx`

### Overview
A card container component with optional title and subtitle, used for organizing content sections.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `title` | string | - | Card title |
| `subtitle` | string | - | Card subtitle |
| `className` | string | `''` | Additional CSS classes |
| `children` | ReactNode | - | Card content |

### Features
- **Optional Header**: Title and subtitle with consistent spacing
- **Card Styling**: Border, padding, and rounded corners
- **Flexible Content**: Accepts any children content

---

## CompactStatusBar Component

### File Location
`frontend/src/components/ui/CompactStatusBar.jsx`

### Overview
A status display component that shows key-value pairs in a compact format.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `items` | Array | `[]` | Array of status items |
| `className` | string | `''` | Additional CSS classes |

### Item Structure
Each item in the `items` array should have:
- `label` - The status label
- `value` - The status value
- `color` - Optional color for the value text

### Usage Example
```jsx
const statusItems = [
  { label: 'Status', value: 'Ready', color: 'success' },
  { label: 'Progress', value: '75%', color: 'primary' }
];

<CompactStatusBar items={statusItems} />
```

---

## LoadingOverlay Component

### File Location
`frontend/src/components/ui/LoadingOverlay.jsx`

### Overview
A loading indicator component that can be used as an overlay or inline element.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `text` | string | `"Loading..."` | Loading message |
| `overlay` | boolean | `true` | Whether to render as full-screen overlay |
| `size` | string | `'default'` | Spinner and text size |
| `className` | string | `''` | Additional CSS classes |

### Sizes
- `sm` - Small spinner (w-6 h-6) with small text
- `default` - Medium spinner (w-8 h-8) with base text
- `lg` - Large spinner (w-12 h-12) with large text

### Features
- **Full-screen Overlay**: Modal-style overlay with backdrop blur
- **Inline Mode**: Can be used inline without overlay
- **Animated Spinner**: CSS animation with accent color
- **Custom Text**: Configurable loading message

### Usage Example
```jsx
// Full-screen overlay
<LoadingOverlay text="Processing..." />

// Inline loading
<LoadingOverlay overlay={false} size="sm" />
```

---

## EmptyState Component

### File Location
`frontend/src/components/ui/EmptyState.jsx`

### Overview
A component for displaying empty states with icon, title, description, and optional action.

### Props
| Prop Name | Type | Default | Purpose |
|-----------|------|---------|---------|
| `icon` | string/ReactNode | - | Icon or emoji to display |
| `title` | string | - | Empty state title |
| `description` | string | - | Descriptive text |
| `action` | ReactNode | - | Action button or element |
| `size` | string | `'default'` | Component size variant |

### Sizes
- `sm` - Compact empty state (py-8, text-4xl icon)
- `default` - Standard empty state (py-12, text-6xl icon)
- `lg` - Large empty state (py-16, text-8xl icon)

### Features
- **Flexible Icon**: Supports emoji strings or React elements
- **Responsive Typography**: Size-appropriate text scaling
- **Optional Action**: Can include buttons or other interactive elements
- **Consistent Spacing**: Proper vertical spacing between elements

### Usage Example
```jsx
<EmptyState 
  icon="🔧"
  title="Ready to Generate Mesh"
  description="Select a mesh, predictor, and reference selector to begin mesh generation."
  action={<Button variant="primary">Get Started</Button>}
/>
```

---

## Common Patterns

### Class Name Utilities
All components use the `cn` utility function from `'../../lib/utils'` for merging class names:
```jsx
className={cn('base-classes', conditional && 'conditional-classes', className)}
```

### Consistent Styling
Components follow consistent design patterns:
- **Colors**: Use CSS custom properties (text-text-primary, bg-card, etc.)
- **Spacing**: Consistent padding and margin values
- **Borders**: Standard border radius (rounded-lg, rounded-xl)
- **Focus States**: Accent color rings with proper outline removal
- **Transitions**: Smooth color and state transitions

### Accessibility
- **forwardRef**: Most form components support ref forwarding
- **Disabled States**: Proper disabled styling and cursor states
- **Semantic HTML**: Use appropriate HTML elements
- **Error States**: Visual error indication with color changes

### Known Issues
1. **Color Classes**: Some components use template literals for dynamic colors that may not be included in CSS build
2. **Missing ARIA**: Limited ARIA attributes for screen readers
3. **Focus Management**: Some complex components need better focus management
4. **TypeScript**: No TypeScript definitions for better type safety

### Potential Improvements
1. **TypeScript**: Add comprehensive TypeScript definitions
2. **Theme System**: More comprehensive theming support
3. **Accessibility**: Enhanced ARIA support and keyboard navigation
4. **Animation**: More sophisticated animation and transition systems
5. **Documentation**: Interactive component documentation/storybook
6. **Testing**: Unit tests for all components
