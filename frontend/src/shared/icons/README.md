# Icon System

This directory contains a comprehensive icon system that replaces emojis with accessible, consistent SVG icons using Lucide React.

## Features

- ✅ **Accessible**: All icons include proper ARIA labels, roles, and focus management
- ✅ **Consistent**: Unified sizing, styling, and interaction patterns
- ✅ **Semantic**: Icons convey meaning beyond visual representation
- ✅ **Keyboard navigable**: Full keyboard support with focus indicators
- ✅ **WCAG AA compliant**: Proper color contrast and accessibility attributes
- ✅ **Theme aware**: Icons adapt to light/dark themes

## Components

### Base Icon Component

The `Icon` component provides the foundation for all icons with built-in accessibility features.

```jsx
import { Icon } from '../../shared/icons'

<Icon 
  size={24}
  aria-label="Custom icon"
  onClick={handleClick}
  variant="primary"
>
  <CustomLucideIcon />
</Icon>
```

### Navigation Icons

Pre-built icon components for navigation and main UI elements:

```jsx
import { DashboardIcon, TrainIcon, HistoryIcon } from '../../shared/icons'

<DashboardIcon size={20} />
<TrainIcon size={24} onClick={handleTraining} />
<HistoryIcon size={16} variant="muted" />
```

### Status Icons

Icons for training status, controls, and indicators:

```jsx
import { PlayIcon, StatusDot, LoadingIcon } from '../../shared/icons'

<PlayIcon onClick={startTraining} />
<StatusDot status="running" />
<LoadingIcon size={20} />
```

## Props

### Icon Component Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `size` | `number` | `20` | Icon size in pixels (12, 14, 16, 20, 24, 28, 32, 40, 48) |
| `className` | `string` | `''` | Additional CSS classes |
| `aria-label` | `string` | - | Accessible label for screen readers |
| `aria-hidden` | `boolean` | `false` | Hide from screen readers (decorative icons) |
| `title` | `string` | - | Tooltip text |
| `role` | `string` | `'img'` | ARIA role |
| `focusable` | `boolean` | `false` | Whether icon can receive focus |
| `onClick` | `function` | - | Click handler (makes icon interactive) |
| `disabled` | `boolean` | `false` | Disabled state |
| `variant` | `string` | `'default'` | Color variant (default, primary, secondary, success, warning, danger, muted) |

## Accessibility Guidelines

### Proper ARIA Labels

Always provide meaningful labels for screen readers:

```jsx
// Good: Descriptive label
<TrainIcon aria-label="Start model training" onClick={startTraining} />

// Bad: Generic or missing label
<TrainIcon onClick={startTraining} />
```

### Decorative vs Functional Icons

Mark decorative icons as hidden from screen readers:

```jsx
// Functional icon (conveys meaning)
<StatusDot status="running" />

// Decorative icon (visual enhancement only)
<Icon aria-hidden="true">
  <DecorativeIcon />
</Icon>
```

### Interactive Icons

Icons that can be clicked should be focusable and have proper event handling:

```jsx
// Interactive icon with keyboard support
<PlayIcon 
  onClick={handlePlay}
  aria-label="Start training session"
  focusable={true}
/>
```

### Color Contrast

Use semantic variants to ensure proper contrast:

```jsx
// Good: Uses semantic colors that meet WCAG AA
<StatusErrorIcon variant="danger" />
<StatusSuccessIcon variant="success" />

// Avoid: Custom colors that may not have sufficient contrast
<Icon className="text-pink-300">
  <SomeIcon />
</Icon>
```

## Migration from Emojis

To replace existing emojis:

1. **Find the emoji**: Identify emoji usage in your component
2. **Choose the icon**: Select appropriate icon from available components
3. **Add accessibility**: Include proper `aria-label` and other attributes
4. **Test**: Verify with screen readers and keyboard navigation

### Before (Emoji)
```jsx
<span className="text-4xl">🚂</span>
```

### After (Accessible Icon)
```jsx
<TrainIcon 
  size={32} 
  aria-label="Training section"
  variant="primary" 
/>
```

## Emoji Mapping

Common emoji replacements:

| Emoji | Icon Component | Usage |
|-------|----------------|-------|
| 📊 | `DashboardIcon` | Dashboard navigation |
| 🚂 | `TrainIcon` | Training/ML operations |
| 📋 | `HistoryIcon` | History/logs |
| ⭐ | `QualityIcon` | Quality metrics |
| 📐 | `GeometryIcon`, `AngleIcon` | Geometry/angle analysis |
| 🎨 | `CanvasIcon` | Canvas/visualization |
| ⚡ | `ActionIcon` | Actions/quick operations |
| 🔧 | `GeneratorIcon` | Tools/generation |
| ☀️ | `SunIcon` | Light theme |
| 🌙 | `MoonIcon` | Dark theme |
| 💻 | `MonitorIcon` | System theme |

## Best Practices

### 1. Consistent Sizing
Use the predefined size scale for visual consistency:

```jsx
// Good: Standard sizes
<Icon size={16} />  // Small
<Icon size={20} />  // Default
<Icon size={24} />  // Medium
<Icon size={32} />  // Large

// Avoid: Custom sizes
<Icon className="w-[23px] h-[23px]" />
```

### 2. Semantic Variants
Use variants that convey meaning:

```jsx
// Good: Semantic meaning
<StatusErrorIcon variant="danger" />
<StatusSuccessIcon variant="success" />

// Better: Component handles semantics
<StatusDot status="error" />  // Automatically uses danger variant
```

### 3. Keyboard Navigation
Ensure interactive icons are keyboard accessible:

```jsx
// Good: Focusable with proper label
<PlayIcon 
  onClick={handlePlay}
  focusable={true}
  aria-label="Start training session"
  onKeyDown={handleKeyDown}
/>
```

### 4. Loading States
Use loading icons for async operations:

```jsx
{isLoading ? (
  <LoadingIcon aria-label="Training in progress" />
) : (
  <PlayIcon 
    aria-label="Start training" 
    onClick={handleStart} 
  />
)}
```

## Testing Accessibility

Test your icon implementation with:

1. **Screen readers**: NVDA, JAWS, VoiceOver
2. **Keyboard navigation**: Tab, Enter, Space
3. **Color contrast**: Tools like WebAIM's contrast checker
4. **Focus indicators**: Ensure visible focus states

## Browser Support

- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- Full SVG support required
- CSS focus-visible support for enhanced focus indicators
