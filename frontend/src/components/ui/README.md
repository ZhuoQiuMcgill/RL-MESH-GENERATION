# UI Components

This directory contains reusable React UI components built with Tailwind CSS, following the design system established in the existing pages. These components mirror the markup patterns found in the train and dashboard pages to enable declarative page composition.

## Components

### Button
Flexible button component with multiple variants and sizes.

```jsx
import { Button } from './components/ui'

// Basic usage
<Button>Click me</Button>

// Variants
<Button variant="primary">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="danger">Danger</Button>
<Button variant="success">Success</Button>
<Button variant="outline">Outline</Button>
<Button variant="ghost">Ghost</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="default">Default</Button>
<Button size="lg">Large</Button>

// States
<Button disabled>Disabled</Button>
```

### PanelCard
Container component for wrapping content in a card layout with optional title and subtitle.

```jsx
import { PanelCard } from './components/ui'

<PanelCard 
  title="Training Configuration" 
  subtitle="Set up your training parameters"
>
  <div>Card content goes here</div>
</PanelCard>
```

### CompactStatusBar
Component for displaying key-value status information in a compact format.

```jsx
import { CompactStatusBar } from './components/ui'

const statusItems = [
  { label: 'Status', value: 'Running', color: 'green-500' },
  { label: 'Episodes', value: '850/1000' },
  { label: 'Best Reward', value: '245.7' }
]

<CompactStatusBar items={statusItems} />
```

### FormInput
Styled input field with label and error handling.

```jsx
import { FormInput } from './components/ui'

<FormInput
  label="Learning Rate"
  type="number"
  step="0.0001"
  value={learningRate}
  onChange={(e) => setLearningRate(e.target.value)}
  error={validationError}
  placeholder="0.001"
/>
```

### FormSelect
Styled select dropdown with label and error handling.

```jsx
import { FormSelect } from './components/ui'

const options = [
  { value: 'ppo', label: 'PPO (Proximal Policy Optimization)' },
  { value: 'sac', label: 'SAC (Soft Actor-Critic)' }
]

<FormSelect
  label="Model Type"
  options={options}
  value={modelType}
  onChange={(e) => setModelType(e.target.value)}
  placeholder="Select a model type"
/>
```

### LoadingOverlay
Loading spinner component with optional overlay and different sizes.

```jsx
import { LoadingOverlay } from './components/ui'

// Full-screen overlay
<LoadingOverlay text="Training in progress..." />

// Inline spinner
<LoadingOverlay 
  overlay={false} 
  size="sm" 
  text="Loading..." 
/>
```

### EmptyState
Empty state component with icon, title, description, and optional action.

```jsx
import { EmptyState } from './components/ui'

<EmptyState
  icon="📊"
  title="No Training Data"
  description="Start your first training session to see results here."
  action={<Button variant="primary">Start Training</Button>}
/>
```

## Design System

All components follow the established design tokens:

- **Colors**: Uses CSS custom properties and Tailwind classes defined in `tailwind.config.js`
- **Typography**: Consistent text sizing and color hierarchy
- **Spacing**: Standard padding, margins, and gaps
- **Borders**: Consistent border radius and colors
- **Animations**: Smooth transitions and hover effects

## Usage Patterns

### Replacing Existing Markup

Instead of writing custom HTML/CSS, use these components:

```jsx
// Before
<div className="bg-card border border-border-custom rounded-xl p-6">
  <h3 className="text-xl font-semibold text-text-primary mb-4">Title</h3>
  <div>Content</div>
</div>

// After
<PanelCard title="Title">
  <div>Content</div>
</PanelCard>
```

### Consistent Form Styling

```jsx
// Before
<div>
  <label className="block text-text-secondary mb-2">Learning Rate</label>
  <input 
    type="number" 
    className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2 text-text-primary" 
  />
</div>

// After
<FormInput label="Learning Rate" type="number" />
```

## Examples

See `examples.jsx` for comprehensive usage examples of all components. You can temporarily add this route to your application to view all components in action:

```jsx
import UIExamples from './components/ui/examples'

// Add to your router
<Route path="/ui-examples" element={<UIExamples />} />
```

## Development

When creating new UI components:

1. Follow the existing patterns for props and styling
2. Use React.forwardRef for proper ref handling
3. Include TypeScript-style prop validation via defaultProps or PropTypes
4. Export from the main index file
5. Add usage examples to the examples file
6. Update this README
