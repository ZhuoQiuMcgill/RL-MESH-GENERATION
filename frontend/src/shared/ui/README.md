# Shared UI Kit

A comprehensive set of reusable React UI components built with Tailwind CSS, following consistent design patterns and providing standardized props across all components.

## Overview

This shared UI kit provides 12 essential components with consistent:
- **Variants**: Different visual styles (primary, secondary, success, warning, danger, etc.)
- **Sizes**: Multiple size options (xs, sm, default, lg, xl)
- **States**: Disabled and loading states where applicable
- **Design System**: Unified color scheme using CSS custom properties

## Components

### Button
Flexible button component with loading states and multiple variants.

```jsx
import { Button } from 'src/shared/ui'

// Basic usage
<Button>Click me</Button>

// With variants and sizes
<Button variant="primary" size="lg">Primary Large</Button>
<Button variant="danger" size="sm">Danger Small</Button>

// With loading state
<Button loading>Processing...</Button>

// Disabled state
<Button disabled>Can't click</Button>
```

**Props:**
- `variant`: 'primary' | 'secondary' | 'danger' | 'success' | 'warning' | 'outline' | 'ghost'
- `size`: 'xs' | 'sm' | 'default' | 'lg' | 'xl'
- `disabled`: boolean
- `loading`: boolean

### Input
Form input component with labels, error states, and validation.

```jsx
import { Input } from 'src/shared/ui'

<Input 
  label="Email Address"
  type="email"
  placeholder="Enter your email"
  error={errors.email}
  size="lg"
/>
```

**Props:**
- `label`: string
- `type`: HTML input type
- `size`: 'sm' | 'default' | 'lg'
- `variant`: 'default' | 'error' | 'success'
- `disabled`: boolean
- `error`: string (error message)

### Select
Dropdown select component with consistent styling.

```jsx
import { Select } from 'src/shared/ui'

const options = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' }
]

<Select
  label="Choose Option"
  options={options}
  placeholder="Select an option"
  error={errors.selection}
/>
```

**Props:**
- `label`: string
- `options`: Array of {value, label} objects
- `placeholder`: string
- `size`: 'sm' | 'default' | 'lg'
- `disabled`: boolean
- `error`: string

### Card
Container component for grouping related content.

```jsx
import { Card } from 'src/shared/ui'

<Card
  title="Training Configuration"
  subtitle="Configure your model parameters"
  variant="elevated"
  size="lg"
  headerAction={<Button size="sm">Edit</Button>}
  footer={<p>Last updated: Today</p>}
>
  <p>Card content goes here</p>
</Card>
```

**Props:**
- `title`: string
- `subtitle`: string
- `variant`: 'default' | 'elevated' | 'outlined' | 'ghost'
- `size`: 'sm' | 'default' | 'lg'
- `headerAction`: React node
- `footer`: React node

### Table
Comprehensive table component with headers, rows, and styling variants.

```jsx
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from 'src/shared/ui'

<Table variant="striped">
  <TableHeader>
    <TableRow>
      <TableHead>Name</TableHead>
      <TableHead>Status</TableHead>
      <TableHead>Actions</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    <TableRow>
      <TableCell>Training Session 1</TableCell>
      <TableCell>Running</TableCell>
      <TableCell>
        <Button size="sm">Stop</Button>
      </TableCell>
    </TableRow>
  </TableBody>
</Table>
```

**Props:**
- `variant`: 'default' | 'striped' | 'bordered'
- `size`: 'sm' | 'default' | 'lg' (affects cell padding)

### Badge
Small status indicators and labels.

```jsx
import { Badge } from 'src/shared/ui'

<Badge variant="success">Running</Badge>
<Badge variant="warning" size="lg">Pending</Badge>
<Badge variant="outline">Draft</Badge>
```

**Props:**
- `variant`: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info' | 'outline' | 'solid'
- `size`: 'xs' | 'sm' | 'default' | 'lg'

### Tabs
Tab navigation component with multiple styling variants.

```jsx
import { Tabs, TabsList, TabsTrigger, TabsContent } from 'src/shared/ui'

<Tabs defaultValue="overview" variant="underline">
  <TabsList>
    <TabsTrigger value="overview">Overview</TabsTrigger>
    <TabsTrigger value="settings">Settings</TabsTrigger>
    <TabsTrigger value="history">History</TabsTrigger>
  </TabsList>
  <TabsContent value="overview">
    <p>Overview content</p>
  </TabsContent>
  <TabsContent value="settings">
    <p>Settings content</p>
  </TabsContent>
</Tabs>
```

**Props:**
- `variant`: 'default' | 'underline' | 'pills'
- `size`: 'sm' | 'default' | 'lg'
- `defaultValue`: string (initial tab)
- `value`: string (controlled)
- `onValueChange`: function

### Modal
Dialog/modal component with backdrop and keyboard handling.

```jsx
import { Modal, Button } from 'src/shared/ui'

<Modal
  isOpen={isModalOpen}
  onClose={() => setIsModalOpen(false)}
  title="Confirm Action"
  size="lg"
  footer={
    <>
      <Button variant="outline" onClick={() => setIsModalOpen(false)}>
        Cancel
      </Button>
      <Button variant="danger" onClick={handleConfirm}>
        Delete
      </Button>
    </>
  }
>
  <p>Are you sure you want to delete this item?</p>
</Modal>
```

**Props:**
- `isOpen`: boolean
- `onClose`: function
- `title`: string
- `size`: 'sm' | 'default' | 'lg' | 'xl' | 'full'
- `footer`: React node
- `closeOnOverlayClick`: boolean
- `showCloseButton`: boolean

### Tooltip
Hover tooltips with positioning and styling options.

```jsx
import { Tooltip, Button } from 'src/shared/ui'

<Tooltip content="This action cannot be undone" placement="top" variant="warning">
  <Button variant="danger">Delete</Button>
</Tooltip>
```

**Props:**
- `content`: string | React node
- `placement`: 'top' | 'bottom' | 'left' | 'right'
- `variant`: 'default' | 'light' | 'error' | 'warning' | 'success'
- `size`: 'sm' | 'default' | 'lg'
- `delay`: number (ms)
- `disabled`: boolean

### Spinner
Loading spinner with different sizes and colors.

```jsx
import { Spinner } from 'src/shared/ui'

<Spinner size="lg" variant="primary" />
<Spinner size="sm" variant="white" />
```

**Props:**
- `variant`: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'white'
- `size`: 'xs' | 'sm' | 'default' | 'lg' | 'xl' | '2xl'

### Skeleton
Loading placeholder components.

```jsx
import { Skeleton, SkeletonText, SkeletonCircle, SkeletonCard } from 'src/shared/ui'

// Basic skeleton
<Skeleton className="w-full h-4" />

// Text skeleton with multiple lines
<SkeletonText lines={3} />

// Circle/avatar skeleton
<SkeletonCircle size="lg" />

// Card skeleton pattern
<SkeletonCard />
```

**Props:**
- `variant`: 'default' | 'light' | 'dark'
- `size`: 'sm' | 'default' | 'lg' | 'xl'
- `width`: string | number
- `height`: string | number
- `rounded`: boolean | 'sm' | 'md' | 'lg' | 'xl' | 'full'
- `animation`: 'pulse' | 'wave' | 'none'

### Alert
Alert and notification components.

```jsx
import { Alert } from 'src/shared/ui'

<Alert variant="success" title="Success!" dismissible onDismiss={handleDismiss}>
  Your changes have been saved successfully.
</Alert>

<Alert variant="warning" size="lg">
  Please review your settings before proceeding.
</Alert>
```

**Props:**
- `variant`: 'default' | 'info' | 'success' | 'warning' | 'danger' | 'destructive'
- `size`: 'sm' | 'default' | 'lg'
- `title`: string
- `icon`: React node (overrides default)
- `dismissible`: boolean
- `onDismiss`: function

## Usage Patterns

### Replacing Existing Utility Classes

Replace ad-hoc styles with these components incrementally:

```jsx
// Before
<div className="bg-card border border-border-custom rounded-xl p-6">
  <h3 className="text-xl font-semibold text-text-primary mb-4">Training Status</h3>
  <div className="space-y-4">
    <div>
      <label className="block text-text-secondary mb-2">Learning Rate</label>
      <input className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2" />
    </div>
    <button className="bg-accent text-white px-4 py-2 rounded-lg">
      Start Training
    </button>
  </div>
</div>

// After
<Card title="Training Status">
  <div className="space-y-4">
    <Input label="Learning Rate" />
    <Button variant="primary">Start Training</Button>
  </div>
</Card>
```

### Form Composition

```jsx
import { Card, Input, Select, Button, Alert } from 'src/shared/ui'

<Card title="Training Configuration" subtitle="Set up your model parameters">
  <form onSubmit={handleSubmit} className="space-y-4">
    <Input
      label="Learning Rate"
      type="number"
      step="0.0001"
      value={formData.learningRate}
      onChange={(e) => setFormData({...formData, learningRate: e.target.value})}
      error={errors.learningRate}
    />
    
    <Select
      label="Model Type"
      options={modelOptions}
      value={formData.modelType}
      onChange={(e) => setFormData({...formData, modelType: e.target.value})}
      error={errors.modelType}
    />
    
    {errors.general && (
      <Alert variant="danger" dismissible onDismiss={() => setErrors({...errors, general: null})}>
        {errors.general}
      </Alert>
    )}
    
    <div className="flex gap-3">
      <Button type="button" variant="outline">Reset</Button>
      <Button type="submit" loading={isSubmitting}>
        Start Training
      </Button>
    </div>
  </form>
</Card>
```

### Table with Actions

```jsx
import { Card, Table, TableHeader, TableBody, TableRow, TableHead, TableCell, Badge, Button } from 'src/shared/ui'

<Card title="Training Sessions">
  <Table variant="striped">
    <TableHeader>
      <TableRow>
        <TableHead>Session ID</TableHead>
        <TableHead>Status</TableHead>
        <TableHead>Progress</TableHead>
        <TableHead>Actions</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {sessions.map(session => (
        <TableRow key={session.id}>
          <TableCell>{session.id}</TableCell>
          <TableCell>
            <Badge 
              variant={session.status === 'running' ? 'success' : 'secondary'}
            >
              {session.status}
            </Badge>
          </TableCell>
          <TableCell>{session.progress}%</TableCell>
          <TableCell>
            <div className="flex gap-2">
              <Button size="sm" variant="outline">View</Button>
              {session.status === 'running' ? (
                <Button size="sm" variant="danger">Stop</Button>
              ) : (
                <Button size="sm" variant="primary">Start</Button>
              )}
            </div>
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
</Card>
```

## Design System Integration

All components use the existing CSS custom properties defined in your theme:

- `--color-text-primary` / `text-text-primary`
- `--color-text-secondary` / `text-text-secondary`
- `--color-bg-primary` / `bg-bg-primary`
- `--color-bg-secondary` / `bg-bg-secondary`
- `--color-card` / `bg-card`
- `--color-card-hover` / `bg-card-hover`
- `--color-border-custom` / `border-border-custom`
- `--color-accent` / `bg-accent`

## Migration Strategy

1. **Incremental Replacement**: Replace existing utility classes with components page by page
2. **Component First**: Use these components for all new features
3. **Consistent Patterns**: Apply the same props pattern (variant, size, disabled, loading) everywhere
4. **Type Safety**: Consider adding TypeScript definitions for better development experience

## Development Guidelines

When extending or modifying components:

1. **Maintain Consistency**: Keep the same prop patterns across all components
2. **Forward Refs**: All components use `React.forwardRef` for proper ref handling
3. **Accessibility**: Include appropriate ARIA labels and keyboard navigation
4. **Performance**: Use `cn` utility for efficient class name merging
5. **Documentation**: Update this README when adding new variants or props

## Examples

For comprehensive examples and live demos of all components, see the examples file:

```jsx
import UIExamples from 'src/shared/ui/examples'

// Add to your router for development/testing
<Route path="/ui-examples" element={<UIExamples />} />
```
