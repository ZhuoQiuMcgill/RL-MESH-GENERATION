# Design System

This document outlines the comprehensive design system for the RL Mesh Generation frontend application, including all design tokens, components, and usage guidelines.

## Overview

The design system is built around a token-based architecture that provides:
- **Single source of truth** for all design values
- **Consistent theming** with light/dark mode support
- **Scalable design tokens** organized by category
- **Component-level tokens** for reusable UI patterns

## Design Tokens

All design tokens are defined in `src/styles/design-tokens.css` and referenced through CSS custom properties and Tailwind CSS configuration.

### Color System

#### Primary Brand Colors
```css
--color-primary-start: #6366f1    /* Indigo-500 - primary gradient start */
--color-primary-end: #8b5cf6      /* Violet-500 - primary gradient end */
--color-accent: #f472b6           /* Pink-400 - accent/highlight color */
```

#### Theme Colors
The system supports both light and dark themes with automatic switching:

**Light Theme:**
- Background Primary: `#ffffff`
- Background Secondary: `#f8fafc`
- Background Card: `#ffffff`
- Text Primary: `#1e293b`
- Text Secondary: `#64748b`
- Border: `#e2e8f0`

**Dark Theme (Default):**
- Background Primary: `#1e1b2e`
- Background Secondary: `#2a273a`
- Background Card: `#1f2937`
- Text Primary: `#e2e8f0`
- Text Secondary: `#94a3b8`
- Border: `#374151`

#### Semantic Colors
```css
--color-success: #10b981    /* Green for success states */
--color-warning: #f59e0b    /* Orange for warning states */
--color-error: #ef4444      /* Red for error states */
--color-info: #3b82f6       /* Blue for informational states */
```

**Usage:**
```jsx
// Using with Tailwind classes
<div className="bg-bg-primary text-text-primary">
<button className="bg-success text-white">Success</button>
<span className="text-error">Error message</span>

// Using with CSS custom properties
<div style={{backgroundColor: 'var(--color-bg-primary)'}}>
```

### Spacing Scale

Built on a base unit of `0.25rem` (4px) for consistent spacing:

```css
--space-xs: 0.25rem     /* 4px */
--space-sm: 0.5rem      /* 8px */  
--space-md: 0.75rem     /* 12px */
--space-lg: 1rem        /* 16px */
--space-xl: 1.5rem      /* 24px */
--space-2xl: 2rem       /* 32px */
--space-3xl: 3rem       /* 48px */
--space-4xl: 4rem       /* 64px */
```

**Usage:**
```jsx
// With Tailwind utilities
<div className="p-lg m-xl gap-md">
<div className="space-y-lg">

// With CSS custom properties  
<div style={{padding: 'var(--space-lg)'}}>
```

### Border Radius

Consistent border radius values across all components:

```css
--radius-sm: 0.375rem    /* 6px */
--radius-md: 0.5rem      /* 8px */
--radius-lg: 0.75rem     /* 12px */
--radius-xl: 1rem        /* 16px */
```

**Usage:**
```jsx
<div className="rounded-lg">        <!-- Uses --radius-lg -->
<button className="rounded-md">     <!-- Uses --radius-md -->
<div className="rounded-xl">        <!-- Uses --radius-xl -->
```

### Typography

#### Font Families
```css
--font-family-sans: system-ui, Avenir, Helvetica, Arial, sans-serif
--font-family-mono: 'Courier New', Consolas, Monaco, monospace
```

#### Font Sizes
```css
--text-xs: 0.75rem      /* 12px */
--text-sm: 0.875rem     /* 14px */
--text-base: 1rem       /* 16px */
--text-lg: 1.125rem     /* 18px */
--text-xl: 1.25rem      /* 20px */
--text-2xl: 1.5rem      /* 24px */
--text-3xl: 1.875rem    /* 30px */
--text-4xl: 2.25rem     /* 36px */
```

#### Font Weights
```css
--font-weight-normal: 400
--font-weight-medium: 500
--font-weight-semibold: 600
--font-weight-bold: 700
```

#### Line Heights
```css
--leading-tight: 1.25      /* For headings */
--leading-normal: 1.5      /* For body text */
--leading-relaxed: 1.625   /* For large text blocks */
```

**Typography Usage:**
```jsx
<h1 className="text-4xl font-bold leading-tight">
<h2 className="text-3xl font-semibold">
<p className="text-base leading-normal">
<small className="text-sm text-text-secondary">
```

### Shadows

Layered shadow system for depth and elevation:

```css
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05)
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)
--shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)
```

**Usage:**
```jsx
<div className="shadow-md">     <!-- Cards, panels -->
<div className="shadow-lg">     <!-- Modals, dropdowns -->
<div className="shadow-xl">     <!-- Overlays, toasts -->
```

## Layout System

### Container Max-Widths

Responsive container system for consistent page layouts:

```css
--container-sm: 640px     /* Small screens */
--container-md: 768px     /* Medium screens */
--container-lg: 1024px    /* Large screens */
--container-xl: 1280px    /* Extra large screens */
--container-2xl: 1536px   /* 2X large screens */
```

**Usage:**
```jsx
<div className="max-w-xl mx-auto">        <!-- Centered container -->
<div className="max-w-6xl mx-auto px-lg"> <!-- With padding -->
```

### Component Heights

Standard heights for consistent component sizing:

```css
--height-nav-header: 4rem      /* 64px - Navigation bar */
--height-status-bar: 3rem      /* 48px - Status/breadcrumb bar */
--height-button-sm: 2rem       /* 32px - Small buttons */
--height-button-md: 2.5rem     /* 40px - Default buttons */
--height-button-lg: 3rem       /* 48px - Large buttons */
--height-input: 2.5rem         /* 40px - Form inputs */
```

### Grid Systems

Pre-configured grid patterns for common layouts:

```css
--grid-cols-dashboard: repeat(auto-fit, minmax(280px, 1fr))  /* Dashboard cards */
--grid-cols-form: repeat(auto-fit, minmax(200px, 1fr))       /* Form fields */
```

**Usage:**
```jsx
<!-- Dashboard grid -->
<div className="grid" style={{gridTemplateColumns: 'var(--grid-cols-dashboard)', gap: 'var(--spacing-grid-gap)'}}>

<!-- Using Tailwind utilities -->
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-lg">
```

## Component Tokens

### Cards

All card-related tokens for consistent card styling:

```css
--card-padding: var(--space-xl)           /* 24px */
--card-border-radius: var(--radius-xl)    /* 16px */
--card-shadow: var(--shadow-md)
```

**Card Component Usage:**
```jsx
<!-- Using utility classes -->
<div className="bg-card border border-border-custom rounded-xl p-xl shadow-md">

<!-- Using component class -->
<div className="card-base">
  <h3 className="text-xl font-semibold text-text-primary">Title</h3>
  <p className="text-text-secondary">Content</p>
</div>
```

### Buttons

Button-specific tokens for consistent button styling:

```css
--button-padding-x: var(--space-lg)       /* 16px horizontal */
--button-padding-y: var(--space-md)       /* 12px vertical */
--button-border-radius: var(--radius-md)  /* 8px */
--button-font-weight: var(--font-weight-medium)
--button-transition: var(--transition-fast)
```

**Button Variants:**
```jsx
<!-- Primary button -->
<button className="btn-primary">Primary Action</button>

<!-- Using Button component -->
<Button variant="primary">Primary Action</Button>
<Button variant="secondary">Secondary Action</Button>
<Button variant="danger">Delete</Button>
<Button variant="success">Save</Button>
```

### Forms

Form-related tokens for consistent input styling:

```css
--form-gap: var(--space-lg)               /* 16px gap between form fields */
--input-height: var(--height-input)       /* 40px */
--input-border-radius: var(--radius-md)   /* 8px */
--input-padding-x: var(--space-md)        /* 12px horizontal */
--input-padding-y: var(--space-sm)        /* 8px vertical */
```

**Form Usage:**
```jsx
<!-- Form layout -->
<form className="space-y-lg">
  <div>
    <label className="block text-text-secondary mb-sm">Label</label>
    <input className="form-input w-full" type="text" />
  </div>
</form>

<!-- Using FormInput component -->
<FormInput 
  label="Learning Rate"
  type="number"
  placeholder="0.001"
/>
```

### Navigation

Navigation-specific tokens:

```css
--nav-header-height: var(--height-nav-header)  /* 64px */
--nav-header-padding-x: var(--space-xl)        /* 24px */
--nav-header-padding-y: var(--space-md)        /* 12px */
```

## Animations & Transitions

Consistent timing functions for smooth interactions:

```css
--transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1)     /* Quick interactions */
--transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1)   /* Standard transitions */
--transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1)     /* Slow, deliberate animations */
```

**Usage:**
```jsx
<!-- With Tailwind utilities -->
<div className="transition-all duration-fast hover:scale-105">

<!-- With CSS custom properties -->
<div style={{transition: 'all var(--transition-fast)'}}>
```

## Theme Switching

The design system supports automatic theme switching based on the `.dark` class on the document root:

**Implementation:**
```javascript
// Toggle dark mode
document.documentElement.classList.toggle('dark')

// Check current theme
const isDark = document.documentElement.classList.contains('dark')
```

**CSS Implementation:**
```css
/* Default (Dark) Theme */
:root {
  --color-bg-primary: var(--color-bg-primary-dark);
  --color-text-primary: var(--color-text-primary-dark);
}

/* Light Theme */
:root:not(.dark) {
  --color-bg-primary: var(--color-bg-primary-light);
  --color-text-primary: var(--color-text-primary-light);
}
```

## Usage Guidelines

### 1. Prefer Design Tokens Over Hard-coded Values

❌ **Don't:**
```jsx
<div style={{backgroundColor: '#1f2937', padding: '24px'}}>
```

✅ **Do:**
```jsx
<div className="bg-card p-xl">
<!-- OR -->
<div style={{
  backgroundColor: 'var(--color-bg-card)', 
  padding: 'var(--space-xl)'
}}>
```

### 2. Use Semantic Color Names

❌ **Don't:**
```jsx
<div className="bg-gray-800 text-gray-200">
```

✅ **Do:**
```jsx
<div className="bg-card text-text-primary">
```

### 3. Maintain Consistent Spacing

❌ **Don't:**
```jsx
<div className="p-3 m-5 gap-2">  <!-- Mixed spacing values -->
```

✅ **Do:**
```jsx
<div className="p-md m-lg gap-sm"> <!-- Consistent spacing scale -->
```

### 4. Use Component Tokens for Reusable Patterns

❌ **Don't:**
```jsx
<div className="bg-card border border-border-custom rounded-xl p-6 shadow-md">
```

✅ **Do:**
```jsx
<div className="card-base">
<!-- OR -->
<PanelCard>
```

## Component Library Integration

The design tokens are fully integrated with the component library in `src/components/ui/`:

- **Button** - Uses button tokens for consistent sizing and styling
- **PanelCard** - Uses card tokens for consistent card layouts  
- **FormInput** - Uses form tokens for consistent input styling
- **All components** inherit the theme system automatically

## Development Workflow

### Adding New Tokens

1. **Add to design-tokens.css:**
   ```css
   --new-component-padding: var(--space-lg);
   ```

2. **Update tailwind.config.js if needed:**
   ```javascript
   spacing: {
     'component-padding': 'var(--new-component-padding)',
   }
   ```

3. **Use in components:**
   ```jsx
   <div className="p-component-padding">
   ```

4. **Document in this file** with usage examples

### Modifying Existing Tokens

1. Update the token value in `design-tokens.css`
2. Test all affected components
3. Update documentation if the usage changes
4. Consider impact on both light and dark themes

## File Structure

```
src/
├── styles/
│   └── design-tokens.css          # Single source of truth
├── index.css                      # Tailwind imports + base styles
├── components/ui/                 # Component library
│   ├── Button.jsx                 # Uses button tokens
│   ├── PanelCard.jsx             # Uses card tokens
│   └── FormInput.jsx             # Uses form tokens
└── tailwind.config.js            # Tailwind integration
```

## Migration Notes

This design system consolidates and replaces:
- Duplicate color definitions in `tailwind.config.js`
- Redundant CSS custom properties in `index.css`
- Inconsistent component styling patterns
- Hard-coded design values throughout the codebase

All existing components should gradually migrate to use the consolidated design tokens for better consistency and maintainability.
