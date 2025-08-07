# ADR 0004: UI Kit Approach

**Status**: Draft  
**Last Updated**: 2025-01-07  
**Owner**: Frontend Team  
**Reviewers**: UI/UX Team, Architecture Team  

## Context

The current application has significant issues with UI component utilization and consistency:

1. **Orphaned Components**: 37% of UI components (7 out of 19) are unused, including core components like Button, FormInput, LoadingOverlay
2. **Ad-hoc Styling**: Components use inline styles and custom classes instead of shared components
3. **No Component Standards**: No documented API, usage guidelines, or design system
4. **Inconsistent Patterns**: Similar UI elements styled differently across pages
5. **Wasted Development Effort**: Time spent creating components that aren't being used

Current state example:
```javascript
// TrainingMonitor.jsx - Custom button styling instead of Button component
<button className="w-full px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark disabled:opacity-50 disabled:cursor-not-allowed">
  Load Full Mesh Data
</button>

// Unused Button.jsx component exists but isn't imported anywhere
```

This creates maintenance burden, inconsistent user experience, and inefficient development.

## Decision

We will establish a **comprehensive UI kit system** with the following architecture:

### 1. Component Audit and Consolidation

First, audit all existing UI components and consolidate into a cohesive system:

```typescript
// Component audit results and decisions
const componentAudit = {
  keep: [
    'Button',           // Most critical - used everywhere
    'FormInput',        // Essential for forms
    'FormSelect',       // Essential for dropdowns  
    'LoadingOverlay',   // Critical for async states
    'PanelCard',        // Layout consistency
    'EmptyState',       // User experience
  ],
  enhance: [
    'CompactStatusBar', // Improve and integrate with TrainingMonitor
  ],
  merge: [
    'examples.jsx',     // Merge useful patterns into main components
    'Train-refactored-example.jsx', // Extract patterns for Button variants
  ],
  archive: [], // None identified for removal yet
};
```

### 2. Component API Standardization

Establish consistent interfaces for all UI components:

```typescript
// Base component interfaces
interface BaseComponentProps {
  className?: string;
  children?: ReactNode;
  testId?: string;
}

interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'accent' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

interface FormInputProps extends BaseComponentProps {
  label?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  error?: string;
  required?: boolean;
  disabled?: boolean;
  type?: 'text' | 'email' | 'password' | 'number';
}
```

### 3. Enhanced Button Component System

Create a comprehensive button system with all necessary variants:

```typescript
// src/components/ui/Button.tsx
const Button = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  children,
  className = '',
  ...props
}: ButtonProps) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2';
  
  const variants = {
    primary: 'bg-primary-start hover:bg-primary-end text-white focus:ring-primary-start/50',
    secondary: 'bg-card hover:bg-card-hover text-text-primary border border-border focus:ring-border/50',
    accent: 'bg-accent hover:bg-accent/90 text-white focus:ring-accent/50',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500/50',
    ghost: 'text-text-primary hover:bg-card-hover focus:ring-border/50',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-lg',
  };

  const disabledStyles = 'opacity-50 cursor-not-allowed';

  const classes = cn(
    baseStyles,
    variants[variant],
    sizes[size],
    disabled && disabledStyles,
    className
  );

  return (
    <button
      className={classes}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Spinner className="mr-2" />}
      {icon && iconPosition === 'left' && <span className="mr-2">{icon}</span>}
      {children}
      {icon && iconPosition === 'right' && <span className="ml-2">{icon}</span>}
    </button>
  );
};
```

### 4. Component Documentation System

Implement comprehensive documentation for each component:

```typescript
// src/components/ui/Button.stories.tsx (Storybook-style docs)
export const ButtonExamples = {
  default: <Button>Default Button</Button>,
  variants: (
    <div className="space-y-4">
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="accent">Accent</Button>
      <Button variant="danger">Danger</Button>
      <Button variant="ghost">Ghost</Button>
    </div>
  ),
  withIcons: (
    <div className="space-x-4">
      <Button icon="⚡" iconPosition="left">Start Training</Button>
      <Button icon="🔄" loading>Loading...</Button>
    </div>
  ),
  disabled: <Button disabled>Disabled Button</Button>,
};

// Usage documentation
export const ButtonDocs = `
## Button Component

### Usage
\`\`\`tsx
import { Button } from '@/components/ui/Button';

<Button variant="primary" onClick={handleClick}>
  Click me
</Button>
\`\`\`

### Props
- variant: Controls button appearance
- size: Controls button size
- disabled: Disables the button
- loading: Shows loading spinner
- icon: Adds icon to button
`;
```

### 5. Component Integration Strategy

Create a phased approach to integrate components throughout the application:

```typescript
// Phase 1: High-impact replacements
const integrationPlan = {
  phase1: {
    // Replace custom buttons in TrainingMonitor
    'TrainingMonitor.jsx': [
      'Load Full Mesh Data button → <Button variant="primary">',
      'Find Reference Point button → <Button variant="accent">',
      'Start/Stop Training buttons → <Button variant="success/danger">',
    ],
    // Replace form elements in various pages
    'Train.jsx': [
      'Custom select elements → <FormSelect>',
      'Custom input fields → <FormInput>',
    ],
  },
  phase2: {
    // Add loading states where missing
    'All async operations': ['Add <LoadingOverlay> to async operations'],
    // Standardize card layouts
    'Dashboard.jsx': ['Replace custom cards with <PanelCard>'],
  },
  phase3: {
    // Add empty states
    'History.jsx': ['Add <EmptyState> for no training history'],
    'Quality.jsx': ['Add <EmptyState> for no quality data'],
  },
};
```

### 6. Theme Integration

Ensure all components integrate seamlessly with the theme system:

```typescript
// Component theme integration
const ThemeAwareButton = ({ variant, ...props }) => {
  const { theme } = useTheme();
  
  // Components automatically use CSS variables that change with theme
  const themeVariants = {
    primary: `bg-primary-start hover:bg-primary-end text-white`,
    secondary: `bg-card hover:bg-card-hover text-text-primary border border-border`,
    // ... other variants use theme-aware CSS variables
  };
  
  // No need for theme-specific logic - CSS variables handle it
  return <button className={themeVariants[variant]} {...props} />;
};
```

## Alternatives Considered

### Alternative 1: Third-party UI Library (Material-UI, Chakra UI)
**Pros**: Comprehensive components, accessibility built-in, well-tested
**Cons**: Bundle size increase, learning curve, customization limitations
**Verdict**: Rejected - Current custom components are sufficient with improvements

### Alternative 2: Headless UI Library (Radix, HeadlessUI)
**Pros**: Accessible primitives, custom styling flexibility
**Cons**: Additional dependency, migration effort for existing components
**Verdict**: Consider for future - good for complex components like modals

### Alternative 3: Keep Current Approach
**Pros**: No changes needed, team familiarity
**Cons**: Waste of existing component development, inconsistent UX
**Verdict**: Rejected - Issues outweigh benefits

### Alternative 4: Build Everything from Scratch
**Pros**: Full control, no dependencies
**Cons**: Significant development time, accessibility challenges
**Verdict**: Rejected - Existing components are good foundation

## Implementation Plan

### Phase 1: Component Audit and Enhancement (Week 1)
1. Audit all existing UI components and their usage
2. Enhance Button component with comprehensive variants
3. Improve FormInput and FormSelect with proper validation
4. Test enhanced components in isolation

### Phase 2: Documentation and Examples (Week 1-2)
1. Create component documentation with examples
2. Build component playground/showcase page
3. Document component APIs and props
4. Create usage guidelines and patterns

### Phase 3: Integration Rollout (Week 2-3)
1. Replace custom buttons in TrainingMonitor with Button component
2. Replace form elements throughout the app
3. Add loading states with LoadingOverlay
4. Standardize card layouts with PanelCard

### Phase 4: Advanced Components (Week 3-4)
1. Enhance CompactStatusBar for better integration
2. Add EmptyState components where appropriate
3. Create any missing components identified during integration
4. Performance optimization and bundle analysis

## Benefits

### User Experience
- **Consistent Interface**: All similar elements behave and look the same
- **Better Accessibility**: Standardized components with proper ARIA attributes
- **Improved Performance**: Optimized components with better rendering
- **Enhanced Interactions**: Consistent hover states, focus management, transitions

### Developer Experience
- **Faster Development**: Reusable components reduce development time
- **Better Maintainability**: Changes to components affect entire app consistently  
- **Clear Documentation**: Well-documented APIs reduce onboarding time
- **Type Safety**: TypeScript interfaces prevent usage errors

### Code Quality
- **Reduced Duplication**: Single implementation of common UI patterns
- **Better Testing**: Centralized component testing is more comprehensive
- **Easier Refactoring**: Changes to components propagate automatically
- **Style Consistency**: Centralized styling prevents style drift

## Risks and Mitigations

### Risk 1: Breaking Existing Functionality
**Risk**: Component changes might break existing pages
**Mitigation**: 
- Implement changes incrementally
- Maintain backward compatibility during transition
- Comprehensive testing before and after changes

### Risk 2: Over-engineering Components
**Risk**: Components become too complex or generic
**Mitigation**:
- Start with simple, focused implementations
- Add complexity only when needed by multiple use cases
- Regular review of component APIs for simplicity

### Risk 3: Performance Impact
**Risk**: Enhanced components might impact performance
**Mitigation**:
- Benchmark component performance
- Use React.memo for expensive components
- Optimize bundle splitting for component library

### Risk 4: Design System Rigidity  
**Risk**: Standardized components limit design flexibility
**Mitigation**:
- Design system supports customization through props
- Escape hatch with className prop for edge cases
- Regular review with design team for necessary changes

## Success Metrics

### Component Utilization
- ✅ >90% of UI components actively used (target: reduce orphaned components from 37% to <10%)
- ✅ All buttons use Button component (currently 0%)
- ✅ All forms use FormInput/FormSelect components
- ✅ All loading states use LoadingOverlay

### Code Quality
- ✅ Zero duplicate button/form implementations
- ✅ All components have TypeScript interfaces
- ✅ All components have usage documentation
- ✅ Component test coverage >80%

### User Experience
- ✅ Consistent interactive states across all components
- ✅ All components respect theme changes
- ✅ Improved accessibility scores
- ✅ No visual regressions from component changes

### Developer Experience
- ✅ Component integration reduces development time for new features
- ✅ Clear component documentation accessible to all team members
- ✅ Easy component discovery and usage
- ✅ Consistent component APIs across the system

## Testing Strategy

### Component Testing
```typescript
// Unit tests for individual components
describe('Button Component', () => {
  test('renders all variants correctly');
  test('handles disabled state properly');
  test('shows loading spinner when loading');
  test('handles click events');
  test('applies custom className');
  test('forwards refs correctly');
});

// Visual regression tests
describe('Button Visual Tests', () => {
  test('renders correctly in light theme');
  test('renders correctly in dark theme');
  test('all variants display consistently');
});
```

### Integration Testing
```typescript
// Test component integration in pages
describe('Component Integration', () => {
  test('TrainingMonitor uses Button components correctly');
  test('Form components work with validation');
  test('Loading states display properly');
  test('Theme changes affect all components');
});
```

### Accessibility Testing
```typescript
describe('Component Accessibility', () => {
  test('Button component has proper ARIA attributes');
  test('FormInput component has proper labels');
  test('Components are keyboard navigable');
  test('Components work with screen readers');
});
```

## Component Migration Guide

### Button Migration Examples
```typescript
// Before: Custom button styling
<button className="w-full px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark disabled:opacity-50 disabled:cursor-not-allowed">
  Load Full Mesh Data
</button>

// After: Button component
<Button 
  variant="primary" 
  disabled={!selectedMesh || isLoading}
  className="w-full"
>
  Load Full Mesh Data
</Button>

// Before: Training control buttons
<button className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
  Start Training
</button>

// After: Semantic button variants
<Button variant="success" className="w-full">
  Start Training
</Button>
```

### Form Migration Examples
```typescript
// Before: Custom form elements
<select className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2 text-text-primary">
  <option value="">Select a mesh...</option>
</select>

// After: FormSelect component
<FormSelect
  options={meshOptions}
  value={selectedMesh}
  onChange={setSelectedMesh}
  placeholder="Select a mesh..."
/>
```

## Future Enhancements

### Advanced Components
```typescript
// Modal/Dialog system
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

// Toast notification system
interface ToastProps {
  variant: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

// Data table component
interface DataTableProps<T> {
  data: T[];
  columns: TableColumn<T>[];
  pagination?: boolean;
  sorting?: boolean;
}
```

### Component Composition
```typescript
// Compound component patterns
<Card>
  <Card.Header>
    <Card.Title>Training Status</Card.Title>
  </Card.Header>
  <Card.Content>
    <StatusDisplay />
  </Card.Content>
  <Card.Footer>
    <Button variant="primary">Start</Button>
  </Card.Footer>
</Card>
```

## Related Decisions

- [ADR 0001: Architecture Goals](./0001-record-architecture-goals.md) - Supports maintainability and consistency goals
- [ADR 0002: Theme Strategy](./0002-theme-strategy.md) - Components integrate with theme system
- [ADR 0003: Routing Configuration](./0003-routing-configuration-pattern.md) - Route metadata supports component integration

## References

- [Current Component Inventory](../overview/code-inventory.md)
- [Design System Best Practices](https://designsystemsrepo.com/design-systems/)
- [React Component Patterns](https://reactpatterns.com/)
- [Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

---

**Next Steps**: Begin Phase 1 with component audit and Button component enhancement, followed by comprehensive documentation.
