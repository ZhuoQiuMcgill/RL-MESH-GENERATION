# ADR 0002: Theme Strategy

**Status**: Draft  
**Last Updated**: 2025-01-07  
**Owner**: Frontend Team  
**Reviewers**: UI/UX Team, Architecture Team  

## Context

The current application has critical issues with theme management:

1. **Duplicated State**: Dark mode state exists separately in `App.jsx` and `NavHeader.jsx`
2. **Inconsistent DOM Manipulation**: App applies dark class to a div, NavHeader to `documentElement`
3. **Broken Light Mode**: Light theme colors exist but switching doesn't work properly
4. **Color Duplication**: Colors defined in 3 places (Tailwind config, @theme, :root)

This creates maintenance burden, inconsistent behavior, and broken functionality.

## Decision

We will implement a **centralized theme management system** with the following architecture:

### 1. Theme Context Provider

Create a single source of truth for theme state using React Context:

```typescript
// src/contexts/ThemeContext.tsx
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

const ThemeProvider = ({ children }) => {
  const [theme, setThemeState] = useState<'light' | 'dark'>(() => {
    // Initialize from localStorage or system preference
    const stored = localStorage.getItem('theme');
    if (stored) return stored as 'light' | 'dark';
    
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  const setTheme = (newTheme: 'light' | 'dark') => {
    setThemeState(newTheme);
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
    localStorage.setItem('theme', newTheme);
  };

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

### 2. Single Color System

Consolidate color definitions to use **only Tailwind v4's @theme** approach:

```css
/* src/index.css */
@import "tailwindcss";

@theme {
  --default-transition-duration: .15s;
  
  /* Light theme (default) */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8fafc;
  --color-text-primary: #1e293b;
  --color-text-secondary: #64748b;
  --color-border: #e2e8f0;
  --color-card: #ffffff;
  --color-card-hover: #f8fafc;
  
  /* Shared colors */
  --color-primary-start: #6366f1;
  --color-primary-end: #8b5cf6;
  --color-accent: #f472b6;
}

/* Dark theme overrides */
.dark {
  --color-bg-primary: #1e1b2e;
  --color-bg-secondary: #2a273a;
  --color-text-primary: #e2e8f0;
  --color-text-secondary: #94a3b8;
  --color-border: #374151;
  --color-card: #1f2937;
  --color-card-hover: #374151;
}
```

### 3. System Preference Detection

Implement automatic theme detection based on user's system preferences:

```javascript
// Detect system preference changes
useEffect(() => {
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const handleChange = (e) => {
    if (!localStorage.getItem('theme')) {
      setTheme(e.matches ? 'dark' : 'light');
    }
  };
  
  mediaQuery.addEventListener('change', handleChange);
  return () => mediaQuery.removeEventListener('change', handleChange);
}, []);
```

### 4. Persistence Strategy

- **localStorage**: Persist user's explicit theme choice
- **System Default**: Use system preference when no explicit choice exists
- **Session Continuity**: Maintain theme across page reloads

## Alternatives Considered

### Alternative 1: Redux/Zustand State Management
**Pros**: Powerful state management, DevTools support
**Cons**: Overkill for single theme state, additional dependency
**Verdict**: Rejected - React Context sufficient for this use case

### Alternative 2: CSS-only Solution
**Pros**: No JavaScript state management
**Cons**: Can't persist user preference, no system preference detection
**Verdict**: Rejected - Need JavaScript for full functionality

### Alternative 3: Keep Current Approach
**Pros**: No change required
**Cons**: Broken functionality, maintenance burden
**Verdict**: Rejected - Critical issues must be fixed

## Implementation Plan

### Phase 1: Context Implementation (Week 1)
1. Create `ThemeContext.tsx` with provider and hook
2. Wrap App component with ThemeProvider
3. Test basic theme switching functionality

### Phase 2: Component Migration (Week 1-2)
1. Replace theme state in `NavHeader.jsx` with context
2. Remove theme state from `App.jsx`
3. Update any other components using theme state

### Phase 3: Color System Consolidation (Week 2)
1. Remove duplicate color definitions from `tailwind.config.js`
2. Remove `:root` CSS variables
3. Ensure all components use new color system
4. Test both light and dark themes thoroughly

### Phase 4: Enhancement Features (Week 3)
1. Add system preference detection
2. Implement theme persistence
3. Add theme transition animations
4. Create theme testing utilities

## Benefits

### User Experience
- **Consistent Theming**: Single source of truth prevents synchronization issues
- **System Integration**: Respects user's OS theme preference
- **Persistence**: Remembers user's choice across sessions
- **Smooth Transitions**: Proper animation between theme changes

### Developer Experience
- **Single Source of Truth**: One place to manage theme state
- **Type Safety**: TypeScript interfaces for theme values
- **Easy Testing**: Centralized theme logic is easier to test
- **Maintainable**: No more hunting for theme-related bugs

### Performance
- **Reduced Bundle Size**: Eliminate duplicate color definitions
- **Efficient Updates**: Only re-render components that use theme context
- **Minimal DOM Manipulation**: Single class toggle on documentElement

## Risks and Mitigations

### Risk 1: Context Re-renders
**Risk**: Theme context changes might cause unnecessary re-renders
**Mitigation**: 
- Use React.memo for expensive components
- Split theme context if performance becomes an issue
- Implement context value memoization

### Risk 2: Migration Complexity  
**Risk**: Existing components might break during migration
**Mitigation**:
- Implement in phases with thorough testing
- Create backward compatibility layer if needed
- Use feature flags for gradual rollout

### Risk 3: Color System Changes
**Risk**: Consolidating color system might break existing styling
**Mitigation**:
- Audit all color usage before changes
- Create mapping between old and new color names
- Test all components in both themes

## Success Metrics

### Functionality
- ✅ Theme changes synchronize across all components
- ✅ Light and dark modes work correctly
- ✅ Theme persists across browser sessions
- ✅ System preference detection works
- ✅ No theme-related console errors

### Code Quality
- ✅ Zero duplicate theme state management
- ✅ Single source of truth for colors
- ✅ All theme-related code covered by tests
- ✅ TypeScript types for theme values

### Performance
- ✅ No performance regression from theme changes
- ✅ Smooth theme transition animations (<300ms)
- ✅ Bundle size reduction from eliminating duplicates

## Testing Strategy

### Unit Tests
```javascript
// Test theme context functionality
describe('ThemeContext', () => {
  test('initializes with system preference');
  test('toggleTheme switches between light and dark');
  test('setTheme updates theme correctly');
  test('persists theme to localStorage');
});
```

### Integration Tests
```javascript
// Test theme integration across components
describe('Theme Integration', () => {
  test('NavHeader theme button updates global theme');
  test('App component reflects theme changes');
  test('All pages respect current theme');
});
```

### Visual Tests
- Screenshot tests for light/dark mode consistency
- Manual testing of theme transitions
- Cross-browser theme compatibility testing

## Monitoring and Maintenance

### Metrics to Track
- Theme switching frequency
- User preference distribution (light vs dark)
- Theme-related error rates
- Performance impact of theme changes

### Regular Reviews
- Monthly review of theme-related issues
- Quarterly accessibility audit for both themes
- Annual evaluation of theme system architecture

## Related Decisions

- [ADR 0001: Architecture Goals](./0001-record-architecture-goals.md) - Establishes consistency as a key goal
- [ADR 0004: UI Kit Approach](./0004-ui-kit-approach.md) - UI components will use theme system

## References

- [Current Theme Issues Documentation](../styling-and-design/tailwind-and-theme.md)
- [Tailwind CSS Dark Mode Documentation](https://tailwindcss.com/docs/dark-mode)
- [React Context Best Practices](https://react.dev/learn/passing-data-deeply-with-context)

---

**Next Steps**: Begin Phase 1 implementation with ThemeContext creation and basic provider setup.
