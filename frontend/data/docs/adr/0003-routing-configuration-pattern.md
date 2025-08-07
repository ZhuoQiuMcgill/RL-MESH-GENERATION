# ADR 0003: Routing Configuration Pattern

**Status**: Draft  
**Last Updated**: 2025-01-07  
**Owner**: Frontend Team  
**Reviewers**: Architecture Team, Product Team  

## Context

The current routing system has several maintenance and consistency issues:

1. **Route Duplication**: Route paths are defined in both `App.jsx` and `NavHeader.jsx` separately
2. **Scattered Configuration**: Route logic spread across multiple files
3. **No Metadata System**: Missing route titles, permissions, breadcrumb information
4. **Maintenance Burden**: Adding new routes requires updates in multiple locations
5. **No Route Guards**: No centralized way to handle permissions or authentication

Current implementation:
```javascript
// App.jsx - Route definitions
<Routes>
  <Route path="/" element={<Dashboard />} />
  <Route path="/train" element={<Train />} />
  // ... 9 routes total
</Routes>

// NavHeader.jsx - Duplicated navigation items
const navItems = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/train', label: 'Train', icon: '🚂' },
  // ... same paths duplicated
]
```

## Decision

We will implement a **centralized route configuration system** with the following architecture:

### 1. Route Configuration Object

Create a single source of truth for all route information:

```typescript
// src/config/routes.ts
interface RouteConfig {
  path: string;
  label: string;
  icon: string;
  component: React.ComponentType;
  meta?: {
    title?: string;
    description?: string;
    requiresAuth?: boolean;
    roles?: string[];
    showInNav?: boolean;
    breadcrumbParent?: string;
  };
}

export const routes: RouteConfig[] = [
  {
    path: '/',
    label: 'Dashboard',
    icon: '📊',
    component: Dashboard,
    meta: {
      title: 'RL Mesh Generation - Dashboard',
      description: 'Overview of mesh generation and training status',
      showInNav: true,
    }
  },
  {
    path: '/train',
    label: 'Train',
    icon: '🚂',
    component: Train,
    meta: {
      title: 'RL Mesh Generation - Training',
      description: 'Configure and start reinforcement learning training',
      showInNav: true,
    }
  },
  {
    path: '/history',
    label: 'History',
    icon: '📋',
    component: History,
    meta: {
      title: 'RL Mesh Generation - Training History',
      description: 'View past training sessions and results',
      showInNav: true,
    }
  },
  // ... other routes
];
```

### 2. Route Generator Utilities

Create utilities to generate routes and navigation from configuration:

```typescript
// src/utils/routeUtils.ts
import { RouteObject } from 'react-router-dom';
import { routes } from '../config/routes';

export const generateRoutes = (): RouteObject[] => {
  return routes.map(route => ({
    path: route.path,
    element: <route.component />,
  }));
};

export const getNavigationItems = () => {
  return routes
    .filter(route => route.meta?.showInNav !== false)
    .map(route => ({
      path: route.path,
      label: route.label,
      icon: route.icon,
    }));
};

export const getRouteByPath = (path: string): RouteConfig | undefined => {
  return routes.find(route => route.path === path);
};

export const getBreadcrumbs = (currentPath: string): RouteConfig[] => {
  const current = getRouteByPath(currentPath);
  if (!current) return [];
  
  const breadcrumbs = [current];
  let parent = current.meta?.breadcrumbParent;
  
  while (parent) {
    const parentRoute = getRouteByPath(parent);
    if (parentRoute) {
      breadcrumbs.unshift(parentRoute);
      parent = parentRoute.meta?.breadcrumbParent;
    } else {
      break;
    }
  }
  
  return breadcrumbs;
};
```

### 3. Route Provider Context

Create a context for route-related information:

```typescript
// src/contexts/RouteContext.tsx
interface RouteContextType {
  currentRoute: RouteConfig | null;
  breadcrumbs: RouteConfig[];
  navigationItems: NavigationItem[];
}

export const RouteProvider = ({ children }: { children: ReactNode }) => {
  const location = useLocation();
  const currentRoute = getRouteByPath(location.pathname);
  const breadcrumbs = getBreadcrumbs(location.pathname);
  const navigationItems = getNavigationItems();

  // Update document title when route changes
  useEffect(() => {
    if (currentRoute?.meta?.title) {
      document.title = currentRoute.meta.title;
    }
  }, [currentRoute]);

  const value = {
    currentRoute,
    breadcrumbs,
    navigationItems,
  };

  return (
    <RouteContext.Provider value={value}>
      {children}
    </RouteContext.Provider>
  );
};
```

### 4. Updated App Structure

Simplify App.jsx to use generated routes:

```typescript
// src/App.tsx
import { generateRoutes } from './utils/routeUtils';

function App() {
  const routes = generateRoutes();

  return (
    <ApiProvider>
      <ThemeProvider>
        <Router>
          <RouteProvider>
            <div className="min-h-screen bg-bg-primary text-text-primary p-8">
              <NavHeader />
              <div className="mb-6">
                <Breadcrumb />
              </div>
              
              <main>
                <Routes>
                  {routes.map((route, index) => (
                    <Route key={index} {...route} />
                  ))}
                </Routes>
              </main>
            </div>
          </RouteProvider>
        </Router>
      </ThemeProvider>
    </ApiProvider>
  );
}
```

### 5. Updated Navigation Component

Simplify NavHeader to use route configuration:

```typescript
// src/components/NavHeader.tsx
import { useRoute } from '../contexts/RouteContext';

const NavHeader = () => {
  const { navigationItems, currentRoute } = useRoute();

  return (
    <header className="gradient-bg p-6 rounded-xl mb-8">
      {/* ... header content ... */}
      
      <nav className="mt-6">
        <div className="flex flex-wrap justify-center gap-2">
          {navigationItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 text-sm font-medium ${
                currentRoute?.path === item.path
                  ? 'bg-white/20 text-white'
                  : 'text-white/80 hover:text-white hover:bg-white/10'
              }`}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
};
```

## Alternatives Considered

### Alternative 1: File-based Routing (Next.js style)
**Pros**: Automatic route generation, convention over configuration
**Cons**: Requires build system changes, less flexible than explicit config
**Verdict**: Rejected - Too much change for current React Router setup

### Alternative 2: Route-specific Configuration Files
**Pros**: Each route owns its metadata, more modular
**Cons**: Distributed configuration, harder to get overview
**Verdict**: Rejected - Centralized config is simpler for this app size

### Alternative 3: Keep Current Approach
**Pros**: No changes needed, familiar to team
**Cons**: Maintenance burden continues, duplication issues
**Verdict**: Rejected - Issues outweigh benefits

### Alternative 4: React Router v6.4+ Data API
**Pros**: Modern React Router features, better data loading
**Cons**: Significant refactoring needed, learning curve
**Verdict**: Considered for future - stick with current React Router for now

## Implementation Plan

### Phase 1: Route Configuration Setup (Week 1)
1. Create `src/config/routes.ts` with all current routes
2. Create `src/utils/routeUtils.ts` with helper functions  
3. Test route generation utilities
4. Create basic TypeScript interfaces

### Phase 2: Route Context Integration (Week 1)
1. Create RouteProvider context
2. Add document title management
3. Integrate breadcrumb generation
4. Test context functionality

### Phase 3: Component Updates (Week 2)
1. Update App.jsx to use generated routes
2. Update NavHeader.jsx to use route context
3. Update Breadcrumb.jsx to use context
4. Remove duplicate route definitions

### Phase 4: Enhanced Features (Week 2-3)
1. Add route metadata (titles, descriptions)
2. Implement route guard placeholder system
3. Add analytics tracking points
4. Create route testing utilities

## Benefits

### Developer Experience
- **Single Source of Truth**: One place to define all route information
- **Type Safety**: TypeScript interfaces prevent configuration errors
- **Easy Maintenance**: Adding routes requires only one configuration change
- **Better Testing**: Centralized routing logic is easier to test

### User Experience  
- **Consistent Navigation**: Navigation always matches actual routes
- **Proper SEO**: Automatic document title management
- **Better Breadcrumbs**: Hierarchical navigation support
- **Future Route Guards**: Framework for authentication/permissions

### Code Quality
- **Reduced Duplication**: Eliminate duplicate route definitions
- **Better Organization**: Clear separation between config and implementation
- **Extensibility**: Easy to add new route features (guards, analytics, etc.)
- **Documentation**: Route metadata serves as documentation

## Risks and Mitigations

### Risk 1: Over-Engineering
**Risk**: Route config system might be too complex for simple app
**Mitigation**: 
- Start with basic implementation
- Add complexity only when needed
- Keep interfaces simple and extensible

### Risk 2: Breaking Changes
**Risk**: Refactoring might break existing routing
**Mitigation**:
- Implement in phases with testing
- Create comprehensive test suite
- Use TypeScript to catch compile-time errors

### Risk 3: Performance Impact
**Risk**: Additional abstraction might impact performance
**Mitigation**:
- Benchmark route generation performance
- Use React.memo for expensive components
- Consider route lazy loading for future optimization

## Success Metrics

### Code Quality
- ✅ Zero duplicate route definitions
- ✅ All routes defined in single configuration file
- ✅ TypeScript coverage for route types
- ✅ Comprehensive route testing

### Maintainability
- ✅ Adding new routes requires only config changes
- ✅ Navigation automatically updates with route changes
- ✅ Document titles update automatically
- ✅ Breadcrumbs work correctly on all pages

### User Experience
- ✅ Navigation state correctly reflects current page
- ✅ All routes accessible and functional
- ✅ Proper page titles for SEO and bookmarks
- ✅ Consistent navigation behavior

## Testing Strategy

### Unit Tests
```typescript
// Test route utilities
describe('routeUtils', () => {
  test('generateRoutes creates correct RouteObject array');
  test('getNavigationItems filters routes correctly');
  test('getRouteByPath finds correct route');
  test('getBreadcrumbs generates correct hierarchy');
});

// Test route context
describe('RouteContext', () => {
  test('provides current route information');
  test('updates document title on route change');
  test('generates correct breadcrumbs');
});
```

### Integration Tests  
```typescript
describe('Route Integration', () => {
  test('navigation items match actual routes');
  test('clicking navigation navigates to correct route');
  test('breadcrumbs update on navigation');
  test('document title updates correctly');
});
```

### Visual Tests
- Test all routes render correctly
- Verify navigation highlighting works
- Check breadcrumb display on all pages

## Future Enhancements

### Route Guards
```typescript
interface RouteConfig {
  // ... existing properties
  guards?: RouteGuard[];
}

interface RouteGuard {
  type: 'auth' | 'role' | 'custom';
  check: (context: RouteGuardContext) => boolean | Promise<boolean>;
  fallback: string; // redirect path if guard fails
}
```

### Route Analytics
```typescript
interface RouteConfig {
  // ... existing properties  
  analytics?: {
    trackPageView?: boolean;
    category?: string;
    additionalParams?: Record<string, any>;
  };
}
```

### Nested Routes
```typescript
interface RouteConfig {
  // ... existing properties
  children?: RouteConfig[];
  layout?: React.ComponentType;
}
```

## Related Decisions

- [ADR 0001: Architecture Goals](./0001-record-architecture-goals.md) - Supports maintainability goal
- [ADR 0002: Theme Strategy](./0002-theme-strategy.md) - Route context integrates with theme context

## References

- [React Router Documentation](https://reactrouter.com/)
- [Route Configuration Patterns](https://github.com/remix-run/react-router/discussions)
- [Single Source of Truth Principle](https://en.wikipedia.org/wiki/Single_source_of_truth)

---

**Next Steps**: Begin Phase 1 with route configuration file creation and utility function implementation.
