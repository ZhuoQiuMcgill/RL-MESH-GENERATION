# Migration Implementation Guide

## Overview

This guide provides detailed technical implementation guidance for each PR in the migration plan. It includes code examples, specific steps, and validation criteria to ensure successful migration.

## Pre-Migration Setup

Before starting any migration PR, ensure the following:

1. **Environment Setup**:
   ```bash
   # Install dependencies
   npm install
   
   # Verify build works
   npm run build
   
   # Run existing tests
   npm run test
   
   # Create migration branch
   git checkout -b migration/setup
   ```

2. **Baseline Documentation**:
   - Document current functionality
   - Take screenshots of current UI
   - Record current bundle size
   - Document current test coverage

## PR1 Implementation Guide

### Step 1.1: Complete Documentation Review

**Checklist**:
- [ ] All ADRs are complete and reviewed
- [ ] Architecture documentation is current
- [ ] Gap analysis is up to date
- [ ] API documentation matches current implementation

**Implementation**:
```bash
# Review documentation completeness
find data/docs -name "*.md" -exec grep -l "TODO\|FIXME\|Draft" {} \;

# Validate ADR links
grep -r "\[ADR" data/docs/adr/ | grep -v "\.md:"
```

### Step 1.2: Route Configuration Infrastructure

**Create Route Types**:
```typescript
// src/types/routes.types.ts
export interface RouteConfig {
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

export interface NavigationItem {
  path: string;
  label: string;
  icon: string;
}
```

**Create Route Configuration**:
```typescript
// src/config/routes.ts
import { RouteConfig } from '../types/routes.types';
import Dashboard from '../pages/Dashboard';
import Train from '../pages/Train';
import History from '../pages/History';
// ... other imports

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
  // ... continue for all routes
];
```

**Create Route Utilities**:
```typescript
// src/utils/routeUtils.ts
import { RouteObject } from 'react-router-dom';
import { routes } from '../config/routes';
import { RouteConfig, NavigationItem } from '../types/routes.types';

export const generateRoutes = (): RouteObject[] => {
  return routes.map(route => ({
    path: route.path,
    element: React.createElement(route.component),
  }));
};

export const getNavigationItems = (): NavigationItem[] => {
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
```

**Validation Tests**:
```typescript
// src/tests/routes.test.ts
import { routes } from '../config/routes';
import { generateRoutes, getNavigationItems } from '../utils/routeUtils';

describe('Route Configuration', () => {
  test('all routes have required properties', () => {
    routes.forEach(route => {
      expect(route.path).toBeDefined();
      expect(route.label).toBeDefined();
      expect(route.component).toBeDefined();
    });
  });

  test('generated routes match configuration', () => {
    const generated = generateRoutes();
    expect(generated).toHaveLength(routes.length);
  });

  test('navigation items filter correctly', () => {
    const navItems = getNavigationItems();
    const expectedCount = routes.filter(r => r.meta?.showInNav !== false).length;
    expect(navItems).toHaveLength(expectedCount);
  });
});
```

## PR2 Implementation Guide

### Step 2.1: Theme Provider Implementation

**Theme Context**:
```tsx
// src/contexts/ThemeContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const [theme, setThemeState] = useState<'light' | 'dark'>(() => {
    // Initialize from localStorage or system preference
    const stored = localStorage.getItem('theme');
    if (stored && (stored === 'light' || stored === 'dark')) {
      return stored;
    }
    
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

  // Apply theme class on mount and change
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // Listen for system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const value = {
    theme,
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
```

### Step 2.2: AppShell Implementation

**AppShell Component**:
```tsx
// src/shared/layout/AppShell.jsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Breadcrumb from './Breadcrumb';

const AppShell = () => {
  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      <Header />
      
      <div className="container mx-auto px-8 py-6">
        <div className="mb-6">
          <Breadcrumb />
        </div>
        
        <main>
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppShell;
```

### Step 2.3: Enhanced Button Component

**Button Component with All Variants**:
```tsx
// src/shared/ui/Button.jsx
import React from 'react';
import { cn } from '../../utils/cn'; // Utility for combining classes

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'accent' | 'danger' | 'ghost' | 'success';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

const Button = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconPosition = 'left',
  children,
  className = '',
  disabled = false,
  ...props
}: ButtonProps) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-primary-start hover:bg-primary-end text-white focus:ring-primary-start/50',
    secondary: 'bg-card hover:bg-card-hover text-text-primary border border-border focus:ring-border/50',
    accent: 'bg-accent hover:bg-accent/90 text-white focus:ring-accent/50',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500/50',
    success: 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500/50',
    ghost: 'text-text-primary hover:bg-card-hover focus:ring-border/50',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-lg',
  };

  const classes = cn(
    baseStyles,
    variants[variant],
    sizes[size],
    className
  );

  const isDisabled = disabled || loading;

  return (
    <button
      className={classes}
      disabled={isDisabled}
      {...props}
    >
      {loading && <Spinner className="mr-2" size={size} />}
      {icon && iconPosition === 'left' && !loading && (
        <span className="mr-2">{icon}</span>
      )}
      {children}
      {icon && iconPosition === 'right' && !loading && (
        <span className="ml-2">{icon}</span>
      )}
    </button>
  );
};

// Spinner component for loading states
const Spinner = ({ className = '', size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  return (
    <div className={cn('animate-spin rounded-full border-2 border-current border-t-transparent', sizeClasses[size], className)} />
  );
};

export default Button;
```

### Step 2.4: Updated App Structure

**Modified App.jsx**:
```jsx
// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ApiProvider } from './context/ApiProvider';
import { ThemeProvider } from './contexts/ThemeContext';
import AppShell from './shared/layout/AppShell';

// Keep existing route imports for backward compatibility
import Dashboard from './pages/Dashboard';
import Train from './pages/Train';
// ... other imports

function App() {
  return (
    <ThemeProvider>
      <ApiProvider>
        <Router>
          <Routes>
            <Route path="/" element={<AppShell />}>
              <Route index element={<Dashboard />} />
              <Route path="train" element={<Train />} />
              {/* ... other routes */}
            </Route>
          </Routes>
        </Router>
      </ApiProvider>
    </ThemeProvider>
  );
}

export default App;
```

## PR3 Implementation Guide

### Step 3.1: API Client Normalization

**Normalized API Client**:
```javascript
// src/core/api/ApiClient.js
import { config } from '../../config/environment';

class ApiClient {
  constructor() {
    if (ApiClient.instance) {
      return ApiClient.instance;
    }
    
    this.baseUrl = config.api.baseUrl;
    this.timeout = config.api.timeout;
    
    // Validate URL on initialization
    try {
      new URL(this.baseUrl);
    } catch (error) {
      throw new Error(`Invalid API base URL: ${this.baseUrl}`);
    }
    
    if (config.api.enableDebug) {
      console.log('API Client initialized:', {
        baseUrl: this.baseUrl,
        timeout: this.timeout,
      });
    }
    
    ApiClient.instance = this;
  }

  async request(endpoint, options = {}, customTimeout = null) {
    const url = `${this.baseUrl}${endpoint}`;
    const timeout = customTimeout || this.timeout;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }
      
      throw error;
    }
  }

  // Keep all existing API methods with same signatures
  async getTrainingStatus() {
    return await this.request('/training/status');
  }

  async startTraining(config) {
    return await this.request('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // ... continue with all existing methods
}

export default ApiClient;
```

### Step 3.2: Environment Configuration

**Configuration System**:
```typescript
// src/config/environment.ts
interface ApiConfig {
  baseUrl: string;
  timeout: number;
  pollingInterval: number;
  retryCount: number;
  retryDelay: number;
  enableDebug: boolean;
}

interface AppConfig {
  api: ApiConfig;
  app: {
    version: string;
    environment: 'development' | 'staging' | 'production';
    buildTime: string;
  };
}

const defaultConfig: AppConfig = {
  api: {
    baseUrl: 'http://localhost:8000',
    timeout: 10000,
    pollingInterval: 2000,
    retryCount: 1,
    retryDelay: 3000,
    enableDebug: false,
  },
  app: {
    version: import.meta.env.PACKAGE_VERSION || '1.0.0',
    environment: (import.meta.env.MODE as any) || 'development',
    buildTime: import.meta.env.VITE_BUILD_TIME || new Date().toISOString(),
  },
};

export function loadConfig(): AppConfig {
  const envConfig = {
    baseUrl: import.meta.env.VITE_API_BASE_URL,
    timeout: parseInt(import.meta.env.VITE_API_TIMEOUT) || undefined,
    pollingInterval: parseInt(import.meta.env.VITE_POLLING_INTERVAL) || undefined,
    retryCount: parseInt(import.meta.env.VITE_RETRY_COUNT) || undefined,
    retryDelay: parseInt(import.meta.env.VITE_RETRY_DELAY) || undefined,
    enableDebug: import.meta.env.VITE_ENABLE_DEBUG === 'true',
  };

  // Remove undefined values
  Object.keys(envConfig).forEach(key => {
    if (envConfig[key] === undefined) {
      delete envConfig[key];
    }
  });

  const config = {
    api: { ...defaultConfig.api, ...envConfig },
    app: defaultConfig.app,
  };

  if (config.api.enableDebug) {
    console.log('Loaded configuration:', config);
  }

  return config;
}

export const config = loadConfig();
```

**Environment Files**:
```bash
# .env.example
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=10000
VITE_POLLING_INTERVAL=2000
VITE_RETRY_COUNT=1
VITE_RETRY_DELAY=3000
VITE_ENABLE_DEBUG=false

# Build Configuration
VITE_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
```

## Validation and Testing Guidelines

### For Each PR

1. **Before Implementation**:
   ```bash
   # Create feature branch
   git checkout -b pr/1-docs-routing-scaffold
   
   # Run baseline tests
   npm run test
   npm run build
   npm run lint
   ```

2. **During Implementation**:
   ```bash
   # Continuous testing
   npm run test -- --watch
   
   # Check build
   npm run build
   
   # Lint code
   npm run lint --fix
   ```

3. **Before PR Submission**:
   ```bash
   # Full test suite
   npm run test:all
   
   # Build validation
   npm run build:analyze
   
   # E2E tests
   npm run e2e
   
   # Visual regression tests (if available)
   npm run test:visual
   ```

### Integration Testing Strategy

**Component Integration Tests**:
```javascript
// Example integration test
describe('Theme Integration', () => {
  test('theme changes propagate to all components', async () => {
    render(
      <ThemeProvider>
        <AppShell />
      </ThemeProvider>
    );
    
    // Test initial theme
    expect(document.documentElement).toHaveClass('dark');
    
    // Toggle theme
    const themeButton = screen.getByLabelText('Toggle theme');
    fireEvent.click(themeButton);
    
    // Verify theme change
    await waitFor(() => {
      expect(document.documentElement).not.toHaveClass('dark');
    });
  });
});
```

## Common Issues and Solutions

### Issue: Import Path Conflicts
**Problem**: Moving files breaks existing imports
**Solution**: 
1. Create temporary re-export files
2. Use IDE refactoring tools
3. Update imports gradually

### Issue: Theme Flash on Load
**Problem**: Brief flash of wrong theme on page load
**Solution**: Implement theme detection in index.html script tag

### Issue: API Compatibility
**Problem**: New API structure breaks existing usage
**Solution**: Maintain backward compatibility facade until all components are migrated

## Rollback Procedures

For each PR, maintain rollback capability:

1. **Feature Flags**: Use environment variables to toggle new features
2. **Gradual Migration**: Keep old code paths until new ones are proven
3. **Database/State Compatibility**: Ensure state structures remain compatible
4. **Documentation**: Document rollback steps for each major change

This implementation guide provides the detailed technical steps needed to execute the migration plan successfully while maintaining backward compatibility and ensuring system stability.
