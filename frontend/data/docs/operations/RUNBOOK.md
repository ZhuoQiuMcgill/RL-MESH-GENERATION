# Operational Runbook

**Status**: Approved  
**Last Updated**: 2024-01-20  
**Owner**: DevOps & Frontend Team  
**Reviewers**: Development Team  

## Overview

This runbook provides comprehensive operational guidance for the RL Mesh Generation frontend application, including environment configuration, deployment procedures, monitoring, and maintenance tasks.

## Environment Configuration

### Environment Variables

#### Core Configuration (.env)
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000     # Backend API endpoint
VITE_API_TIMEOUT=10000                      # Request timeout in ms
VITE_POLLING_INTERVAL=2000                  # Default polling interval in ms
VITE_RETRY_COUNT=1                          # Number of retry attempts
VITE_RETRY_DELAY=3000                       # Delay between retries in ms

# Development Configuration  
VITE_ENABLE_DEBUG=false                     # Enable debug logging
VITE_ENABLE_MOCK_API=false                  # Use mock API responses

# Build Configuration
VITE_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # Build timestamp
```

#### Environment-Specific Configurations

**Development (.env.development)**
```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_ENABLE_DEBUG=true
VITE_ENABLE_MOCK_API=true
VITE_POLLING_INTERVAL=1000
```

**Production (.env.production)**
```bash
VITE_API_BASE_URL=https://api.production-domain.com
VITE_ENABLE_DEBUG=false
VITE_ENABLE_MOCK_API=false
VITE_API_TIMEOUT=15000
VITE_RETRY_COUNT=3
```

**Testing (.env.test)**
```bash
VITE_API_BASE_URL=http://localhost:3001
VITE_ENABLE_DEBUG=false
VITE_ENABLE_MOCK_API=true
VITE_POLLING_INTERVAL=5000
```

### Environment Setup

#### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd rl-mesh-generation/frontend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env  # or your preferred editor
```

#### 2. Verification
```bash
# Verify environment configuration
npm run dev

# Check environment loading
npm run build:analyze

# Validate API connectivity
curl $VITE_API_BASE_URL/health
```

## Deployment Procedures

### Local Development Deployment

```bash
# Start development server
npm run dev

# Available at: http://localhost:5173
# Features: Hot reload, debugging, mock API
```

### Staging Deployment

```bash
# 1. Build for staging
npm run build

# 2. Preview build locally
npm run preview

# 3. Run tests before deployment
npm run test:all

# 4. Deploy to staging server
# (Deployment method depends on your infrastructure)
rsync -av dist/ user@staging-server:/var/www/html/
```

### Production Deployment

```bash
# 1. Pre-deployment checklist
npm run lint                    # Code quality check
npm run test:coverage          # Ensure test coverage
npm run build:analyze          # Bundle analysis
npm run e2e                    # End-to-end tests

# 2. Build production bundle
NODE_ENV=production npm run build

# 3. Security scan
npm audit --audit-level=moderate

# 4. Deploy to production
# (Use your CI/CD pipeline or deployment script)

# 5. Post-deployment verification
curl https://your-domain.com/health
```

## Application Management

### Adding a New Route/Module

#### 1. Create Module Structure
```bash
# Create module directory
mkdir -p src/modules/[module-name]/{pages,components,hooks,services}

# Create module index
touch src/modules/[module-name]/index.js
```

#### 2. Module Template
```javascript
// src/modules/[module-name]/index.js
export { default as [ModuleName]Page } from './pages/[ModuleName]';
export { default as use[ModuleName]Hooks } from './hooks/use[ModuleName]Hooks';
```

#### 3. Add Route Configuration
```javascript
// src/app/routes.js
import { [ModuleName]Page } from '../modules/[module-name]';

export const routes = [
  // ... existing routes
  {
    path: '/[route-path]',
    element: <[ModuleName]Page />,
    // ... route configuration
  }
];
```

#### 4. Update Navigation
```javascript
// Update navigation in src/components/NavHeader.jsx
const navigationItems = [
  // ... existing items
  {
    path: '/[route-path]',
    label: '[Display Name]',
    icon: '[Icon]'
  }
];
```

### Adding New Components

#### 1. UI Components
```bash
# Create component file
touch src/shared/ui/[ComponentName].jsx

# Add component tests
touch src/shared/ui/__tests__/[ComponentName].test.jsx

# Update index exports
echo "export { default as [ComponentName] } from './[ComponentName]';" >> src/shared/ui/index.js
```

#### 2. Component Template
```jsx
// src/shared/ui/[ComponentName].jsx
import React from 'react';

/**
 * [ComponentName] component description
 * @param {Object} props - Component properties
 * @param {string} props.className - Additional CSS classes
 */
const [ComponentName] = ({ className = '', ...props }) => {
  return (
    <div className={`component-base ${className}`} {...props}>
      {/* Component content */}
    </div>
  );
};

export default [ComponentName];
```

#### 3. Component Tests
```jsx
// src/shared/ui/__tests__/[ComponentName].test.jsx
import { render, screen } from '@testing-library/react';
import [ComponentName] from '../[ComponentName]';

describe('[ComponentName]', () => {
  test('renders correctly', () => {
    render(<[ComponentName] />);
    // Add specific tests
  });
});
```

## Testing Operations

### Running Tests

```bash
# Unit tests
npm run test                   # Watch mode
npm run test:run              # Single run
npm run test:coverage         # With coverage report

# E2E tests
npm run e2e                   # Headless mode
npm run e2e:headed           # With browser UI
npm run e2e:ui               # Interactive mode
npm run e2e:debug            # Debug mode

# All tests
npm run test:all             # Unit + E2E tests
```

### Test Configuration

#### Unit Test Coverage Thresholds
```javascript
// vitest.config.js
coverage: {
  thresholds: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  }
}
```

#### E2E Test Configuration
```javascript
// playwright.config.js
use: {
  baseURL: 'http://localhost:5173',
  trace: 'on-first-retry',
  screenshot: 'only-on-failure',
  video: 'retain-on-failure'
}
```

### Adding New Tests

#### 1. Component Tests
```bash
# Create test file
touch src/[path]/[Component].test.jsx

# Test template
cat > src/[path]/[Component].test.jsx << 'EOF'
import { render, screen, fireEvent } from '@testing-library/react';
import [Component] from './[Component]';

describe('[Component]', () => {
  test('renders correctly', () => {
    render(<[Component] />);
    expect(screen.getByRole('...')).toBeInTheDocument();
  });

  test('handles user interaction', async () => {
    render(<[Component] />);
    fireEvent.click(screen.getByRole('button'));
    // Add assertions
  });
});
EOF
```

#### 2. E2E Tests
```bash
# Create E2E test file
touch e2e/[feature].spec.js

# E2E test template
cat > e2e/[feature].spec.js << 'EOF'
import { test, expect } from '@playwright/test';

test.describe('[Feature]', () => {
  test('should [behavior]', async ({ page }) => {
    await page.goto('/[route]');
    await expect(page.locator('[selector]')).toBeVisible();
    // Add test steps
  });
});
EOF
```

## Monitoring and Debugging

### Application Health Checks

#### 1. Basic Health Check
```bash
# Check application availability
curl -f http://localhost:5173/ || echo "App not responding"

# Check API connectivity
curl -f $VITE_API_BASE_URL/health || echo "API not responding"
```

#### 2. Performance Monitoring
```bash
# Bundle size analysis
npm run build:analyze

# Performance audit (if lighthouse is installed)
lighthouse http://localhost:5173 --output json --output-path ./performance-report.json
```

### Debugging Procedures

#### 1. Application Won't Start
```bash
# Check Node.js version
node --version  # Should be >= 18

# Clear dependencies
rm -rf node_modules package-lock.json
npm install

# Check for port conflicts
lsof -i :5173

# Verbose logging
npm run dev --verbose
```

#### 2. Build Failures
```bash
# Clear build cache
rm -rf dist/
rm -rf .vite/

# Check for TypeScript errors
npm run build 2>&1 | grep -E "(error|Error)"

# Memory issues
NODE_OPTIONS="--max_old_space_size=4096" npm run build
```

#### 3. API Connection Issues
```bash
# Test API endpoint directly
curl -v $VITE_API_BASE_URL/health

# Check network connectivity
ping api-server-hostname

# Verify environment variables
echo $VITE_API_BASE_URL

# Check proxy configuration (if applicable)
cat vite.config.js | grep proxy
```

### Error Tracking

#### 1. Browser Console Errors
```javascript
// Enable verbose logging in development
if (import.meta.env.VITE_ENABLE_DEBUG === 'true') {
  console.log('[DEBUG] Application started');
}
```

#### 2. API Error Monitoring
```javascript
// API client includes automatic error logging
// Check browser network tab for failed requests
// Review console for API error messages
```

## Maintenance Tasks

### Daily Maintenance

```bash
# Check application status
curl -f http://your-domain.com/

# Review error logs
tail -f /var/log/nginx/error.log  # or your web server logs

# Monitor resource usage
df -h  # Disk usage
free -m  # Memory usage
```

### Weekly Maintenance

```bash
# Security updates
npm audit
npm audit fix

# Dependency updates (check for breaking changes)
npm outdated
npm update

# Test coverage review
npm run test:coverage
```

### Monthly Maintenance

```bash
# Full dependency audit
npm audit --audit-level=low

# Performance benchmark
npm run build:analyze

# Clean up old builds/logs
find . -name "node_modules" -type d -mtime +30 -exec rm -rf {} +
```

### Database Maintenance (if applicable)

```bash
# Clear old session data
# Clear cached API responses
# Archive old training data
```

## Backup and Recovery

### Code Backup
```bash
# Ensure code is committed to version control
git status
git add .
git commit -m "Pre-maintenance checkpoint"
git push origin main
```

### Configuration Backup
```bash
# Backup environment files
cp .env .env.backup.$(date +%Y%m%d)
cp .env.production .env.production.backup.$(date +%Y%m%d)
```

### Recovery Procedures
```bash
# Restore from version control
git reset --hard HEAD~1  # Revert last commit
git checkout main         # Switch to stable branch

# Restore configuration
cp .env.backup.YYYYMMDD .env

# Verify recovery
npm run build
npm run test
```

## Performance Optimization

### Bundle Optimization
```bash
# Analyze bundle size
npm run build:analyze

# Check for unused dependencies
npx depcheck

# Tree-shaking verification
npm run build -- --minify
```

### Runtime Performance
```bash
# Enable React DevTools profiler
# Monitor component re-renders
# Check memory usage in browser dev tools
# Profile WebGL performance for canvas components
```

## Security Measures

### Dependency Security
```bash
# Regular security audits
npm audit --audit-level=moderate

# Update vulnerable packages
npm audit fix

# Check for known vulnerabilities
npm audit --parseable | grep -E "(high|critical)"
```

### Content Security
```bash
# Validate CSP headers
curl -I https://your-domain.com/ | grep -i "content-security-policy"

# Check for XSS vulnerabilities
# Review user input handling
# Validate API response sanitization
```

## Troubleshooting Guide

### Common Issues

#### 1. "Module not found" errors
```bash
# Check import paths
# Verify file existence
# Check case sensitivity
# Clear module cache: rm -rf node_modules/.vite/
```

#### 2. "Network request failed" errors
```bash
# Verify VITE_API_BASE_URL
# Check CORS configuration
# Test API endpoint directly
# Review browser network tab
```

#### 3. Build/deployment failures
```bash
# Check Node.js version compatibility
# Verify environment variables
# Clear build cache
# Check disk space
```

### Support Contacts

- **Technical Issues**: Development Team
- **Infrastructure Issues**: DevOps Team  
- **Security Issues**: Security Team
- **Emergency Contacts**: [Emergency contact information]

---

*This runbook should be updated as operational procedures evolve and new issues are discovered.*
