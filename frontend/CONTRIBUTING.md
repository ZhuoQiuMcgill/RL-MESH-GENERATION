# Contributing Guide

Thank you for your interest in contributing to the RL Mesh Generation frontend! This guide will help you get started with the development process, coding standards, and contribution workflow.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- **Node.js**: >= 18.0.0
- **npm**: >= 8.0.0
- **Git**: Latest version
- **Modern Browser**: Chrome, Firefox, Safari, or Edge

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/YOUR_USERNAME/rl-mesh-generation.git
   cd rl-mesh-generation/frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit environment variables for your setup
   nano .env
   ```

4. **Verify Setup**
   ```bash
   # Start development server
   npm run dev
   
   # Run tests
   npm run test
   
   # Check build
   npm run build
   ```

5. **IDE Setup** (Recommended)
   - **VSCode Extensions**:
     - ES7+ React/Redux/React-Native snippets
     - Prettier - Code formatter
     - ESLint
     - Tailwind CSS IntelliSense
     - Auto Rename Tag
     - Bracket Pair Colorizer

## Development Workflow

### Branch Strategy

We follow a **Git Flow** approach:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/[feature-name]` - Individual features
- `bugfix/[bug-name]` - Bug fixes
- `hotfix/[fix-name]` - Critical production fixes

### Creating a Feature

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Development Process**
   ```bash
   # Make changes
   # Test changes
   npm run test
   
   # Lint code
   npm run lint
   
   # Commit changes (follow conventional commits)
   git add .
   git commit -m "feat: add new dashboard component"
   ```

3. **Keep Branch Updated**
   ```bash
   # Regularly sync with develop
   git checkout develop
   git pull origin develop
   git checkout feature/your-feature-name
   git rebase develop
   ```

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tools

**Examples:**
```bash
feat(dashboard): add training status cards
fix(api): resolve polling memory leak
docs(readme): update installation instructions
test(components): add Button component tests
```

## Coding Standards

### JavaScript/JSX Standards

#### 1. General Principles
- **Functional Components**: Use functional components with hooks
- **ESLint**: Follow configured ESLint rules
- **Prettier**: Use Prettier for code formatting
- **ES6+**: Use modern JavaScript features

#### 2. Naming Conventions

```javascript
// Components: PascalCase
const UserProfile = () => {};
const TrainingMonitor = () => {};

// Files: PascalCase for components, camelCase for utilities
UserProfile.jsx
trainingUtils.js

// Variables/Functions: camelCase
const userName = 'john';
const handleSubmit = () => {};

// Constants: SCREAMING_SNAKE_CASE
const API_BASE_URL = 'http://localhost:8000';
const MAX_RETRY_ATTEMPTS = 3;

// CSS Classes: kebab-case (following Tailwind)
className="training-status-card"
className="btn-primary"
```

#### 3. Component Structure

```jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types'; // If using PropTypes

// Import order: React -> External -> Internal -> Relative
import { Button, Card } from '../shared/ui';
import { useApi } from '../core/hooks';
import './ComponentName.css'; // If component-specific styles needed

/**
 * Component description
 * @param {Object} props - Component properties
 * @param {string} props.title - The title to display
 * @param {Function} props.onSubmit - Callback for form submission
 * @param {boolean} props.loading - Loading state
 * @returns {JSX.Element} Rendered component
 */
const ComponentName = ({ 
  title, 
  onSubmit, 
  loading = false,
  className = '',
  ...restProps 
}) => {
  // Hooks at the top
  const [localState, setLocalState] = useState(null);
  const { data, error } = useApi();

  // Effects
  useEffect(() => {
    // Side effects
  }, []);

  // Event handlers
  const handleClick = useCallback((event) => {
    event.preventDefault();
    onSubmit?.(event);
  }, [onSubmit]);

  // Early returns for loading/error states
  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message={error.message} />;
  }

  // Main render
  return (
    <div className={`component-base ${className}`} {...restProps}>
      <h2 className="text-xl font-semibold">{title}</h2>
      <Button onClick={handleClick} loading={loading}>
        Submit
      </Button>
    </div>
  );
};

// PropTypes (optional but recommended for complex components)
ComponentName.propTypes = {
  title: PropTypes.string.isRequired,
  onSubmit: PropTypes.func,
  loading: PropTypes.bool,
  className: PropTypes.string,
};

export default ComponentName;
```

#### 4. Hook Guidelines

```javascript
// Custom hook naming: use[HookName]
const useTrainingData = (trainingId) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Hook logic...

  return { data, loading, error, refetch };
};

// Hook usage in components
const TrainingComponent = () => {
  const { data, loading, error, refetch } = useTrainingData(trainingId);
  
  // Component logic...
};
```

### CSS/Styling Standards

#### 1. Tailwind CSS Usage
```jsx
// Prefer utility classes over custom CSS
<div className="bg-card text-text-primary p-xl rounded-lg shadow-md">
  <h2 className="text-2xl font-bold mb-lg">Title</h2>
  <p className="text-text-secondary">Content</p>
</div>

// Use design tokens for consistency
<Button className="bg-primary-start hover:bg-primary-end">
  Click me
</Button>

// Responsive design
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-lg">
  {/* Responsive grid */}
</div>
```

#### 2. Component-Specific Styles (when needed)
```css
/* ComponentName.css */
.component-name {
  /* Use CSS custom properties from design tokens */
  background: var(--color-bg-card);
  border-radius: var(--radius-lg);
  padding: var(--space-xl);
}

/* Use BEM methodology for complex components */
.training-monitor__status-card {
  /* Specific styling */
}

.training-monitor__status-card--active {
  /* Modifier styling */
}
```

### API Integration Standards

```javascript
// Use the centralized API client
import { useApi } from '../core/api/hooks';

const TrainingComponent = () => {
  const api = useApi();
  
  const handleStartTraining = async (config) => {
    try {
      const response = await api.startTraining(config);
      // Handle success
    } catch (error) {
      // Error handling is built into the API client
      console.error('Training failed:', error.message);
    }
  };

  // For real-time data, use polling hook
  const { data, error, isLoading } = usePolling('getTrainingStatus', 2000, {
    enabled: trainingActive,
  });

  return (
    // Component JSX
  );
};
```

## Testing Guidelines

### 1. Test Structure

```javascript
// Component.test.jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ApiProvider } from '../context/ApiProvider';
import Component from './Component';

// Test wrapper for context providers
const TestWrapper = ({ children }) => (
  <ApiProvider>
    {children}
  </ApiProvider>
);

describe('Component', () => {
  // Test organization
  describe('Rendering', () => {
    test('renders correctly with default props', () => {
      render(<Component />, { wrapper: TestWrapper });
      expect(screen.getByText('Expected Text')).toBeInTheDocument();
    });

    test('renders correctly with custom props', () => {
      render(<Component title="Custom Title" />, { wrapper: TestWrapper });
      expect(screen.getByText('Custom Title')).toBeInTheDocument();
    });
  });

  describe('User Interactions', () => {
    test('handles button click', async () => {
      const handleClick = jest.fn();
      render(<Component onClick={handleClick} />, { wrapper: TestWrapper });
      
      await userEvent.click(screen.getByRole('button'));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });
  });

  describe('API Integration', () => {
    test('displays loading state', async () => {
      render(<Component />, { wrapper: TestWrapper });
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    test('displays error state', async () => {
      // Mock API error
      render(<Component />, { wrapper: TestWrapper });
      
      await waitFor(() => {
        expect(screen.getByText(/error/i)).toBeInTheDocument();
      });
    });
  });
});
```

### 2. Test Coverage Requirements

- **Minimum Coverage**: 70% for all metrics (lines, functions, branches, statements)
- **Component Tests**: Test rendering, user interactions, and edge cases
- **Hook Tests**: Test custom hooks in isolation
- **Integration Tests**: Test component interactions with API and context
- **E2E Tests**: Test critical user workflows

### 3. E2E Testing

```javascript
// e2e/training-workflow.spec.js
import { test, expect } from '@playwright/test';

test.describe('Training Workflow', () => {
  test('user can start and monitor training', async ({ page }) => {
    await page.goto('/train');
    
    // Fill training configuration
    await page.fill('[data-testid="learning-rate"]', '0.001');
    await page.fill('[data-testid="episodes"]', '1000');
    
    // Start training
    await page.click('[data-testid="start-training"]');
    
    // Verify training started
    await expect(page.locator('[data-testid="training-status"]')).toContainText('Running');
    
    // Navigate to monitor
    await page.click('[data-testid="monitor-link"]');
    
    // Verify monitoring interface
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
  });
});
```

## Documentation Standards

### 1. JSDoc Comments

```javascript
/**
 * Calculates the mesh quality score based on various metrics
 * @param {Object} meshData - The mesh data object
 * @param {number[]} meshData.vertices - Array of vertex coordinates
 * @param {number[]} meshData.faces - Array of face indices
 * @param {Object} options - Configuration options
 * @param {boolean} options.includeAngles - Include angle analysis
 * @param {number} options.threshold - Quality threshold (0-1)
 * @returns {Promise<number>} Quality score between 0 and 1
 * @throws {Error} When mesh data is invalid
 * 
 * @example
 * const quality = await calculateMeshQuality(meshData, {
 *   includeAngles: true,
 *   threshold: 0.7
 * });
 */
const calculateMeshQuality = async (meshData, options = {}) => {
  // Implementation
};
```

### 2. README Updates

When adding new features, update relevant README files:
- Main `README.md` for major features
- Module-specific READMEs in `src/modules/[module]/README.md`
- Component documentation in `src/shared/ui/README.md`

### 3. Architecture Documentation

Update architecture docs for significant changes:
- `data/docs/architecture/FINAL_ARCHITECTURE.md`
- `data/docs/architecture/COMPONENT_MAP.md`
- Relevant ADRs in `data/docs/adr/`

## Pull Request Process

### 1. Pre-PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass (`npm run test:all`)
- [ ] Code is properly linted (`npm run lint`)
- [ ] Documentation is updated
- [ ] Breaking changes are documented
- [ ] Commit messages follow conventional format

### 2. PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### 3. PR Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one approval required
3. **Testing**: Manual testing for significant changes
4. **Documentation**: Verify docs are updated
5. **Merge**: Use "Squash and merge" for clean history

## Code Review Guidelines

### For Authors

- Keep PRs focused and reasonably sized
- Provide clear descriptions and context
- Respond promptly to feedback
- Address all comments before requesting re-review

### For Reviewers

- Be constructive and respectful
- Focus on code quality, performance, and maintainability
- Check for test coverage and documentation
- Verify that changes align with architectural decisions

### Review Checklist

- [ ] **Functionality**: Does the code work as intended?
- [ ] **Code Quality**: Is the code clean and well-structured?
- [ ] **Performance**: Are there any performance implications?
- [ ] **Security**: Are there any security concerns?
- [ ] **Testing**: Is there adequate test coverage?
- [ ] **Documentation**: Is documentation updated?
- [ ] **Architecture**: Does it follow established patterns?

## Release Process

### Version Numbering

We follow [Semantic Versioning (SemVer)](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Steps

1. **Prepare Release**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.2.0
   ```

2. **Update Version**
   ```bash
   npm version minor  # or patch/major
   ```

3. **Update Changelog**
   - Document new features
   - List bug fixes
   - Note breaking changes

4. **Final Testing**
   ```bash
   npm run test:all
   npm run build
   npm run e2e
   ```

5. **Merge to Main**
   ```bash
   git checkout main
   git merge release/v1.2.0
   git tag v1.2.0
   git push origin main --tags
   ```

6. **Deploy**
   - Deploy to staging
   - Test staging environment
   - Deploy to production

## Getting Help

- **Documentation**: Check `data/docs/` for detailed documentation
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Team Chat**: [Internal team communication channels]

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment
- Follow the project's code of conduct

---

Thank you for contributing to the RL Mesh Generation project! Your contributions help make this project better for everyone.
