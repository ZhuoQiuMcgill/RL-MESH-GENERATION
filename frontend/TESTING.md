# Testing Guide

This guide covers the comprehensive testing strategy implemented for the RL Mesh Generation frontend application.

## Overview

Our testing setup includes:
- **Unit Tests**: Component and hook testing with Vitest + React Testing Library
- **API Mocking**: MSW (Mock Service Worker) for reliable API testing
- **E2E Tests**: Playwright for end-to-end smoke testing
- **Coverage**: Code coverage reporting with Vitest

## Testing Stack

### Core Testing Tools
- **Vitest**: Fast unit test runner with native ES modules support
- **React Testing Library**: Component testing utilities focused on user behavior
- **MSW**: API mocking for both unit and integration tests
- **Playwright**: E2E testing framework
- **jsdom**: Browser environment simulation

### Supporting Libraries
- **@testing-library/jest-dom**: Extended matchers for DOM testing
- **@testing-library/user-event**: User interaction simulation

## Project Structure

```
src/
├── test/
│   ├── setup.js                 # Global test setup
│   ├── mocks/
│   │   ├── handlers.js          # MSW request handlers
│   │   └── server.js            # MSW test server
│   └── utils/
│       └── test-utils.jsx       # Custom render utilities
├── hooks/__tests__/             # Hook unit tests
├── shared/ui/__tests__/         # UI component tests
├── context/__tests__/           # Context provider tests
└── pages/__tests__/             # Page component tests

e2e/                             # E2E tests
├── dashboard.spec.js            # Dashboard smoke tests
└── training.spec.js             # Training & API error tests
```

## Configuration Files

### Vitest Configuration (`vitest.config.js`)
- JSDOM environment for React components
- Global test utilities
- Coverage configuration with thresholds
- CSS processing support

### Playwright Configuration (`playwright.config.js`)
- Multi-browser testing (Chrome, Firefox, Safari)
- Mobile device testing
- Automatic dev server startup
- Screenshot and video capture on failures

## Running Tests

### Unit Tests
```bash
# Run all unit tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Open test UI
npm run test:ui
```

### E2E Tests
```bash
# Run all E2E tests
npm run e2e

# Run E2E tests with UI
npm run e2e:ui

# Run E2E tests in headed mode (visible browser)
npm run e2e:headed

# Debug E2E tests
npm run e2e:debug
```

### All Tests
```bash
# Run both unit and E2E tests
npm run test:all
```

## Test Categories

### 1. Unit Tests

#### Hook Tests (`src/hooks/__tests__/`)
- **useTrainingHooks.test.js**: Tests for training-related custom hooks
  - `useMeshBoundary`: Mesh boundary data loading
  - `useMeshData`: Mesh visualization data
  - `useReferencePoint`: Reference point finding
  - `useTrainingStatus`: Training status management with polling

**Example:**
```javascript
it('should load mesh boundary data successfully', async () => {
  const { result } = renderHook(() => useMeshBoundary('simple_square.obj'), {
    wrapper: HookWrapper
  })

  await waitFor(() => {
    result.current.loadBoundary()
  })

  expect(result.current.boundaryData).toEqual(expectedBoundaryData)
})
```

#### Component Tests (`src/shared/ui/__tests__/`)
- **Button.test.jsx**: Comprehensive button component testing
  - Variant styling (primary, secondary, danger, etc.)
  - Size variations (xs, sm, default, lg, xl)
  - Loading and disabled states
  - User interactions and accessibility

**Example:**
```javascript
it('handles loading state', () => {
  render(<Button loading>Loading</Button>)
  
  const button = screen.getByRole('button')
  expect(button).toBeDisabled()
  expect(button.querySelector('svg.animate-spin')).toBeInTheDocument()
})
```

- **Toast.test.jsx**: Toast notification component testing
  - Different toast types (info, success, warning, error)
  - Auto-dismiss functionality
  - Pause on hover behavior
  - Action buttons and close functionality

#### Context Tests (`src/context/__tests__/`)
- **ApiProvider.test.jsx**: API provider and hooks testing
  - Context provider functionality
  - API method availability
  - Error handling and retry logic
  - Polling hook behavior

#### Page Tests (`src/pages/__tests__/`)
- **Dashboard.test.jsx**: Dashboard page component testing
  - Content rendering
  - Navigation links
  - Statistics display
  - Responsive layout

### 2. API Mocking with MSW

#### Golden Response Fixtures
MSW handlers provide consistent, realistic API responses for testing:

```javascript
export const fixtures = {
  trainingStatus: {
    success: true,
    status: {
      is_training: false,
      status: 'idle',
      episode: 0,
      total_episodes: 0,
      current_reward: 0,
      best_reward: 0,
      elapsed_time: 0
    }
  },
  // ... more fixtures
}
```

#### Request Handlers
- Training endpoints (start/stop/status)
- Mesh endpoints (boundary/data/list)
- Error simulation endpoints
- Network failure simulation

### 3. E2E Smoke Tests

#### Dashboard Tests (`e2e/dashboard.spec.js`)
- Page loading and content display
- Navigation between modules
- Responsive design testing
- Accessibility features

#### Training & API Error Tests (`e2e/training.spec.js`)
- Training page interactions
- API error handling scenarios
- Network failure recovery
- Timeout and malformed response handling

**Example:**
```javascript
test('should handle network errors gracefully', async ({ page }) => {
  await page.route('http://127.0.0.1:5000/**', route => route.abort())
  await page.goto('/train')
  
  // Page should still load despite API failures
  await expect(page).toHaveURL('/train')
})
```

## Test Utilities

### Custom Render Functions
The `test-utils.jsx` file provides enhanced render functions:

```javascript
// Render with all providers (API, Toast, Router)
render(<Component />)

// Render with specific providers
renderWithApi(<Component />)
renderWithRouter(<Component />)
renderWithToast(<Component />)

// Render without any providers
renderWithoutProviders(<Component />)
```

### Hook Testing Wrapper
```javascript
const HookWrapper = createHookWrapper(['api', 'toast', 'router'])
```

## Coverage Configuration

Coverage thresholds are set at 70% for:
- Branches
- Functions
- Lines
- Statements

Excluded from coverage:
- Test files
- Example components
- Configuration files
- Main entry point

## Best Practices

### Unit Testing
1. **Test Behavior, Not Implementation**: Focus on what the component does, not how
2. **Use Descriptive Test Names**: Clear test descriptions help with debugging
3. **Arrange-Act-Assert**: Structure tests with clear setup, action, and assertion phases
4. **Mock External Dependencies**: Use MSW for API calls, mock complex dependencies

### E2E Testing
1. **Smoke Tests Only**: Focus on critical user paths and error scenarios
2. **Robust Selectors**: Use role-based selectors and text content over CSS selectors
3. **Handle Async Operations**: Properly wait for network requests and page loads
4. **Error Scenarios**: Test graceful degradation and error recovery

### API Testing
1. **Golden Fixtures**: Use consistent, realistic test data
2. **Error Scenarios**: Test various failure modes
3. **Network Conditions**: Simulate timeouts, network failures, malformed responses

## Continuous Integration

Tests are designed to run in CI environments:
- Deterministic test execution
- Proper cleanup and teardown
- Screenshot and video capture on failures
- Parallel test execution where possible

## Troubleshooting

### Common Issues
1. **Async Operations**: Use `waitFor` for async state changes
2. **Provider Errors**: Ensure components are wrapped with appropriate providers
3. **MSW Not Working**: Check that server is started in test setup
4. **E2E Timeouts**: Adjust timeouts for slower CI environments

### Debugging
- Use `test.only()` to focus on specific tests
- Add `console.log` statements for debugging
- Use Playwright's debug mode: `npm run e2e:debug`
- Check coverage reports for untested code paths

## Future Enhancements

Potential testing improvements:
- Visual regression testing with Playwright
- Performance testing for API operations
- Accessibility testing automation
- Integration testing with real backend
- Load testing for concurrent operations

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [MSW Documentation](https://mswjs.io/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
