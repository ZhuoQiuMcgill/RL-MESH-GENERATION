# ADR 0005: API Base URL Sourcing

**Status**: Draft  
**Last Updated**: 2025-01-07  
**Owner**: Frontend Team  
**Reviewers**: DevOps Team, Architecture Team  

## Context

The current API configuration has significant limitations for deployment flexibility and environment management:

1. **Hardcoded URL**: API base URL is hardcoded in `ApiProvider.jsx` as `'http://localhost:8000'`
2. **No Environment Support**: Cannot configure different API endpoints for different environments
3. **Deployment Inflexibility**: Requires code changes to deploy to different environments
4. **Development Friction**: Researchers cannot easily point to custom backend instances
5. **Security Risk**: Production URLs might be exposed in development builds

Current implementation:
```javascript
// ApiProvider.jsx - Hardcoded configuration
const CONSTANTS = {
  API_BASE_URL: 'http://localhost:8000',
  CONNECTION_TIMEOUT: 10000,
  // ...
};
```

This creates maintenance overhead and prevents flexible deployment across different environments (development, staging, production, researcher-specific instances).

## Decision

We will implement an **environment-driven configuration system** with the following architecture:

### 1. Environment Variable Configuration

Utilize Vite's environment variable system for configuration:

```typescript
// Environment variable definitions
interface EnvironmentConfig {
  VITE_API_BASE_URL?: string;
  VITE_API_TIMEOUT?: string;
  VITE_POLLING_INTERVAL?: string;
  VITE_RETRY_COUNT?: string;
  VITE_RETRY_DELAY?: string;
  VITE_ENABLE_DEBUG?: string;
}

// .env.development
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=10000
VITE_POLLING_INTERVAL=2000
VITE_RETRY_COUNT=1
VITE_RETRY_DELAY=3000
VITE_ENABLE_DEBUG=true

// .env.production  
VITE_API_BASE_URL=https://api.mesh-rl.research.org
VITE_API_TIMEOUT=15000
VITE_POLLING_INTERVAL=5000
VITE_RETRY_COUNT=3
VITE_RETRY_DELAY=2000
VITE_ENABLE_DEBUG=false

// .env.staging
VITE_API_BASE_URL=https://api-staging.mesh-rl.research.org
VITE_API_TIMEOUT=12000
VITE_POLLING_INTERVAL=3000
VITE_RETRY_COUNT=2
VITE_RETRY_DELAY=2500
VITE_ENABLE_DEBUG=true
```

### 2. Configuration Management System

Create a centralized configuration system with validation and defaults:

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

// Default configuration with fallbacks
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

// Configuration validation
function validateUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

function validateConfig(config: Partial<ApiConfig>): string[] {
  const errors: string[] = [];
  
  if (config.baseUrl && !validateUrl(config.baseUrl)) {
    errors.push(`Invalid API base URL: ${config.baseUrl}`);
  }
  
  if (config.timeout && (config.timeout < 1000 || config.timeout > 60000)) {
    errors.push(`API timeout must be between 1000ms and 60000ms, got: ${config.timeout}`);
  }
  
  if (config.retryCount && (config.retryCount < 0 || config.retryCount > 5)) {
    errors.push(`Retry count must be between 0 and 5, got: ${config.retryCount}`);
  }
  
  return errors;
}

// Environment configuration loader
export function loadConfig(): AppConfig {
  const envConfig: Partial<ApiConfig> = {
    baseUrl: import.meta.env.VITE_API_BASE_URL,
    timeout: parseInt(import.meta.env.VITE_API_TIMEOUT) || undefined,
    pollingInterval: parseInt(import.meta.env.VITE_POLLING_INTERVAL) || undefined,
    retryCount: parseInt(import.meta.env.VITE_RETRY_COUNT) || undefined,
    retryDelay: parseInt(import.meta.env.VITE_RETRY_DELAY) || undefined,
    enableDebug: import.meta.env.VITE_ENABLE_DEBUG === 'true',
  };

  // Remove undefined values
  Object.keys(envConfig).forEach(key => {
    if (envConfig[key as keyof ApiConfig] === undefined) {
      delete envConfig[key as keyof ApiConfig];
    }
  });

  // Validate configuration
  const errors = validateConfig(envConfig);
  if (errors.length > 0) {
    console.error('Configuration validation errors:', errors);
    throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
  }

  // Merge with defaults
  const config: AppConfig = {
    api: { ...defaultConfig.api, ...envConfig },
    app: defaultConfig.app,
  };

  // Log configuration in development
  if (config.api.enableDebug) {
    console.log('Loaded configuration:', {
      ...config,
      api: {
        ...config.api,
        // Don't log potentially sensitive URLs in production
        baseUrl: config.app.environment === 'production' ? '[HIDDEN]' : config.api.baseUrl,
      },
    });
  }

  return config;
}

export const config = loadConfig();
```

### 3. Updated API Provider

Modify ApiProvider to use environment configuration:

```typescript
// src/context/ApiProvider.jsx
import { config } from '../config/environment';

// Use configuration instead of hardcoded constants
const CONSTANTS = {
  API_BASE_URL: config.api.baseUrl,
  CONNECTION_TIMEOUT: config.api.timeout,
  DEFAULT_RETRY_COUNT: config.api.retryCount,
  DEFAULT_RETRY_DELAY: config.api.retryDelay,
  DEFAULT_POLLING_INTERVAL: config.api.pollingInterval,
};

class ApiClient {
  constructor() {
    if (ApiClient.instance) {
      return ApiClient.instance;
    }
    
    this.baseUrl = CONSTANTS.API_BASE_URL;
    
    // Validate URL on initialization
    try {
      new URL(this.baseUrl);
    } catch (error) {
      throw new Error(`Invalid API base URL: ${this.baseUrl}`);
    }
    
    if (config.api.enableDebug) {
      console.log('API Client initialized with base URL:', this.baseUrl);
    }
    
    ApiClient.instance = this;
  }
  
  // ... rest of implementation
}
```

### 4. Runtime Configuration Support

Add support for runtime configuration overrides:

```typescript
// src/config/runtime.ts
interface RuntimeConfig {
  apiBaseUrl?: string;
}

class ConfigManager {
  private static instance: ConfigManager;
  private runtimeConfig: RuntimeConfig = {};

  static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  updateConfig(newConfig: Partial<RuntimeConfig>) {
    this.runtimeConfig = { ...this.runtimeConfig, ...newConfig };
    
    // Trigger API client recreation if URL changed
    if (newConfig.apiBaseUrl) {
      this.recreateApiClient();
    }
  }

  getCurrentConfig(): RuntimeConfig {
    return { ...this.runtimeConfig };
  }

  getEffectiveApiUrl(): string {
    return this.runtimeConfig.apiBaseUrl || config.api.baseUrl;
  }

  private recreateApiClient() {
    // Reset singleton to force recreation with new URL
    ApiClient.instance = null;
    
    // Notify components that API client changed
    window.dispatchEvent(new CustomEvent('api-config-changed'));
  }
}

export const configManager = ConfigManager.getInstance();
```

### 5. Development Tools Integration

Add configuration debugging tools for development:

```typescript
// src/components/dev/ConfigDebugger.tsx
const ConfigDebugger = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [customUrl, setCustomUrl] = useState('');

  if (config.app.environment === 'production') {
    return null; // Don't show in production
  }

  const handleUrlChange = () => {
    if (customUrl && validateUrl(customUrl)) {
      configManager.updateConfig({ apiBaseUrl: customUrl });
      alert(`API URL changed to: ${customUrl}`);
    } else {
      alert('Invalid URL');
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="bg-orange-500 text-white px-3 py-1 rounded text-xs"
      >
        Config
      </button>
      
      {isVisible && (
        <div className="absolute bottom-8 right-0 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-4 shadow-lg w-80">
          <h3 className="font-bold mb-2">Configuration Debug</h3>
          
          <div className="space-y-2 text-sm">
            <div>
              <strong>Environment:</strong> {config.app.environment}
            </div>
            <div>
              <strong>API URL:</strong> {configManager.getEffectiveApiUrl()}
            </div>
            <div>
              <strong>Timeout:</strong> {config.api.timeout}ms
            </div>
            
            <div className="pt-2 border-t">
              <label className="block text-xs mb-1">Override API URL:</label>
              <input
                type="text"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
                placeholder="http://localhost:8001"
                className="w-full px-2 py-1 text-xs border rounded"
              />
              <button
                onClick={handleUrlChange}
                className="mt-1 px-2 py-1 bg-blue-500 text-white text-xs rounded"
              >
                Update
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
```

## Alternatives Considered

### Alternative 1: Build-time Configuration Only
**Pros**: Simple, no runtime overhead, secure
**Cons**: Requires rebuild for configuration changes, less flexible for researchers
**Verdict**: Insufficient - need runtime flexibility for research environment

### Alternative 2: JSON Configuration File
**Pros**: Easy to edit, no environment variable limitations
**Cons**: Requires server to serve config file, security concerns with public config
**Verdict**: Rejected - environment variables are more secure and standard

### Alternative 3: URL Query Parameters
**Pros**: Easy for users to override, no build changes needed
**Cons**: Security issues with URLs in browser history, not suitable for production
**Verdict**: Rejected - too insecure for production use

### Alternative 4: Keep Current Hardcoded Approach
**Pros**: Simple, no changes needed
**Cons**: All current limitations remain, blocks proper deployment
**Verdict**: Rejected - current limitations are blocking proper deployment

## Implementation Plan

### Phase 1: Environment Variable Setup (Week 1)
1. Create environment configuration files (.env.development, .env.staging, .env.production)
2. Implement configuration loading and validation system
3. Create TypeScript interfaces for all configuration
4. Add configuration unit tests

### Phase 2: API Provider Integration (Week 1)
1. Update ApiProvider to use environment configuration
2. Add configuration validation on API client initialization
3. Test with different environment configurations
4. Add logging for configuration debugging

### Phase 3: Runtime Configuration (Week 2)
1. Implement runtime configuration management
2. Add development-mode configuration debugger
3. Create configuration override mechanisms
4. Test configuration hot-reloading

### Phase 4: Deployment Configuration (Week 2)
1. Set up environment-specific configuration files
2. Document deployment configuration procedures
3. Create configuration validation for CI/CD
4. Test all environment configurations

## Benefits

### Deployment Flexibility
- **Multi-environment Support**: Easy deployment to development, staging, production
- **Custom Research Setups**: Researchers can configure custom backend URLs
- **Hot Configuration**: Runtime configuration changes without rebuilds
- **Environment Isolation**: Clear separation between environment configurations

### Security Improvements
- **No Hardcoded URLs**: Production URLs not exposed in development code
- **Environment-specific Secrets**: Sensitive configuration isolated per environment
- **Validation**: Configuration validation prevents misconfigurations
- **Debugging Controls**: Debug features only enabled in development

### Developer Experience
- **Local Development**: Easy configuration for local API servers
- **Testing**: Simple configuration override for testing different scenarios
- **Documentation**: Clear configuration options and validation
- **Error Handling**: Clear error messages for configuration problems

### Operational Benefits
- **Automated Deployment**: CI/CD can set appropriate configuration per environment
- **Configuration Management**: Centralized configuration system
- **Monitoring**: Configuration logging for operational visibility
- **Rollback Support**: Easy configuration rollback without code changes

## Risks and Mitigations

### Risk 1: Configuration Complexity
**Risk**: Complex configuration system might be harder to understand
**Mitigation**:
- Clear documentation with examples
- Simple default configuration for common cases
- Development tools to visualize current configuration

### Risk 2: Environment Variable Exposure
**Risk**: Sensitive URLs might be exposed in build artifacts
**Mitigation**:
- Only use VITE_ prefixed variables (automatically exposed)
- Keep sensitive configuration on server side where possible
- Use different configuration approaches for highly sensitive values

### Risk 3: Runtime Configuration Confusion
**Risk**: Runtime overrides might create confusion about effective configuration
**Mitigation**:
- Clear logging of effective configuration in development
- Debug tools show current configuration state
- Runtime configuration only available in development mode

### Risk 4: Backward Compatibility
**Risk**: Configuration changes might break existing deployments
**Mitigation**:
- Maintain backward compatibility with sensible defaults
- Gradual migration with fallback to current values
- Clear migration documentation

## Success Metrics

### Configuration Flexibility
- ✅ Support for development, staging, production environments
- ✅ Runtime configuration override capability
- ✅ Custom researcher backend configuration support
- ✅ Zero hardcoded URLs in production builds

### Operational Excellence
- ✅ Configuration validation prevents invalid deployments
- ✅ Clear error messages for configuration issues
- ✅ Configuration logging for operational visibility
- ✅ Easy configuration rollback without code deployment

### Developer Experience
- ✅ Local development setup requires minimal configuration
- ✅ Clear documentation for all configuration options
- ✅ Development tools for configuration debugging
- ✅ TypeScript types for configuration validation

### Security & Reliability
- ✅ No sensitive configuration exposed in client code
- ✅ Environment-specific configuration isolation
- ✅ Configuration validation prevents runtime errors
- ✅ Graceful degradation for missing configuration

## Testing Strategy

### Configuration Testing
```typescript
describe('Configuration System', () => {
  test('loads default configuration when no env vars provided');
  test('overrides defaults with environment variables');
  test('validates URL format correctly');
  test('validates timeout ranges');
  test('validates retry count limits');
  test('throws errors for invalid configuration');
});

describe('Runtime Configuration', () => {
  test('allows runtime URL override in development');
  test('prevents runtime override in production');
  test('recreates API client when URL changes');
  test('maintains configuration state correctly');
});
```

### Integration Testing
```typescript
describe('API Configuration Integration', () => {
  test('API client uses configured base URL');
  test('API client uses configured timeout');
  test('API client handles invalid configuration gracefully');
  test('configuration changes trigger API client recreation');
});
```

### Environment Testing
```typescript
describe('Environment-specific Configuration', () => {
  test('development environment loads dev configuration');
  test('production environment loads prod configuration');
  test('staging environment loads staging configuration');
  test('unknown environment falls back to defaults');
});
```

## Configuration Documentation

### Environment Variables Reference
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000    # API server base URL
VITE_API_TIMEOUT=10000                     # Request timeout (milliseconds)
VITE_POLLING_INTERVAL=2000                 # Default polling interval (milliseconds)
VITE_RETRY_COUNT=1                         # Number of retry attempts
VITE_RETRY_DELAY=3000                      # Initial retry delay (milliseconds)
VITE_ENABLE_DEBUG=false                    # Enable debug logging

# Application Configuration  
VITE_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # Build timestamp
```

### Example Environment Files
```bash
# .env.local (researcher custom setup)
VITE_API_BASE_URL=http://192.168.1.100:8000
VITE_ENABLE_DEBUG=true

# .env.production  
VITE_API_BASE_URL=https://api.mesh-rl.research.org
VITE_API_TIMEOUT=15000
VITE_RETRY_COUNT=3
VITE_ENABLE_DEBUG=false
```

## Related Decisions

- [ADR 0001: Architecture Goals](./0001-record-architecture-goals.md) - Supports reliability and maintainability goals
- [Gap Analysis](../architecture/gap-analysis.md) - Addresses environment configuration gap

## References

- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)
- [12-Factor App Configuration](https://12factor.net/config)
- [Environment Configuration Best Practices](https://blog.logrocket.com/handling-environment-variables-node-js-typescript/)

---

**Next Steps**: Begin Phase 1 with environment file creation and configuration system implementation, followed by API Provider integration.
