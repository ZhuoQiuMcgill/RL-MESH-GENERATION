# Feature Modules

This directory contains the feature modules for the RL Mesh Generation frontend application. Each module follows a consistent architectural pattern and provides a specific domain of functionality.

## Module Structure

Each module follows the same internal structure for consistency and maintainability:

```
module-name/
├── pages/                  # Main page components
├── components/            # Presentational components
├── hooks/                 # Business logic (data fetching, polling)
├── services/             # Module-specific services
├── index.js              # Module exports (barrel file)
└── README.md             # Module documentation
```

## Available Modules

### Core Functionality Modules

- **`training/`** - RL model training and monitoring
  - Training session management
  - Real-time progress monitoring
  - Training analytics and metrics

- **`generator/`** - Mesh generation algorithms
  - AI/ML-powered mesh generation
  - Algorithm configuration and selection
  - Generation progress tracking

- **`canvas/`** - 3D visualization and rendering
  - Interactive 3D mesh display
  - Canvas controls and viewport management
  - High-performance WebGL rendering

### Analysis and Quality Modules

- **`quality/`** - Mesh quality analysis
  - Quality metrics calculation
  - Quality visualization and comparison
  - Quality improvement recommendations

- **`angle/`** - Angle analysis and measurement
  - Mesh angle analysis
  - Angle quality assessment
  - Angular distribution visualization

- **`geometry/`** - Geometric operations
  - Shape management and transformation
  - Geometric calculations and validation
  - Geometric data processing

### Management and Interface Modules

- **`dashboard/`** - System overview and metrics
  - System status monitoring
  - Performance metrics visualization
  - Activity feed and notifications

- **`history/`** - Activity and session history
  - Session tracking and replay
  - History filtering and search
  - Data export functionality

- **`action/`** - Action management and workflow
  - User action tracking
  - Undo/redo functionality
  - Batch operation management

## Module Communication

Modules communicate through:

1. **Shared Services**: Common API clients and utilities
2. **Core Infrastructure**: Shared hooks and state management
3. **Event System**: Cross-module event publishing/subscribing
4. **Routing**: Page navigation and state sharing via URL

## Import Patterns

### Individual Module Import
```javascript
import { TrainPage, useTraining } from 'modules/training';
```

### Multiple Module Import
```javascript
import { DashboardPage } from 'modules/dashboard';
import { CanvasPage, MeshCanvas } from 'modules/canvas';
```

### All Modules Import
```javascript
import { TrainPage, DashboardPage, CanvasPage } from 'modules';
```

## Development Guidelines

### Creating New Modules
1. Follow the established directory structure
2. Include a comprehensive README.md
3. Implement consistent naming conventions
4. Provide barrel exports via index.js
5. Document public interfaces and dependencies

### Module Dependencies
- Prefer dependency injection over direct imports
- Use shared services for cross-module functionality
- Avoid circular dependencies between modules
- Keep modules as self-contained as possible

### Testing Strategy
- Each module should be testable in isolation
- Mock external dependencies and services
- Test public interfaces and key functionality
- Include integration tests for module interactions

## Architecture Benefits

1. **Modularity**: Clear separation of concerns and responsibilities
2. **Scalability**: Easy to add new modules and features
3. **Maintainability**: Isolated changes and reduced coupling
4. **Reusability**: Shared components and consistent patterns
5. **Testing**: Easier unit and integration testing
6. **Performance**: Natural boundaries for code splitting and lazy loading

## Migration Notes

This modular structure supports the migration from the existing page-based architecture to a more scalable and maintainable system. Each module encapsulates related functionality while maintaining clear interfaces for inter-module communication.
