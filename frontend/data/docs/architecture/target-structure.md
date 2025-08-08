# Target Architecture and Modular Structure

## Overview

This document outlines the proposed modular architecture for the RL Mesh Generation frontend application. The restructure aims to improve maintainability, scalability, and developer experience through better separation of concerns and modular organization.

## Target Directory Structure

```
src/
├── app/                          # Application shell and configuration
│   ├── App.jsx                   # Main application component
│   ├── routes.jsx               # Routing configuration
│   ├── providers.jsx            # Application-level providers
│   └── index.js                 # App module exports
│
├── modules/                      # Feature modules
│   ├── training/                # Training-related functionality
│   │   ├── pages/
│   │   │   ├── Train.jsx
│   │   │   └── TrainingMonitor.jsx
│   │   ├── components/
│   │   │   └── TrainingMonitor.jsx
│   │   ├── hooks/
│   │   │   └── useTraining.js
│   │   ├── services/
│   │   │   └── trainingApi.js
│   │   └── types/
│   │       └── training.types.js
│   │
│   ├── dashboard/               # Dashboard module
│   │   ├── pages/
│   │   │   └── Dashboard.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   ├── history/                 # History tracking module
│   │   ├── pages/
│   │   │   └── History.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   ├── quality/                 # Mesh quality analysis
│   │   ├── pages/
│   │   │   └── Quality.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   ├── geometry/                # Geometry management
│   │   ├── pages/
│   │   │   └── Geometry.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   ├── canvas/                  # Canvas visualization
│   │   ├── pages/
│   │   │   └── Canvas.jsx
│   │   ├── components/
│   │   │   ├── MeshCanvas.jsx
│   │   │   └── MeshCanvasTest.jsx
│   │   ├── hooks/
│   │   │   └── useMeshGenerator.js
│   │   ├── services/
│   │   │   └── canvasRenderer.js
│   │   └── types/
│   │       └── canvas.types.js
│   │
│   ├── angle/                   # Angle analysis
│   │   ├── pages/
│   │   │   └── Angle.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   ├── action/                  # Action management
│   │   ├── pages/
│   │   │   └── Action.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   │
│   └── generator/               # Mesh generation
│       ├── pages/
│       │   └── Generator.jsx
│       ├── components/
│       ├── hooks/
│       ├── services/
│       └── types/
│
├── shared/                      # Shared/reusable components and utilities
│   ├── ui/                      # UI component library
│   │   ├── Button.jsx           # Primary button component
│   │   ├── Input.jsx            # Input field component (FormInput)
│   │   ├── Card.jsx             # Card component (PanelCard)
│   │   ├── Table.jsx            # Data table component
│   │   ├── Modal.jsx            # Modal dialog component
│   │   ├── Badge.jsx            # Status badge component
│   │   ├── Tabs.jsx             # Tab navigation component
│   │   ├── Tooltip.jsx          # Tooltip component
│   │   ├── Spinner.jsx          # Loading spinner component
│   │   ├── Skeleton.jsx         # Skeleton loader component
│   │   ├── FormSelect.jsx       # Form select component
│   │   ├── EmptyState.jsx       # Empty state component
│   │   ├── CompactStatusBar.jsx # Status bar component
│   │   ├── LoadingOverlay.jsx   # Loading overlay component
│   │   └── index.js             # UI components index
│   │
│   ├── layout/                  # Layout components
│   │   ├── Header.jsx           # Application header (NavHeader)
│   │   ├── Sidebar.jsx          # Navigation sidebar
│   │   ├── Breadcrumb.jsx       # Breadcrumb navigation
│   │   ├── Page.jsx             # Page layout wrapper
│   │   ├── AppShell.jsx         # Main application shell
│   │   └── index.js             # Layout components index
│   │
│   └── icons/                   # Centralized icon library
│       ├── index.js             # Icon exports
│       └── IconComponents.jsx    # Icon component definitions
│
└── core/                        # Core application functionality
    ├── api/                     # API layer
    │   ├── ApiClient.js         # Main API client
    │   ├── hooks/               # API-related hooks
    │   │   ├── useApi.js
    │   │   └── index.js
    │   └── types/               # API type definitions
    │       ├── api.types.js
    │       └── index.js
    │
    ├── hooks/                   # Core application hooks
    │   ├── useTheme.js          # Theme management hook
    │   ├── useToast.js          # Toast notification hook
    │   ├── useBreakpoint.js     # Responsive breakpoint hook
    │   └── index.js             # Core hooks index
    │
    └── utils/                   # Core utilities
        ├── formatters.js        # Data formatting utilities
        ├── constants.js         # Application constants
        ├── css-variables.js     # CSS variable utilities
        ├── index.js             # Utils index
        └── types/               # Core type definitions
            └── common.types.js
```

## Key Architectural Principles

### 1. Module-Based Organization
- Each feature domain has its own module with consistent internal structure
- Modules are self-contained with their own pages, components, hooks, services, and types
- Clear boundaries between modules promote maintainability

### 2. Shared Resources
- Common UI components centralized in `src/shared/ui/`
- Layout components grouped in `src/shared/layout/`
- Centralized icon management in `src/shared/icons/`

### 3. Core Infrastructure
- API layer abstracted in `src/core/api/`
- Common hooks and utilities in `src/core/`
- Type definitions co-located with their respective modules

### 4. Application Shell
- Main app configuration in `src/app/`
- Routing and provider setup centralized
- Clean separation of app-level concerns

## Current to Target File Mapping

| Current File Path | Target File Path | Notes |
|------------------|------------------|--------|
| **Application Root** | | |
| `src/App.jsx` | `src/app/App.jsx` | Move to app module |
| `src/main.jsx` | `src/main.jsx` | Keep as application entry point |
| | `src/app/routes.jsx` | **New**: Extract routing from App.jsx |
| | `src/app/providers.jsx` | **New**: Extract providers from App.jsx |
| | `src/app/index.js` | **New**: App module exports |
| **Pages** | | |
| `src/pages/Action.jsx` | `src/modules/action/pages/Action.jsx` | Move to action module |
| `src/pages/Angle.jsx` | `src/modules/angle/pages/Angle.jsx` | Move to angle module |
| `src/pages/Canvas.jsx` | `src/modules/canvas/pages/Canvas.jsx` | Move to canvas module |
| `src/pages/Dashboard.jsx` | `src/modules/dashboard/pages/Dashboard.jsx` | Move to dashboard module |
| `src/pages/Generator.jsx` | `src/modules/generator/pages/Generator.jsx` | Move to generator module |
| `src/pages/Geometry.jsx` | `src/modules/geometry/pages/Geometry.jsx` | Move to geometry module |
| `src/pages/History.jsx` | `src/modules/history/pages/History.jsx` | Move to history module |
| `src/pages/Quality.jsx` | `src/modules/quality/pages/Quality.jsx` | Move to quality module |
| `src/pages/Train.jsx` | `src/modules/training/pages/Train.jsx` | Move to training module |
| `src/pages/TrainingMonitor.jsx` | `src/modules/training/pages/TrainingMonitor.jsx` | Move to training module |
| **Components** | | |
| `src/components/Breadcrumb.jsx` | `src/shared/layout/Breadcrumb.jsx` | Move to shared layout |
| `src/components/NavHeader.jsx` | `src/shared/layout/Header.jsx` | Rename and move to layout |
| `src/components/MeshCanvas.jsx` | `src/modules/canvas/components/MeshCanvas.jsx` | Move to canvas module |
| `src/components/MeshCanvasTest.jsx` | `src/modules/canvas/components/MeshCanvasTest.jsx` | Move to canvas module |
| `src/components/TrainingMonitor.jsx` | `src/modules/training/components/TrainingMonitor.jsx` | Move to training module |
| `src/components/index.js` | Multiple locations | Split exports by module |
| **UI Components** | | |
| `src/components/ui/Button.jsx` | `src/shared/ui/Button.jsx` | Move to shared UI |
| `src/components/ui/CompactStatusBar.jsx` | `src/shared/ui/CompactStatusBar.jsx` | Move to shared UI |
| `src/components/ui/EmptyState.jsx` | `src/shared/ui/EmptyState.jsx` | Move to shared UI |
| `src/components/ui/FormInput.jsx` | `src/shared/ui/Input.jsx` | Rename and move to shared UI |
| `src/components/ui/FormSelect.jsx` | `src/shared/ui/FormSelect.jsx` | Move to shared UI |
| `src/components/ui/LoadingOverlay.jsx` | `src/shared/ui/LoadingOverlay.jsx` | Move to shared UI |
| `src/components/ui/PanelCard.jsx` | `src/shared/ui/Card.jsx` | Rename and move to shared UI |
| `src/components/ui/examples.jsx` | Remove or move to stories | Consider removing or convert to Storybook |
| `src/components/ui/Train-refactored-example.jsx` | Remove or move to stories | Consider removing or convert to Storybook |
| `src/components/ui/index.js` | `src/shared/ui/index.js` | Move to shared UI |
| **Missing UI Components** | | |
| | `src/shared/ui/Table.jsx` | **New**: Create data table component |
| | `src/shared/ui/Modal.jsx` | **New**: Create modal dialog component |
| | `src/shared/ui/Badge.jsx` | **New**: Create status badge component |
| | `src/shared/ui/Tabs.jsx` | **New**: Create tab navigation component |
| | `src/shared/ui/Tooltip.jsx` | **New**: Create tooltip component |
| | `src/shared/ui/Spinner.jsx` | **New**: Create loading spinner component |
| | `src/shared/ui/Skeleton.jsx` | **New**: Create skeleton loader component |
| **Layout Components** | | |
| | `src/shared/layout/Sidebar.jsx` | **New**: Create navigation sidebar |
| | `src/shared/layout/Page.jsx` | **New**: Create page layout wrapper |
| | `src/shared/layout/AppShell.jsx` | **New**: Create main application shell |
| | `src/shared/layout/index.js` | **New**: Layout components index |
| **Context/Providers** | | |
| `src/context/ApiProvider.jsx` | `src/core/api/ApiProvider.jsx` | Move to core API |
| **Hooks** | | |
| `src/hooks/useMeshGenerator.js` | `src/modules/canvas/hooks/useMeshGenerator.js` | Move to canvas module |
| | `src/core/hooks/useTheme.js` | **New**: Theme management hook |
| | `src/core/hooks/useToast.js` | **New**: Toast notification hook |
| | `src/core/hooks/useBreakpoint.js` | **New**: Breakpoint management hook |
| | `src/core/hooks/index.js` | **New**: Core hooks index |
| **Services/API** | | |
| `src/lib/api-client.js` | `src/core/api/ApiClient.js` | Rename and move to core API |
| | `src/core/api/hooks/useApi.js` | **New**: Extract API hooks |
| | `src/core/api/hooks/index.js` | **New**: API hooks index |
| **Utils** | | |
| `src/utils/CanvasRenderer.js` | `src/modules/canvas/services/canvasRenderer.js` | Move to canvas module as service |
| `src/utils/constants.js` | `src/core/utils/constants.js` | Move to core utils |
| `src/lib/constants.js` | `src/core/utils/constants.js` | Merge with utils/constants.js |
| `src/lib/css-variables.js` | `src/core/utils/css-variables.js` | Move to core utils |
| `src/lib/utils.js` | `src/core/utils/formatters.js` | Rename and move to core utils |
| `src/lib/index.js` | Remove | Split exports to respective modules |
| | `src/core/utils/index.js` | **New**: Core utils index |
| **Icons** | | |
| | `src/shared/icons/index.js` | **New**: Centralized icon exports |
| | `src/shared/icons/IconComponents.jsx` | **New**: Icon component definitions |
| **Examples** | | |
| `src/components/examples/ApiHookExamples.jsx` | Remove or move to dev tools | Consider removing or convert to dev utilities |

## Migration Strategy

### Phase 1: Core Infrastructure
1. Create new directory structure
2. Move and refactor core utilities and API client
3. Set up shared UI component library
4. Create application shell structure

### Phase 2: Module Migration
1. Migrate one module at a time (start with training or canvas)
2. Update imports throughout the codebase
3. Test each module after migration
4. Update routing configuration

### Phase 3: Optimization
1. Remove unused files and dependencies
2. Optimize barrel exports (index.js files)
3. Update documentation and type definitions
4. Add missing UI components as needed

## Benefits of This Structure

1. **Improved Maintainability**: Clear separation of concerns and modular organization
2. **Better Scalability**: Easy to add new modules and features
3. **Enhanced Developer Experience**: Predictable file locations and consistent patterns
4. **Reusability**: Shared components and utilities reduce code duplication
5. **Testing**: Easier to test modules in isolation
6. **Code Splitting**: Natural boundaries for lazy loading and code splitting

## Implementation Notes

- All index.js files should use barrel exports for clean imports
- Each module should be self-contained with minimal cross-module dependencies
- Shared components should be framework-agnostic and reusable
- API layer should be abstracted and easily mockable for testing
- Type definitions should be co-located with their respective modules
