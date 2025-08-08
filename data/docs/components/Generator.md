# Generator Page

## Overview
An interactive mesh generation tool that allows users to configure and run a complete mesh generation pipeline, from selecting an initial mesh to configuring predictors, reference selectors, and quality methods. Provides step-by-step execution and real-time visualization.

## File Location
`frontend/src/pages/Generator.jsx`

## Props
This page component does not accept any props.

## State Usage
| State Variable | Type | Default | Purpose |
|---------------|------|---------|---------|
| `components` | Object\|null | `null` | Available generation components (meshes, predictors) |
| `selectedMesh` | string | `''` | Selected initial mesh |
| `meshInfo` | Object\|null | `null` | Information about the selected mesh |
| `selectedPredictor` | string | `''` | Selected predictor type |
| `selectedRefSelector` | string | `''` | Selected reference selector type |
| `selectedQualityMethod` | string | `''` | Selected quality evaluation method |
| `predictorConfig` | Object | `{...}` | Configuration for the selected predictor |
| `refSelectorConfig` | Object | `{...}` | Configuration for the reference selector |
| `sessionId` | string\|null | `null` | Current generation session ID |
| `currentStep` | number | `0` | Current step in the generation process |
| `sessionData` | Object\|null | `null` | Data for the current generation session |
| `isLoading` | boolean | `false` | Loading state for async operations |
| `error` | string\|null | `null` | Error messages |
| `log` | Array | `[]` | Operation log entries |
| `actionInfo` | Object\|null | `null` | Information about the last action taken |
| `referencePointInfo` | Object\|null | `null` | Information about the current reference point |
| `elementQuality` | Object\|null | `null` | Quality metrics for the generated elements |

## Dependencies

### React Dependencies
- `useState`, `useEffect`, `useRef`, `useCallback` - Standard React hooks

### Internal Dependencies
- `NavHeader`, `MeshCanvas` from `'../components'`
- UI Components: `Button`, `FormInput`, `FormSelect`, `LoadingOverlay`, `EmptyState`
- `useApi` from `'../context/ApiProvider'` - API client access

### External Dependencies
- None

## Side Effects

### API Integration (Planned)
- **Load Components**: `api.getComponents()` (mocked) - Get available predictors, meshes, etc.
- **Mesh Info**: `api.getMeshInfo()` - Get details of selected mesh
- **Create Session**: `api.createPredictionSession()` - Start a new generation session
- **Execute Steps**: `api.executeNextStep()`, `api.executePreviousStep()`, `api.processAllSteps()` - Control generation flow
- **Session Management**: `api.resetSession()`, `api.deleteSession()` - Manage generation sessions
- **Reference Point Reselection**: `api.reselectReferencePoint()` - Change reference point during generation

### Canvas Integration
- **Initial Mesh Preview**: Renders the boundary of the selected initial mesh
- **Step-by-Step Visualization**: Updates canvas with each generation step
- **Empty State**: Displays an informative message when no mesh is selected

## Features

### Session Configuration
- **Initial Mesh Selection**: Choose a starting mesh from a predefined list
- **Predictor Configuration**: Select a predictor and configure its parameters (n, g, beta, model path)
- **Reference Selector**: Choose a reference selection method and configure its parameters (n)
- **Quality Method**: Select a quality evaluation method
- **Trained Models**: Select from a list of available trained models for the RL predictor

### Session Control
- **Create Session**: Starts a new generation session with the specified configuration
- **Step-by-Step Execution**: Execute the generation process one step at a time (Next/Previous)
- **Process All**: Run the entire generation process automatically
- **Reset Session**: Reset the session to its initial state
- **Delete Session**: Terminate and delete the current session
- **Reselect Reference Point**: Manually trigger reselection of the reference point

### Real-time Data Display
- **Step Details**: Displays information about the last action taken (type, status, new coordinates)
- **Reference Point Info**: Shows details of the current reference point (index, coordinates, angle)
- **Element Quality**: Displays quality metrics for the newly generated elements
- **Session Status**: Shows the current session ID, step number, and other session data

### Logging System
- **Timestamped Log**: Records all configuration changes, session actions, and API calls with timestamps
- **Categorized Messages**: Uses different colors for info, success, warning, and error messages
- **Clearable Log**: Allows the user to clear the operation log

## Complex Layout

### Three-Panel Design
1. **Left Panel (Configuration)**: Contains all the controls for configuring the generation session
2. **Center Panel (Visualization)**: Features the MeshCanvas for visualizing the generation process, along with session control buttons
3. **Right Panel (Data Display)**: Shows detailed information about the current step, reference point, quality, and session status

## Mock Data and API Simulation
- **Component Loading**: The `loadComponents` function simulates an API call by providing a mock object with initial meshes, predictors, selectors, quality methods, and trained models.
- **API Calls**: Most of the API calls in the component are commented out, as they depend on a backend implementation that is not yet available.

## Known Issues
1. **API Not Implemented**: The entire functionality of the page relies on a backend API that is currently simulated with mock data.
2. **State Management**: The component manages a large number of state variables, which could be simplified with a state management library or reducers.
3. **Error Handling**: Error handling is basic and mostly logs errors to the console.
4. **No Input Validation**: There is no validation for the configuration inputs.

## Potential Improvements
1. **Implement Backend API**: Connect the component to a real backend API to enable full functionality.
2. **State Management**: Refactor the state management using `useReducer` or a library like Redux or Zustand.
3. **Input Validation**: Add validation to the configuration forms to prevent errors.
4. **User Experience**: Improve the user experience with more informative messages, better loading states, and a more intuitive layout.
5. **Save/Load Configurations**: Allow users to save and load their generation configurations.
6. **Export Results**: Add functionality to export the generated mesh and session data.

## Related Components
- **Requires**: `NavHeader`, `MeshCanvas`, and various UI components (`Button`, `FormInput`, `FormSelect`, `LoadingOverlay`, `EmptyState`)
- **Uses**: `ApiProvider` context for API access
- **Similar to**: The `Action` page, but with a focus on a full generation pipeline rather than single action testing.
