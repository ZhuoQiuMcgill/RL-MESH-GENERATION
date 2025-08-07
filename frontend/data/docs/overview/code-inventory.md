# Code Inventory Report

Generated on: 2025/8/7 17:36:46

## Summary

| Metric | Count |
|--------|-------|
| Total Files | 43 |
| Code Files | 38 |
| Asset Files | 5 |
| Total Imports | 43 |
| Orphaned Files | 16 |

## Components

Components organized by folder

### components

#### `Breadcrumb.jsx`
**Imported by:**
- `App.jsx`

#### `index.js`
**Imported by:**
- `pages/Action.jsx`
- `pages/Generator.jsx`
- `pages/History.jsx`

#### `MeshCanvas.jsx`
**Imports:**
- `utils/CanvasRenderer.js`
**Imported by:**
- `components/MeshCanvasTest.jsx`
- `components/TrainingMonitor.jsx`
- `pages/TrainingMonitor.jsx`

#### `MeshCanvasTest.jsx`
**Imports:**
- `components/MeshCanvas.jsx`

#### `NavHeader.jsx`
**Imported by:**
- `App.jsx`

#### `TrainingMonitor.jsx`
**Imports:**
- `components/MeshCanvas.jsx`
- `context/ApiProvider.jsx`
**Imported by:**
- `pages/Train.jsx`

### components/examples

#### `ApiHookExamples.jsx`
**Imports:**
- `context/ApiProvider.jsx`

### components/ui

#### `Button.jsx`
**Imports:**
- `lib/utils.js`

#### `CompactStatusBar.jsx`
**Imports:**
- `lib/utils.js`

#### `EmptyState.jsx`
**Imports:**
- `lib/utils.js`

#### `examples.jsx`
*No dependencies tracked*

#### `FormInput.jsx`
**Imports:**
- `lib/utils.js`

#### `FormSelect.jsx`
**Imports:**
- `lib/utils.js`

#### `index.js`
**Imported by:**
- `pages/Action.jsx`
- `pages/Generator.jsx`
- `pages/History.jsx`
- `pages/TrainingMonitor.jsx`

#### `LoadingOverlay.jsx`
**Imports:**
- `lib/utils.js`

#### `PanelCard.jsx`
**Imports:**
- `lib/utils.js`

#### `Train-refactored-example.jsx`
*No dependencies tracked*

## Pages

Page components and their dependencies

### pages

#### `Action.jsx`
**Imports:**
- `components/index.js`
- `components/ui/index.js`
- `context/ApiProvider.jsx`
**Imported by:**
- `App.jsx`

#### `Angle.jsx`
**Imported by:**
- `App.jsx`

#### `Canvas.jsx`
**Imported by:**
- `App.jsx`

#### `Dashboard.jsx`
**Imported by:**
- `App.jsx`

#### `Generator.jsx`
**Imports:**
- `components/index.js`
- `components/ui/index.js`
- `context/ApiProvider.jsx`
**Imported by:**
- `App.jsx`

#### `Geometry.jsx`
**Imported by:**
- `App.jsx`

#### `History.jsx`
**Imports:**
- `components/index.js`
- `components/ui/index.js`
- `context/ApiProvider.jsx`
**Imported by:**
- `App.jsx`

#### `Quality.jsx`
**Imported by:**
- `App.jsx`

#### `Train.jsx`
**Imports:**
- `components/TrainingMonitor.jsx`
**Imported by:**
- `App.jsx`

#### `TrainingMonitor.jsx`
**Imports:**
- `components/ui/index.js`
- `components/MeshCanvas.jsx`
- `context/ApiProvider.jsx`

## Custom Hooks

Custom React hooks

### hooks

#### `useMeshGenerator.js`
**Imports:**
- `context/ApiProvider.jsx`

## Context Providers

React context providers and related files

### context

#### `ApiProvider.jsx`
**Imported by:**
- `App.jsx`
- `components/examples/ApiHookExamples.jsx`
- `components/TrainingMonitor.jsx`
- `hooks/useMeshGenerator.js`
- `pages/Action.jsx`
- `pages/Generator.jsx`
- `pages/History.jsx`
- `pages/TrainingMonitor.jsx`

## Utilities

Utility functions and helpers

### utils

#### `CanvasRenderer.js`
**Imports:**
- `utils/constants.js`
**Imported by:**
- `components/MeshCanvas.jsx`

#### `constants.js`
**Imported by:**
- `utils/CanvasRenderer.js`

## Assets

Static assets (CSS, images, etc.)

### .

#### `App.css`
**Imported by:**
- `App.jsx`

#### `index.css`
**Imported by:**
- `main.jsx`

### assets

#### `react.svg`
*No dependencies tracked*

### lib/assets

#### `react.svg`
*No dependencies tracked*

#### `vite.svg`
*No dependencies tracked*

## Other Files

Other source files

### .

#### `App.jsx`
**Imports:**
- `components/NavHeader.jsx`
- `components/Breadcrumb.jsx`
- `pages/Dashboard.jsx`
- `pages/Train.jsx`
- `pages/History.jsx`
- `pages/Quality.jsx`
- `pages/Geometry.jsx`
- `pages/Canvas.jsx`
- `pages/Angle.jsx`
- `pages/Action.jsx`
- `pages/Generator.jsx`
- `context/ApiProvider.jsx`
- `App.css`
**Imported by:**
- `main.jsx`

#### `main.jsx`
**Imports:**
- `index.css`
- `App.jsx`

### lib

#### `api-client.js`
**Imports:**
- `lib/utils.js`

#### `constants.js`
*No dependencies tracked*

#### `css-variables.js`
*No dependencies tracked*

#### `index.js`
*No dependencies tracked*

#### `utils.js`
**Imported by:**
- `components/ui/Button.jsx`
- `components/ui/CompactStatusBar.jsx`
- `components/ui/EmptyState.jsx`
- `components/ui/FormInput.jsx`
- `components/ui/FormSelect.jsx`
- `components/ui/LoadingOverlay.jsx`
- `components/ui/PanelCard.jsx`
- `lib/api-client.js`

## Orphaned Files

Files that are not imported by any other file:

- `components/examples/ApiHookExamples.jsx`
- `components/MeshCanvasTest.jsx`
- `components/ui/Button.jsx`
- `components/ui/CompactStatusBar.jsx`
- `components/ui/EmptyState.jsx`
- `components/ui/examples.jsx`
- `components/ui/FormInput.jsx`
- `components/ui/FormSelect.jsx`
- `components/ui/LoadingOverlay.jsx`
- `components/ui/PanelCard.jsx`
- `components/ui/Train-refactored-example.jsx`
- `hooks/useMeshGenerator.js`
- `lib/api-client.js`
- `lib/constants.js`
- `lib/css-variables.js`
- `pages/TrainingMonitor.jsx`

## Import Graph Analysis

### Most Imported Files

| File | Imported By | Count |
|------|-------------|-------|
| `context/ApiProvider.jsx` | 8 files | 8 |
| `lib/utils.js` | 8 files | 8 |
| `components/ui/index.js` | 4 files | 4 |
| `components/MeshCanvas.jsx` | 3 files | 3 |
| `components/index.js` | 3 files | 3 |
| `components/NavHeader.jsx` | 1 files | 1 |
| `components/Breadcrumb.jsx` | 1 files | 1 |
| `pages/Dashboard.jsx` | 1 files | 1 |
| `pages/Train.jsx` | 1 files | 1 |
| `pages/History.jsx` | 1 files | 1 |

### Files with Most Dependencies

| File | Dependencies | Count |
|------|--------------|-------|
| `App.jsx` | 13 imports | 13 |
| `pages/Action.jsx` | 3 imports | 3 |
| `pages/Generator.jsx` | 3 imports | 3 |
| `pages/History.jsx` | 3 imports | 3 |
| `pages/TrainingMonitor.jsx` | 3 imports | 3 |
| `components/TrainingMonitor.jsx` | 2 imports | 2 |
| `main.jsx` | 2 imports | 2 |
| `components/examples/ApiHookExamples.jsx` | 1 imports | 1 |
| `components/MeshCanvas.jsx` | 1 imports | 1 |
| `components/MeshCanvasTest.jsx` | 1 imports | 1 |
