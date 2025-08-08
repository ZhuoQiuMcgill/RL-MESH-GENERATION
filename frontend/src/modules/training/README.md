# Training Module

## Overview

The training module manages all aspects of reinforcement learning model training for mesh generation. This module handles training sessions, monitoring progress, and providing training analytics.

## Public Surface

### Pages
- `pages/Train.jsx` - Main training interface for configuring and starting training sessions
- `pages/TrainingMonitor.jsx` - Real-time training progress monitoring and visualization

### Components
- `components/TrainingMonitor.jsx` - Presentational component for displaying training metrics
- `components/TrainingProgress.jsx` - Progress indicator for training sessions
- `components/TrainingConfiguration.jsx` - Form component for training parameters

### Hooks
- `hooks/useTraining.js` - Main training logic and state management
- `hooks/useTrainingMetrics.js` - Training metrics data fetching and polling
- `hooks/useTrainingSession.js` - Individual training session management

### Services
- `services/trainingApi.js` - Training API integration and endpoints
- `services/trainingStorage.js` - Local storage management for training data

## Module Interface

### Exports
```javascript
// Pages
export { default as TrainPage } from './pages/Train'
export { default as TrainingMonitorPage } from './pages/TrainingMonitor'

// Hooks
export { useTraining } from './hooks/useTraining'
export { useTrainingMetrics } from './hooks/useTrainingMetrics'

// Services (if needed by other modules)
export { trainingApi } from './services/trainingApi'
```

### Key Features
- Training session configuration and management
- Real-time training progress monitoring
- Training metrics visualization
- Training history and analytics
- Model checkpoint management

### Dependencies
- Core API client (`core/api`)
- Shared UI components (`shared/ui`)
- Canvas module for visualization integration

### Data Flow
1. User configures training parameters via Train page
2. Training session is initiated through trainingApi service
3. useTrainingMetrics hook polls for real-time updates
4. TrainingMonitor components display progress and metrics
5. Training results are stored and made available to other modules
