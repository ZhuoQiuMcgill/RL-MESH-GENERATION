// Feature Modules - Central Export Point

// Training Module
export * from './training';

// Dashboard Module  
export * from './dashboard';

// History Module
export * from './history';

// Quality Module
export * from './quality';

// Geometry Module
export * from './geometry';

// Canvas Module
export * from './canvas';

// Angle Module
export * from './angle';

// Action Module
export * from './action';

// Generator Module
export * from './generator';

// Module metadata for development and debugging
export const MODULE_INFO = {
  training: {
    name: 'Training Module',
    description: 'RL model training and monitoring',
    version: '1.0.0'
  },
  dashboard: {
    name: 'Dashboard Module', 
    description: 'System overview and metrics',
    version: '1.0.0'
  },
  history: {
    name: 'History Module',
    description: 'Activity and session history',
    version: '1.0.0'
  },
  quality: {
    name: 'Quality Module',
    description: 'Mesh quality analysis',
    version: '1.0.0'
  },
  geometry: {
    name: 'Geometry Module',
    description: 'Geometric operations and management',
    version: '1.0.0'
  },
  canvas: {
    name: 'Canvas Module',
    description: '3D visualization and rendering',
    version: '1.0.0'
  },
  angle: {
    name: 'Angle Module',
    description: 'Angle analysis and measurement',
    version: '1.0.0'
  },
  action: {
    name: 'Action Module',
    description: 'Action management and workflow',
    version: '1.0.0'
  },
  generator: {
    name: 'Generator Module',
    description: 'Mesh generation algorithms',
    version: '1.0.0'
  }
};
