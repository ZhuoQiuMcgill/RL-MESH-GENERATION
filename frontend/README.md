# RL Mesh Generation - Frontend

A React frontend for the RL Mesh Generation project, built with Vite and styled with Tailwind CSS.

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm

### Installation
```bash
npm install
```

### Development Scripts

- **`npm run dev`** - Start the development server with hot reloading
- **`npm run build`** - Build the project for production
- **`npm run preview`** - Preview the production build locally
- **`npm run lint`** - Run ESLint for code quality checks

## Project Architecture

This project follows a feature-based architecture with shared components and utilities:

```
src/
├── components/          # Reusable UI components (Button, Card, Layout)
├── features/           # Feature-based modules
│   ├── train/         # Training functionality
│   ├── predict/       # Prediction functionality  
│   └── history/       # History/results functionality
├── pages/             # Page components
├── router/            # Routing configuration
├── shared/            # Shared resources
│   ├── api/          # API client and utilities
│   ├── components/   # Shared UI components (Header, Layout)
│   └── styles/       # Global styles and themes
└── utils/            # Utility functions and helpers
```

### Folder Structure Conventions

- **`components/`** - Generic, reusable UI components that can be used across features
- **`features/`** - Self-contained feature modules with their own components, logic, and routes
- **`shared/`** - Resources shared across multiple features (layouts, API clients, global styles)
- **`pages/`** - Top-level page components that compose features and shared components
- **`router/`** - Application routing configuration using React Router
- **`utils/`** - Pure utility functions and helper methods

## Coding Conventions

### General Guidelines

1. **File Naming**
   - Use PascalCase for React components: `Button.jsx`, `UserProfile.jsx`
   - Use camelCase for utilities and non-component files: `helpers.js`, `apiClient.js`
   - Use kebab-case for CSS files: `theme.css`

2. **Component Structure**
   - Export components as default exports
   - Use functional components with hooks
   - Destructure props in function parameters
   - Use prop spreading sparingly and explicitly

3. **Code Organization**
   - Group related functionality in feature folders
   - Create index files for clean imports
   - Keep components small and focused on single responsibilities

### Example Component Pattern

```jsx
import React from 'react';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  disabled = false, 
  onClick, 
  className = '',
  ...props 
}) => {
  // Component logic here
  
  return (
    <button
      className={buttonClasses}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
```

## Tailwind CSS Usage

### Configuration

Tailwind CSS is configured to scan all JavaScript and JSX files in the `src/` directory:

```javascript
// tailwind.config.js
export default {
  content: ['src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### Styling Conventions

1. **Utility-First Approach**
   - Use Tailwind utility classes for styling
   - Avoid custom CSS unless absolutely necessary
   - Create reusable component patterns rather than custom styles

2. **Component Variants**
   - Define variants using JavaScript objects
   - Use conditional classes for different states
   - Example pattern from Button component:

```jsx
const variants = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700',
  secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
  outline: 'border border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
};
```

3. **Responsive Design**
   - Use Tailwind's responsive prefixes: `sm:`, `md:`, `lg:`, `xl:`
   - Design mobile-first, then add larger screen styles
   - Example: `px-4 sm:px-6 lg:px-8`

4. **Layout Patterns**
   - Use Flexbox and Grid utilities for layouts
   - Common patterns:
     - `flex items-center justify-center` - Center content
     - `max-w-7xl mx-auto px-4` - Constrain and center container
     - `min-h-screen` - Full viewport height

### Global Styles

Global Tailwind directives are imported in `src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Development Guidelines

1. **State Management**
   - Use React's built-in state management (useState, useReducer)
   - Consider context for shared state across components
   - Keep state as close to where it's used as possible

2. **API Integration**
   - Use the shared API client in `src/shared/api/`
   - Handle loading and error states consistently
   - Use async/await for API calls

3. **Error Handling**
   - Implement proper error boundaries
   - Provide user-friendly error messages
   - Log errors appropriately for debugging

4. **Performance**
   - Use React.memo for expensive components
   - Implement proper key props for lists
   - Avoid inline object/function creation in render methods

## Building and Deployment

The project uses Vite for building:

- **Development**: `npm run dev` starts the dev server at `http://localhost:5173`
- **Production Build**: `npm run build` creates optimized files in the `dist/` directory
- **Preview**: `npm run preview` serves the production build locally

Build output includes:
- Minified JavaScript and CSS
- Asset optimization
- Code splitting for better performance
