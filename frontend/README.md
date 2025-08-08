# RL Mesh Generation Frontend

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-org/rl-mesh-generation)
[![Test Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen)](https://github.com/your-org/rl-mesh-generation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![React](https://img.shields.io/badge/React-19.1.1-61DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-7.1.0-646CFF)](https://vitejs.dev/)

A modern React-based frontend application for Reinforcement Learning Mesh Generation, featuring real-time 3D visualization, training monitoring, and comprehensive mesh analysis tools.

## 🚀 Features

### Core Modules
- **🎯 Dashboard** - System overview and real-time status monitoring
- **🚂 Training** - ML training management with live progress tracking
- **🎨 Canvas** - Interactive 3D mesh visualization with WebGL
- **📊 Analytics** - Training history, quality analysis, and performance metrics
- **🔧 Tools** - Geometry processing, angle analysis, and mesh generation

### Technical Highlights
- **⚡ Modern Stack** - React 19, Vite 7, Tailwind CSS 4
- **🎨 Design System** - Token-based design with light/dark themes
- **🧪 Testing** - Comprehensive unit and E2E testing with Playwright
- **📱 Responsive** - Mobile-first responsive design
- **♿ Accessible** - WCAG-compliant accessibility features
- **🔄 Real-time** - Live data updates with efficient polling
- **⚙️ Performance** - Optimized bundling with lazy loading

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Development](#development)
- [Testing](#testing)
- [Building](#building)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/rl-mesh-generation.git
cd rl-mesh-generation/frontend

# Install dependencies
npm install

# Set up environment
cp .env.example .env

# Start development server
npm run dev

# Open in browser: http://localhost:5173
```

## 📦 Installation

### Prerequisites

- **Node.js** >= 18.0.0
- **npm** >= 8.0.0
- **Modern Browser** (Chrome, Firefox, Safari, Edge)

### Environment Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/your-org/rl-mesh-generation.git
   cd rl-mesh-generation/frontend
   npm install
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   VITE_ENABLE_DEBUG=false
   ```

3. **Verify Installation**
   ```bash
   npm run test
   npm run build
   npm run dev
   ```

## 🛠️ Development

### Available Scripts

```bash
# Development
npm run dev              # Start dev server with HMR
npm run dev:debug        # Start with debug logging

# Building
npm run build            # Production build
npm run build:analyze    # Build with bundle analysis
npm run preview          # Preview production build

# Testing
npm run test             # Run unit tests (watch mode)
npm run test:run         # Run tests once
npm run test:coverage    # Generate coverage report
npm run e2e              # Run E2E tests
npm run e2e:ui           # Run E2E tests with UI
npm run test:all         # Run all tests

# Code Quality
npm run lint             # Lint code
npm run lint:fix         # Fix linting issues
npm run inventory        # Generate code inventory
```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make Changes**
   - Follow [coding standards](CONTRIBUTING.md#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Changes**
   ```bash
   npm run test:all
   npm run lint
   ```

4. **Submit PR**
   - Follow [PR process](CONTRIBUTING.md#pull-request-process)
   - Ensure all checks pass
   - Request review

## 🧪 Testing

### Test Structure

```
src/
├── **/__tests__/        # Unit tests
├── **/*.test.jsx        # Component tests
e2e/                     # E2E tests
├── *.spec.js           # Playwright tests
src/test/               # Test utilities
├── setup.js            # Test configuration
├── mocks/              # API mocks
└── utils/              # Test helpers
```

### Running Tests

```bash
# Unit tests
npm run test             # Watch mode
npm run test:run         # Single run
npm run test:ui          # Interactive UI

# E2E tests
npm run e2e              # Headless
npm run e2e:headed       # With browser
npm run e2e:debug        # Debug mode

# Coverage
npm run test:coverage    # Generate coverage report
```

### Test Coverage Requirements

- **Minimum**: 70% coverage for all metrics
- **Components**: Render, interaction, and edge case tests
- **Hooks**: Business logic and state management
- **E2E**: Critical user workflows

## 🏗️ Building

### Production Build

```bash
# Standard build
npm run build

# Build with analysis
npm run build:analyze

# Preview build
npm run preview
```

### Build Output

```
dist/
├── index.html           # Main entry point
├── assets/             # Optimized assets
│   ├── index-[hash].js # Main application bundle
│   ├── vendor-[hash].js # Third-party dependencies
│   └── [route]-[hash].js # Route-based chunks
└── bundle-analysis.html # Bundle analysis report
```

### Performance Targets

- **Bundle Size**: < 200KB gzipped
- **First Paint**: < 1.5s
- **Lighthouse Score**: > 90
- **Test Coverage**: > 70%

## 🏛️ Architecture

### High-Level Structure

```
src/
├── app/                 # Application configuration
├── modules/             # Feature modules (domain-driven)
│   ├── training/       # Training management
│   ├── dashboard/      # System overview
│   ├── canvas/         # 3D visualization
│   └── .../           # Other modules
├── shared/             # Shared components & utilities
│   ├── ui/            # Component library
│   ├── layout/        # Layout components
│   └── icons/         # Icon system
├── core/               # Core infrastructure
│   ├── api/           # API client & hooks
│   ├── hooks/         # Common React hooks
│   └── utils/         # Core utilities
└── context/            # Global state providers
```

### Key Principles

- **Modular Design** - Domain-driven module organization
- **Component Composition** - Reusable, composable UI components
- **Custom Hooks** - Encapsulated business logic
- **Design System** - Consistent styling with design tokens
- **Performance First** - Optimized for speed and efficiency

### Technology Stack

| Category | Technology | Version | Purpose |
|----------|------------|---------|----------|
| **Framework** | React | 19.1.1 | UI library |
| **Build Tool** | Vite | 7.1.0 | Development & bundling |
| **Routing** | React Router | 7.8.0 | Client-side routing |
| **Styling** | Tailwind CSS | 4.1.11 | Utility-first CSS |
| **Testing** | Vitest | 3.2.4 | Unit testing |
| **E2E Testing** | Playwright | 1.54.2 | End-to-end testing |
| **Icons** | Lucide React | 0.537.0 | Icon library |
| **Linting** | ESLint | 9.32.0 | Code quality |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information about:

- [Development setup](CONTRIBUTING.md#getting-started)
- [Coding standards](CONTRIBUTING.md#coding-standards)
- [Testing guidelines](CONTRIBUTING.md#testing-guidelines)
- [Pull request process](CONTRIBUTING.md#pull-request-process)

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Documentation

### Comprehensive Documentation

- **[Architecture Overview](data/docs/architecture/FINAL_ARCHITECTURE.md)** - System architecture and design decisions
- **[Component Map](data/docs/architecture/COMPONENT_MAP.md)** - Component relationships and patterns
- **[Design System](data/docs/styling-and-design/design-system.md)** - Design tokens and styling guidelines
- **[Operational Runbook](data/docs/operations/RUNBOOK.md)** - Deployment and maintenance guide
- **[API Documentation](data/docs/state-and-api/)** - API integration patterns
- **[Migration Guide](data/docs/MIGRATION_IMPLEMENTATION_GUIDE.md)** - Migration instructions

### Quick References

- **[Component Library](src/shared/ui/README.md)** - UI component documentation
- **[Hook Documentation](src/hooks/README.md)** - Custom hooks reference
- **[Module Structure](src/modules/README.md)** - Module organization guide
- **[Testing Guide](CONTRIBUTING.md#testing-guidelines)** - Testing best practices

## 🚀 Deployment

### Environment Configuration

```bash
# Development
VITE_API_BASE_URL=http://localhost:8000
VITE_ENABLE_DEBUG=true

# Production
VITE_API_BASE_URL=https://api.production.com
VITE_ENABLE_DEBUG=false
```

### Deployment Steps

1. **Build Application**
   ```bash
   npm run build
   ```

2. **Run Tests**
   ```bash
   npm run test:all
   ```

3. **Deploy**
   ```bash
   # Deploy dist/ folder to your hosting provider
   ```

See [Operational Runbook](data/docs/operations/RUNBOOK.md) for detailed deployment procedures.

## 📊 Performance

### Bundle Analysis

```bash
npm run build:analyze
```

View the generated report at `dist/bundle-analysis.html`

### Key Metrics

- **Bundle Size**: ~150KB gzipped
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3s
- **Lighthouse Performance**: 90+

## 🐛 Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check Node.js version
node --version  # Should be >= 18

# Clear dependencies
rm -rf node_modules package-lock.json
npm install
```

**Build failures:**
```bash
# Clear build cache
rm -rf dist/ .vite/
npm run build
```

**API connection issues:**
```bash
# Verify environment variables
echo $VITE_API_BASE_URL

# Test API endpoint
curl $VITE_API_BASE_URL/health
```

See [Troubleshooting Guide](data/docs/operations/RUNBOOK.md#troubleshooting-guide) for more solutions.

## 📈 Roadmap

### Upcoming Features

- [ ] **Real-time Collaboration** - Multi-user mesh editing
- [ ] **Advanced Analytics** - ML model performance insights  
- [ ] **Export System** - Multiple mesh format support
- [ ] **Plugin Architecture** - Extensible module system
- [ ] **Mobile App** - React Native companion app

### Technical Improvements

- [ ] **Server-Side Rendering** - Next.js migration
- [ ] **GraphQL Integration** - Efficient data fetching
- [ ] **Service Workers** - Offline functionality
- [ ] **WebAssembly** - High-performance mesh processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **React Team** - For the excellent React framework
- **Vite Team** - For the fast and modern build tool
- **Tailwind CSS** - For the utility-first CSS framework
- **Playwright Team** - For comprehensive E2E testing
- **Open Source Community** - For the amazing ecosystem

## 📞 Support

- **Documentation**: Check our [comprehensive docs](data/docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/rl-mesh-generation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rl-mesh-generation/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for the RL Mesh Generation community**
