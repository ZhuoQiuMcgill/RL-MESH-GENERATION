import { lazy } from 'react'
import {
  DashboardIcon,
  TrainIcon,
  HistoryIcon,
  QualityIcon,
  GeometryIcon,
  CanvasIcon,
  AngleIcon,
  ActionIcon,
  GeneratorIcon
} from '../shared/icons'

// Lazy-loaded page components
const Dashboard = lazy(() => import('../pages/Dashboard'))
const Train = lazy(() => import('../pages/Train'))
const History = lazy(() => import('../pages/History'))
const Quality = lazy(() => import('../pages/Quality'))
const Geometry = lazy(() => import('../pages/Geometry'))
const Canvas = lazy(() => import('../pages/Canvas'))
const Angle = lazy(() => import('../pages/Angle'))
const Action = lazy(() => import('../pages/Action'))
const Generator = lazy(() => import('../pages/Generator'))

// Route configuration with metadata
export const routes = [
  {
    path: '/',
    element: Dashboard,
    title: 'Dashboard',
    breadcrumb: 'Dashboard',
    icon: DashboardIcon,
    description: 'Overview and main dashboard'
  },
  {
    path: '/train',
    element: Train,
    title: 'Train Model',
    breadcrumb: 'Train',
    icon: TrainIcon,
    description: 'Training configuration and monitoring'
  },
  {
    path: '/history',
    element: History,
    title: 'Training History',
    breadcrumb: 'History',
    icon: HistoryIcon,
    description: 'View training history and results'
  },
  {
    path: '/quality',
    element: Quality,
    title: 'Mesh Quality',
    breadcrumb: 'Quality',
    icon: QualityIcon,
    description: 'Analyze mesh quality metrics'
  },
  {
    path: '/geometry',
    element: Geometry,
    title: 'Geometry Analysis',
    breadcrumb: 'Geometry',
    icon: GeometryIcon,
    description: 'Geometric analysis and visualization'
  },
  {
    path: '/canvas',
    element: Canvas,
    title: 'Mesh Canvas',
    breadcrumb: 'Canvas',
    icon: CanvasIcon,
    description: 'Interactive mesh visualization'
  },
  {
    path: '/angle',
    element: Angle,
    title: 'Angle Analysis',
    breadcrumb: 'Angle',
    icon: AngleIcon,
    description: 'Angle distribution analysis'
  },
  {
    path: '/action',
    element: Action,
    title: 'Action Space',
    breadcrumb: 'Action',
    icon: ActionIcon,
    description: 'RL action space configuration'
  },
  {
    path: '/generator',
    element: Generator,
    title: 'Mesh Generator',
    breadcrumb: 'Generator',
    icon: GeneratorIcon,
    description: 'Generate and export meshes'
  }
]

// Helper functions for working with routes
export const getRouteByPath = (path) => {
  return routes.find(route => route.path === path)
}

export const getBreadcrumbLabel = (path) => {
  const route = getRouteByPath(path)
  return route ? route.breadcrumb : path
}

export const getRouteTitle = (path) => {
  const route = getRouteByPath(path)
  return route ? route.title : 'RL Mesh Generation'
}

export const getNavigationItems = () => {
  return routes.map(route => ({
    path: route.path,
    label: route.breadcrumb,
    icon: route.icon,
    title: route.title,
    description: route.description
  }))
}
