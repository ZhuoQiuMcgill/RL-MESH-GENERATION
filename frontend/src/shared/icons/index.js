// Base icon component
export { default as Icon } from './Icon'

// Navigation icons
export {
  DashboardIcon,
  TrainIcon,
  HistoryIcon,
  QualityIcon,
  GeometryIcon,
  CanvasIcon,
  AngleIcon,
  ActionIcon,
  GeneratorIcon,
  SunIcon,
  MoonIcon,
  MonitorIcon,
  MenuIcon,
  CloseIcon,
  BackIcon,
  NavigationIconMap
} from './NavigationIcons'

// Status and training icons
export {
  StatusActiveIcon,
  StatusCompleteIcon,
  StatusErrorIcon,
  StatusWarningIcon,
  StatusInfoIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ResetIcon,
  LoadingIcon,
  VisibleIcon,
  HiddenIcon,
  TrendingUpIcon,
  ActivityIcon,
  StatusDot,
  StatusIconMap
} from './StatusIcons'

/**
 * Get an icon component by name
 * @param {string} iconName - The name of the icon
 * @returns {React.Component|null} The icon component or null if not found
 */
export const getIcon = (iconName) => {
  const allIcons = {
    // Navigation icons
    dashboard: 'DashboardIcon',
    train: 'TrainIcon',
    history: 'HistoryIcon',
    quality: 'QualityIcon',
    geometry: 'GeometryIcon',
    canvas: 'CanvasIcon',
    angle: 'AngleIcon',
    action: 'ActionIcon',
    generator: 'GeneratorIcon',
    sun: 'SunIcon',
    moon: 'MoonIcon',
    monitor: 'MonitorIcon',
    menu: 'MenuIcon',
    close: 'CloseIcon',
    back: 'BackIcon',
    
    // Status icons
    active: 'StatusActiveIcon',
    complete: 'StatusCompleteIcon',
    error: 'StatusErrorIcon',
    warning: 'StatusWarningIcon',
    info: 'StatusInfoIcon',
    play: 'PlayIcon',
    pause: 'PauseIcon',
    stop: 'StopIcon',
    reset: 'ResetIcon',
    loading: 'LoadingIcon',
    visible: 'VisibleIcon',
    hidden: 'HiddenIcon',
    trending: 'TrendingUpIcon',
    activity: 'ActivityIcon'
  }
  
  const componentName = allIcons[iconName?.toLowerCase()]
  return componentName || null
}

/**
 * Migration helper for emoji to icon conversion
 * Maps emoji characters to icon component names
 */
export const emojiToIconMap = {
  '📊': 'dashboard',
  '🚂': 'train', 
  '📋': 'history',
  '⭐': 'quality',
  '📐': 'geometry', // also used for angle
  '🎨': 'canvas',
  '⚡': 'action',
  '🔧': 'generator',
  '☀️': 'sun',
  '🌙': 'moon',
  '💻': 'monitor',
  '☰': 'menu',
  '✕': 'close',
  '←': 'back'
}

/**
 * Convert emoji to icon component name
 * @param {string} emoji - The emoji character
 * @returns {string|null} The icon component name or null if not found
 */
export const convertEmojiToIcon = (emoji) => {
  return emojiToIconMap[emoji] || null
}

// Development utilities can be added here as needed
