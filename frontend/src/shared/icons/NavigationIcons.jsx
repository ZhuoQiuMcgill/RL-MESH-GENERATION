import { 
  BarChart3,
  Train,
  History,
  Star,
  Ruler,
  Palette,
  Zap,
  Settings,
  Sun,
  Moon,
  Monitor,
  Menu,
  X,
  ArrowLeft
} from 'lucide-react'
import Icon from './Icon'

/**
 * Navigation Icons - Replaces emoji icons with accessible Lucide icons
 */

// Dashboard icon (📊 → BarChart3)
export const DashboardIcon = (props) => (
  <Icon aria-label="Dashboard" {...props}>
    <BarChart3 />
  </Icon>
)

// Train icon (🚂 → Train)
export const TrainIcon = (props) => (
  <Icon aria-label="Train" {...props}>
    <Train />
  </Icon>
)

// History icon (📋 → History)
export const HistoryIcon = (props) => (
  <Icon aria-label="History" {...props}>
    <History />
  </Icon>
)

// Quality/Star icon (⭐ → Star)
export const QualityIcon = (props) => (
  <Icon aria-label="Quality" {...props}>
    <Star />
  </Icon>
)

// Geometry/Ruler icon (📐 → Ruler)
export const GeometryIcon = (props) => (
  <Icon aria-label="Geometry" {...props}>
    <Ruler />
  </Icon>
)

// Canvas/Palette icon (🎨 → Palette)
export const CanvasIcon = (props) => (
  <Icon aria-label="Canvas" {...props}>
    <Palette />
  </Icon>
)

// Angle icon (📐 → Ruler)
export const AngleIcon = (props) => (
  <Icon aria-label="Angle" {...props}>
    <Ruler />
  </Icon>
)

// Action icon (⚡ → Zap)
export const ActionIcon = (props) => (
  <Icon aria-label="Action" {...props}>
    <Zap />
  </Icon>
)

// Generator icon (🔧 → Settings)
export const GeneratorIcon = (props) => (
  <Icon aria-label="Generator" {...props}>
    <Settings />
  </Icon>
)

// Theme icons
export const SunIcon = (props) => (
  <Icon aria-label="Light mode" {...props}>
    <Sun />
  </Icon>
)

export const MoonIcon = (props) => (
  <Icon aria-label="Dark mode" {...props}>
    <Moon />
  </Icon>
)

export const MonitorIcon = (props) => (
  <Icon aria-label="System theme" {...props}>
    <Monitor />
  </Icon>
)

// Menu icons
export const MenuIcon = (props) => (
  <Icon aria-label="Open menu" {...props}>
    <Menu />
  </Icon>
)

export const CloseIcon = (props) => (
  <Icon aria-label="Close menu" {...props}>
    <X />
  </Icon>
)

// Back arrow icon
export const BackIcon = (props) => (
  <Icon aria-label="Go back" {...props}>
    <ArrowLeft />
  </Icon>
)

// Icon mapping object for easy reference
export const NavigationIconMap = {
  dashboard: DashboardIcon,
  train: TrainIcon,
  history: HistoryIcon,
  quality: QualityIcon,
  geometry: GeometryIcon,
  canvas: CanvasIcon,
  angle: AngleIcon,
  action: ActionIcon,
  generator: GeneratorIcon,
  sun: SunIcon,
  moon: MoonIcon,
  monitor: MonitorIcon,
  menu: MenuIcon,
  close: CloseIcon,
  back: BackIcon
}
