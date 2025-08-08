import { 
  Circle,
  CheckCircle,
  XCircle,
  AlertCircle,
  Play,
  Pause,
  Square,
  RotateCcw,
  Eye,
  EyeOff,
  Info,
  Loader2,
  TrendingUp,
  Activity
} from 'lucide-react'
import Icon from './Icon'

/**
 * Status and Training Icons - Provides semantic meaning with proper accessibility
 */

// Status indicators
export const StatusActiveIcon = (props) => (
  <Icon aria-label="Active status" variant="success" {...props}>
    <Circle className="fill-current" />
  </Icon>
)

export const StatusCompleteIcon = (props) => (
  <Icon aria-label="Complete status" variant="success" {...props}>
    <CheckCircle />
  </Icon>
)

export const StatusErrorIcon = (props) => (
  <Icon aria-label="Error status" variant="danger" {...props}>
    <XCircle />
  </Icon>
)

export const StatusWarningIcon = (props) => (
  <Icon aria-label="Warning status" variant="warning" {...props}>
    <AlertCircle />
  </Icon>
)

export const StatusInfoIcon = (props) => (
  <Icon aria-label="Information" variant="primary" {...props}>
    <Info />
  </Icon>
)

// Training controls
export const PlayIcon = (props) => (
  <Icon aria-label="Start training" variant="success" {...props}>
    <Play className="fill-current" />
  </Icon>
)

export const PauseIcon = (props) => (
  <Icon aria-label="Pause training" variant="warning" {...props}>
    <Pause className="fill-current" />
  </Icon>
)

export const StopIcon = (props) => (
  <Icon aria-label="Stop training" variant="danger" {...props}>
    <Square className="fill-current" />
  </Icon>
)

export const ResetIcon = (props) => (
  <Icon aria-label="Reset" {...props}>
    <RotateCcw />
  </Icon>
)

// Loading indicator
export const LoadingIcon = (props) => (
  <Icon aria-label="Loading" {...props}>
    <Loader2 className="animate-spin" />
  </Icon>
)

// Visibility toggles
export const VisibleIcon = (props) => (
  <Icon aria-label="Show" {...props}>
    <Eye />
  </Icon>
)

export const HiddenIcon = (props) => (
  <Icon aria-label="Hide" {...props}>
    <EyeOff />
  </Icon>
)

// Training progress and metrics
export const TrendingUpIcon = (props) => (
  <Icon aria-label="Trending up" variant="success" {...props}>
    <TrendingUp />
  </Icon>
)

export const ActivityIcon = (props) => (
  <Icon aria-label="Activity" {...props}>
    <Activity />
  </Icon>
)

// Color-coded status dots for training states
export const StatusDot = ({ status, className = '', ...props }) => {
  const getStatusProps = (status) => {
    switch (status) {
      case 'running':
      case 'active':
        return { 
          variant: 'success', 
          'aria-label': 'Running',
          className: 'text-green-500 animate-pulse'
        }
      case 'complete':
      case 'finished':
        return { 
          variant: 'success', 
          'aria-label': 'Complete',
          className: 'text-green-500'
        }
      case 'error':
      case 'failed':
        return { 
          variant: 'danger', 
          'aria-label': 'Error',
          className: 'text-red-500'
        }
      case 'warning':
      case 'paused':
        return { 
          variant: 'warning', 
          'aria-label': 'Warning',
          className: 'text-yellow-500'
        }
      case 'idle':
      case 'inactive':
      default:
        return { 
          variant: 'muted', 
          'aria-label': 'Idle',
          className: 'text-gray-500'
        }
    }
  }
  
  const statusProps = getStatusProps(status)
  
  return (
    <Icon 
      size={12}
      className={`${statusProps.className} ${className}`}
      {...statusProps}
      {...props}
    >
      <Circle className="fill-current" />
    </Icon>
  )
}

// Status icon mapping
export const StatusIconMap = {
  active: StatusActiveIcon,
  complete: StatusCompleteIcon,
  error: StatusErrorIcon,
  warning: StatusWarningIcon,
  info: StatusInfoIcon,
  play: PlayIcon,
  pause: PauseIcon,
  stop: StopIcon,
  reset: ResetIcon,
  loading: LoadingIcon,
  visible: VisibleIcon,
  hidden: HiddenIcon,
  trending: TrendingUpIcon,
  activity: ActivityIcon
}
