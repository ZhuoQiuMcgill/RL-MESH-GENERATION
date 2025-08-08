import React from 'react'
import { cn } from '../utils/cn'

const StatsCard = React.forwardRef(({ 
  className, 
  title,
  value,
  description,
  icon,
  trend,
  trendDirection = 'up', // 'up', 'down', 'neutral'
  trendValue,
  loading = false,
  ...props 
}, ref) => {
  const baseClasses = 'bg-card border border-border-custom rounded-xl p-6'
  
  const getTrendColor = () => {
    switch (trendDirection) {
      case 'up': return 'text-green-500'
      case 'down': return 'text-red-500'
      case 'neutral': 
      default: return 'text-text-secondary'
    }
  }

  const getTrendIcon = () => {
    switch (trendDirection) {
      case 'up': return '↗'
      case 'down': return '↘'
      case 'neutral': 
      default: return '→'
    }
  }
  
  const classes = cn(baseClasses, className)
  
  if (loading) {
    return (
      <div ref={ref} className={classes} {...props}>
        <div className="animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <div className="h-4 bg-bg-secondary rounded w-20"></div>
            {icon && <div className="h-8 w-8 bg-bg-secondary rounded"></div>}
          </div>
          <div className="h-8 bg-bg-secondary rounded w-16 mb-2"></div>
          <div className="h-3 bg-bg-secondary rounded w-24"></div>
        </div>
      </div>
    )
  }
  
  return (
    <div ref={ref} className={classes} {...props}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wide">
          {title}
        </h3>
        {icon && (
          <div className="text-2xl opacity-60">
            {typeof icon === 'string' ? icon : icon}
          </div>
        )}
      </div>
      
      <div className="mb-2">
        <div className="text-3xl font-bold text-text-primary">
          {value}
        </div>
      </div>
      
      <div className="flex items-center justify-between">
        <p className="text-sm text-text-secondary">
          {description}
        </p>
        {(trend || trendValue) && (
          <div className={cn("flex items-center text-xs font-medium", getTrendColor())}>
            <span className="mr-1">{getTrendIcon()}</span>
            {trendValue && <span>{trendValue}</span>}
            {trend && <span className="ml-1">{trend}</span>}
          </div>
        )}
      </div>
    </div>
  )
})

StatsCard.displayName = 'StatsCard'

export default StatsCard
