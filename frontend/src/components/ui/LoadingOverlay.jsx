import React from 'react'
import { cn } from '../../shared/utils/cn'

const LoadingOverlay = ({ 
  className, 
  text = "Loading...",
  overlay = true,
  size = 'default',
  ...props 
}) => {
  const spinnerSizes = {
    sm: 'w-6 h-6',
    default: 'w-8 h-8',
    lg: 'w-12 h-12'
  }

  const textSizes = {
    sm: 'text-sm',
    default: 'text-base',
    lg: 'text-lg'
  }

  const Spinner = () => (
    <div className={cn(
      'animate-spin rounded-full border-2 border-border-custom border-t-accent',
      spinnerSizes[size]
    )}>
    </div>
  )

  const Content = () => (
    <div className="flex flex-col items-center justify-center space-y-3">
      <Spinner />
      {text && (
        <p className={cn(
          'text-text-secondary font-medium',
          textSizes[size]
        )}>
          {text}
        </p>
      )}
    </div>
  )

  if (!overlay) {
    return (
      <div className={cn('flex items-center justify-center p-4', className)} {...props}>
        <Content />
      </div>
    )
  }

  return (
    <div
      className={cn(
        'fixed inset-0 bg-bg-primary/80 backdrop-blur-sm flex items-center justify-center z-50',
        className
      )}
      {...props}
    >
      <div className="bg-card border border-border-custom rounded-xl p-8 shadow-lg">
        <Content />
      </div>
    </div>
  )
}

LoadingOverlay.displayName = 'LoadingOverlay'

export default LoadingOverlay
