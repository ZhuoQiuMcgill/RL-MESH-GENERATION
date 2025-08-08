import React from 'react'
import { cn } from '../../shared/utils/cn'

const EmptyState = ({ 
  className,
  icon,
  title,
  description,
  action,
  size = 'default',
  ...props 
}) => {
  const sizes = {
    sm: {
      container: 'py-8',
      icon: 'text-4xl mb-3',
      title: 'text-lg mb-2',
      description: 'text-sm'
    },
    default: {
      container: 'py-12',
      icon: 'text-6xl mb-4',
      title: 'text-xl mb-3',
      description: 'text-base'
    },
    lg: {
      container: 'py-16',
      icon: 'text-8xl mb-6',
      title: 'text-2xl mb-4',
      description: 'text-lg'
    }
  }

  const currentSize = sizes[size]

  return (
    <div
      className={cn(
        'text-center',
        currentSize.container,
        className
      )}
      {...props}
    >
      {icon && (
        <div className={cn(
          'text-text-secondary opacity-60',
          currentSize.icon
        )}>
          {icon}
        </div>
      )}
      
      {title && (
        <h3 className={cn(
          'font-semibold text-text-primary',
          currentSize.title
        )}>
          {title}
        </h3>
      )}
      
      {description && (
        <p className={cn(
          'text-text-secondary max-w-md mx-auto',
          currentSize.description
        )}>
          {description}
        </p>
      )}
      
      {action && (
        <div className="mt-6">
          {action}
        </div>
      )}
    </div>
  )
}

EmptyState.displayName = 'EmptyState'

export default EmptyState
