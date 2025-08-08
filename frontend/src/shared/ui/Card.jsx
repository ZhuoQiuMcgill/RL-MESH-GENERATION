import React from 'react'
import { cn } from '../utils/cn'

const Card = React.forwardRef(({ 
  className, 
  variant = 'default',
  size = 'default',
  children, 
  title,
  subtitle,
  headerAction,
  footer,
  ...props 
}, ref) => {
  const baseClasses = 'bg-card border border-border-custom rounded-xl'
  
  const variants = {
    default: 'bg-card border-border-custom',
    elevated: 'bg-card border-border-custom shadow-lg',
    outlined: 'bg-transparent border-2 border-border-custom',
    ghost: 'bg-transparent border-0'
  }
  
  const sizes = {
    sm: 'p-4',
    default: 'p-6',
    lg: 'p-8'
  }
  
  const classes = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    className
  )
  
  return (
    <div
      ref={ref}
      className={classes}
      {...props}
    >
      {(title || subtitle || headerAction) && (
        <div className="flex items-start justify-between mb-4">
          <div>
            {title && (
              <h3 className="text-xl font-semibold text-text-primary mb-1">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-text-secondary text-sm">
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && (
            <div className="ml-4">
              {headerAction}
            </div>
          )}
        </div>
      )}
      <div className="flex-1">
        {children}
      </div>
      {footer && (
        <div className="mt-4 pt-4 border-t border-border-custom">
          {footer}
        </div>
      )}
    </div>
  )
})

Card.displayName = 'Card'

export default Card
