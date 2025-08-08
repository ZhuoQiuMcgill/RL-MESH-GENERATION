import React from 'react'
import { cn } from '../../shared/utils/cn'

const PanelCard = React.forwardRef(({ 
  className, 
  children, 
  title,
  subtitle,
  ...props 
}, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        'bg-card border border-border-custom rounded-xl p-6',
        className
      )}
      {...props}
    >
      {(title || subtitle) && (
        <div className="mb-4">
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
      )}
      {children}
    </div>
  )
})

PanelCard.displayName = 'PanelCard'

export default PanelCard
