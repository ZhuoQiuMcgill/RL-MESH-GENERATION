import React from 'react'
import { cn } from '../../shared/utils/cn'

const CompactStatusBar = React.forwardRef(({ 
  className, 
  items = [],
  ...props 
}, ref) => {
  return (
    <div
      ref={ref}
      className={cn('space-y-4', className)}
      {...props}
    >
      {items.map((item, index) => (
        <div key={index} className="flex justify-between items-center">
          <span className="text-text-secondary text-sm">
            {item.label}
          </span>
          <span className={cn(
            'font-medium text-sm',
            item.color ? `text-${item.color}` : 'text-text-primary'
          )}>
            {item.value}
          </span>
        </div>
      ))}
    </div>
  )
})

CompactStatusBar.displayName = 'CompactStatusBar'

export default CompactStatusBar
