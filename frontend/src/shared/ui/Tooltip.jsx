import React, { useState } from 'react'
import { cn } from '../utils/cn'

const Tooltip = ({ 
  children,
  content,
  placement = 'top',
  variant = 'default',
  size = 'default',
  delay = 200,
  className,
  disabled = false,
  ...props 
}) => {
  const [isVisible, setIsVisible] = useState(false)
  const [showTimeout, setShowTimeout] = useState(null)
  const [hideTimeout, setHideTimeout] = useState(null)

  if (disabled || !content) {
    return children
  }

  const handleMouseEnter = () => {
    if (hideTimeout) {
      clearTimeout(hideTimeout)
      setHideTimeout(null)
    }
    
    const timeout = setTimeout(() => {
      setIsVisible(true)
    }, delay)
    
    setShowTimeout(timeout)
  }

  const handleMouseLeave = () => {
    if (showTimeout) {
      clearTimeout(showTimeout)
      setShowTimeout(null)
    }
    
    const timeout = setTimeout(() => {
      setIsVisible(false)
    }, 100)
    
    setHideTimeout(timeout)
  }

  const variants = {
    default: 'bg-gray-900 text-white border border-gray-700',
    light: 'bg-white text-gray-900 border border-gray-300 shadow-lg',
    error: 'bg-red-600 text-white border border-red-500',
    warning: 'bg-yellow-600 text-white border border-yellow-500',
    success: 'bg-green-600 text-white border border-green-500'
  }

  const sizes = {
    sm: 'px-2 py-1 text-xs',
    default: 'px-3 py-2 text-sm',
    lg: 'px-4 py-3 text-base'
  }

  const placements = {
    top: {
      tooltip: '-translate-x-1/2 bottom-full left-1/2 mb-2',
      arrow: 'top-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-b-transparent'
    },
    bottom: {
      tooltip: '-translate-x-1/2 top-full left-1/2 mt-2',
      arrow: 'bottom-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-t-transparent'
    },
    left: {
      tooltip: 'right-full top-1/2 -translate-y-1/2 mr-2',
      arrow: 'left-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-r-transparent'
    },
    right: {
      tooltip: 'left-full top-1/2 -translate-y-1/2 ml-2',
      arrow: 'right-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-l-transparent'
    }
  }

  const tooltipClasses = cn(
    'absolute z-50 rounded-lg font-medium pointer-events-none transition-opacity duration-200',
    'whitespace-nowrap max-w-xs',
    variants[variant],
    sizes[size],
    placements[placement].tooltip,
    isVisible ? 'opacity-100' : 'opacity-0 pointer-events-none',
    className
  )

  const arrowClasses = cn(
    'absolute w-0 h-0 border-4',
    placements[placement].arrow,
    variant === 'default' && 'border-gray-900',
    variant === 'light' && 'border-white',
    variant === 'error' && 'border-red-600',
    variant === 'warning' && 'border-yellow-600',
    variant === 'success' && 'border-green-600'
  )

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      {...props}
    >
      {children}
      <div className={tooltipClasses}>
        {content}
        <div className={arrowClasses} />
      </div>
    </div>
  )
}

Tooltip.displayName = 'Tooltip'

export default Tooltip
