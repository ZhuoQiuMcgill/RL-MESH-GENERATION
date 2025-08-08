import React from 'react'
import { cn } from '../../utils/constants'

const Button = React.forwardRef(({ 
  className, 
  variant = 'primary', 
  size = 'default',
  disabled = false,
  loading = false,
  children, 
  ...props 
}, ref) => {
  const baseClasses = 'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-card border border-border-custom text-text-primary hover:bg-card-hover focus:ring-accent',
    secondary: 'bg-bg-secondary border border-border-custom text-text-secondary hover:bg-card hover:text-text-primary focus:ring-accent',
    danger: 'bg-red-600 hover:bg-red-700 text-white border border-red-600 focus:ring-red-500',
    success: 'bg-green-600 hover:bg-green-700 text-white border border-green-600 focus:ring-green-500',
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white border border-yellow-600 focus:ring-yellow-500',
    outline: 'border border-border-custom text-text-primary hover:bg-card-hover focus:ring-accent',
    ghost: 'text-text-primary hover:bg-card-hover focus:ring-accent'
  }
  
  const sizes = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    default: 'px-5 py-3',
    lg: 'px-6 py-4 text-lg',
    xl: 'px-8 py-5 text-xl'
  }
  
  const classes = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    loading && 'cursor-wait',
    className
  )
  
  return (
    <button
      className={classes}
      ref={ref}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <svg 
          className="animate-spin -ml-1 mr-3 h-5 w-5" 
          xmlns="http://www.w3.org/2000/svg" 
          fill="none" 
          viewBox="0 0 24 24"
        >
          <circle 
            className="opacity-25" 
            cx="12" 
            cy="12" 
            r="10" 
            stroke="currentColor" 
            strokeWidth="4"
          />
          <path 
            className="opacity-75" 
            fill="currentColor" 
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      {children}
    </button>
  )
})

Button.displayName = 'Button'

export default Button
