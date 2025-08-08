import React from 'react'
import { cn } from '../../shared/utils/cn'

const Button = React.forwardRef(({ 
  className, 
  variant = 'primary', 
  size = 'default',
  disabled = false,
  children, 
  ...props 
}, ref) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-card border border-border-custom text-text-primary hover:bg-card-hover focus:ring-primary-start',
    secondary: 'bg-bg-secondary border border-border-custom text-text-secondary hover:bg-card hover:text-text-primary focus:ring-primary-start',
    danger: 'bg-error hover:bg-red-700 text-white border border-error focus:ring-red-500',
    success: 'bg-success hover:bg-green-700 text-white border border-success focus:ring-green-500',
    warning: 'bg-warning hover:bg-yellow-600 text-white border border-warning focus:ring-yellow-500',
    info: 'bg-info hover:bg-blue-700 text-white border border-info focus:ring-blue-500',
    outline: 'border border-border-custom text-text-primary hover:bg-card-hover focus:ring-primary-start',
    ghost: 'text-text-primary hover:bg-card-hover focus:ring-primary-start'
  }
  
  const sizes = {
    sm: 'h-8 px-3 text-sm rounded-md',
    default: 'h-10 px-4 rounded-md',
    lg: 'h-12 px-6 text-lg rounded-lg'
  }
  
  const classes = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    className
  )
  
  return (
    <button
      className={classes}
      ref={ref}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  )
})

Button.displayName = 'Button'

export default Button
