import React from 'react'
import { cn } from '../utils/cn'

const Badge = React.forwardRef(({ 
  className,
  variant = 'default',
  size = 'default',
  children,
  ...props 
}, ref) => {
  const baseClasses = 'inline-flex items-center rounded-full font-medium transition-all'
  
  const variants = {
    default: 'bg-bg-secondary text-text-primary border border-border-custom',
    primary: 'bg-accent text-white',
    secondary: 'bg-gray-100 text-gray-800 border border-gray-300',
    success: 'bg-green-100 text-green-800 border border-green-300',
    warning: 'bg-yellow-100 text-yellow-800 border border-yellow-300',
    danger: 'bg-red-100 text-red-800 border border-red-300',
    info: 'bg-blue-100 text-blue-800 border border-blue-300',
    outline: 'bg-transparent border border-border-custom text-text-primary',
    solid: 'bg-card text-text-primary shadow-sm'
  }
  
  const sizes = {
    xs: 'px-2 py-0.5 text-xs',
    sm: 'px-2.5 py-1 text-xs',
    default: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base'
  }
  
  const classes = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    className
  )
  
  return (
    <span
      ref={ref}
      className={classes}
      {...props}
    >
      {children}
    </span>
  )
})

Badge.displayName = 'Badge'

export default Badge
