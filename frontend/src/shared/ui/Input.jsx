import React from 'react'
import { cn } from '../utils/cn'

const Input = React.forwardRef(({ 
  className, 
  type = 'text',
  size = 'default',
  variant = 'default',
  disabled = false,
  error,
  label,
  placeholder,
  ...props 
}, ref) => {
  const baseClasses = 'w-full bg-bg-secondary border rounded-lg px-3 py-2 text-text-primary transition-colors focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    default: 'border-border-custom focus:ring-accent focus:border-accent',
    error: 'border-red-500 focus:ring-red-500 focus:border-red-500',
    success: 'border-green-500 focus:ring-green-500 focus:border-green-500'
  }
  
  const sizes = {
    sm: 'px-2 py-1 text-sm',
    default: 'px-3 py-2',
    lg: 'px-4 py-3 text-lg'
  }
  
  const inputVariant = error ? 'error' : variant
  
  const classes = cn(
    baseClasses,
    variants[inputVariant],
    sizes[size],
    className
  )
  
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-text-secondary text-sm font-medium">
          {label}
        </label>
      )}
      <input
        type={type}
        className={classes}
        ref={ref}
        disabled={disabled}
        placeholder={placeholder}
        {...props}
      />
      {error && (
        <p className="text-red-500 text-sm">
          {error}
        </p>
      )}
    </div>
  )
})

Input.displayName = 'Input'

export default Input
