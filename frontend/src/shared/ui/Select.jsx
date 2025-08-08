import React from 'react'
import { cn } from '../utils/cn'

const Select = React.forwardRef(({ 
  className, 
  size = 'default',
  variant = 'default',
  disabled = false,
  error,
  label,
  placeholder,
  options = [],
  children,
  ...props 
}, ref) => {
  const baseClasses = 'w-full bg-bg-secondary border rounded-lg px-3 py-2 text-text-primary transition-colors focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed appearance-none'
  
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
  
  const selectVariant = error ? 'error' : variant
  
  const classes = cn(
    baseClasses,
    variants[selectVariant],
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
      <div className="relative">
        <select
          className={classes}
          ref={ref}
          disabled={disabled}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((option, index) => (
            <option key={index} value={option.value}>
              {option.label}
            </option>
          ))}
          {children}
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
          <svg
            className="w-4 h-4 text-text-secondary"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>
      {error && (
        <p className="text-red-500 text-sm">
          {error}
        </p>
      )}
    </div>
  )
})

Select.displayName = 'Select'

export default Select
