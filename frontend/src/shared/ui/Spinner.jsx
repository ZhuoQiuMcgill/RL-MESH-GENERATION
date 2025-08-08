import React from 'react'
import { cn } from '../utils/cn'

const Spinner = React.forwardRef(({ 
  className,
  variant = 'default',
  size = 'default',
  ...props 
}, ref) => {
  const variants = {
    default: 'text-accent',
    primary: 'text-text-primary',
    secondary: 'text-text-secondary',
    success: 'text-green-500',
    warning: 'text-yellow-500',
    danger: 'text-red-500',
    white: 'text-white'
  }

  const sizes = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4',
    default: 'w-5 h-5',
    lg: 'w-6 h-6',
    xl: 'w-8 h-8',
    '2xl': 'w-10 h-10'
  }

  const classes = cn(
    'animate-spin',
    variants[variant],
    sizes[size],
    className
  )

  return (
    <svg
      ref={ref}
      className={classes}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      {...props}
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
  )
})

Spinner.displayName = 'Spinner'

export default Spinner
