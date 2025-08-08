import React from 'react'
import { cn } from '../../shared/utils/cn'

const FormInput = React.forwardRef(({ 
  className, 
  type = 'text',
  label,
  error,
  disabled = false,
  ...props 
}, ref) => {
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-text-secondary text-sm font-medium">
          {label}
        </label>
      )}
      <input
        type={type}
        className={cn(
          'w-full bg-bg-secondary border rounded-lg px-3 py-2 text-text-primary transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-accent focus:border-accent',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          error 
            ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
            : 'border-border-custom',
          className
        )}
        ref={ref}
        disabled={disabled}
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

FormInput.displayName = 'FormInput'

export default FormInput
