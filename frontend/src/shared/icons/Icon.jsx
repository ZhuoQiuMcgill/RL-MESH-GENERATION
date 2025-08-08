import { forwardRef } from 'react'

/**
 * Base Icon component with accessibility features
 * 
 * Features:
 * - ARIA attributes for screen readers
 * - Focus styles for keyboard navigation
 * - Consistent sizing and styling
 * - Support for different states (hover, focus, disabled)
 */
const Icon = forwardRef(({
  children,
  size = 20,
  className = '',
  'aria-label': ariaLabel,
  'aria-hidden': ariaHidden = false,
  title,
  role = 'img',
  focusable = false,
  onClick,
  disabled = false,
  variant = 'default',
  ...props
}, ref) => {
  // Base classes for consistent styling
  const baseClasses = 'inline-flex items-center justify-center'
  
  // Size classes
  const sizeClasses = {
    12: 'w-3 h-3',
    14: 'w-3.5 h-3.5',
    16: 'w-4 h-4',
    20: 'w-5 h-5',
    24: 'w-6 h-6',
    28: 'w-7 h-7',
    32: 'w-8 h-8',
    40: 'w-10 h-10',
    48: 'w-12 h-12'
  }
  
  // Variant classes for different contexts - WCAG AA compliant colors
  const variantClasses = {
    default: 'text-current',
    // AA compliant: 5.17:1 ratio on white, 6.98:1 on dark
    primary: 'text-blue-600 dark:text-blue-400',
    // AA compliant: 7.56:1 ratio on white, 12.04:1 on dark
    secondary: 'text-gray-600 dark:text-gray-300',
    // Better green for AA compliance: 4.87:1 ratio on white
    success: 'text-green-700 dark:text-green-400',
    // Better orange for AA compliance: 4.63:1 ratio on white
    warning: 'text-orange-600 dark:text-yellow-400',
    // AA compliant: 4.83:1 ratio on white, 6.41:1 on dark
    danger: 'text-red-600 dark:text-red-400',
    // AA compliant: 4.83:1 ratio on white, 9.96:1 on dark
    muted: 'text-gray-500 dark:text-gray-300'
  }
  
  // Interactive states
  const interactiveClasses = onClick && !disabled 
    ? 'cursor-pointer hover:opacity-80 transition-opacity duration-200' 
    : ''
  
  // Focus styles for accessibility
  const focusClasses = focusable || onClick 
    ? 'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 rounded-sm' 
    : ''
  
  // Disabled styles
  const disabledClasses = disabled 
    ? 'opacity-50 cursor-not-allowed' 
    : ''
  
  const combinedClassName = [
    baseClasses,
    sizeClasses[size] || `w-5 h-5`, // fallback to default size
    variantClasses[variant] || variantClasses.default,
    interactiveClasses,
    focusClasses,
    disabledClasses,
    className
  ].filter(Boolean).join(' ')
  
  // Accessibility attributes
  const accessibilityProps = {
    role: ariaHidden ? undefined : role,
    'aria-label': ariaHidden ? undefined : ariaLabel,
    'aria-hidden': ariaHidden,
    title: title || (ariaLabel && !ariaHidden ? ariaLabel : undefined),
    tabIndex: (focusable || onClick) && !disabled ? 0 : -1,
    focusable: focusable ? 'true' : 'false'
  }
  
  // Handle click events
  const handleClick = (event) => {
    if (disabled) {
      event.preventDefault()
      return
    }
    if (onClick) {
      onClick(event)
    }
  }
  
  // Handle keyboard navigation
  const handleKeyDown = (event) => {
    if (disabled) return
    
    if ((event.key === 'Enter' || event.key === ' ') && onClick) {
      event.preventDefault()
      onClick(event)
    }
  }
  
  return (
    <span
      ref={ref}
      className={combinedClassName}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      {...accessibilityProps}
      {...props}
    >
      {children}
    </span>
  )
})

Icon.displayName = 'Icon'

export default Icon
