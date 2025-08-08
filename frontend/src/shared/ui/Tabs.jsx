import React, { useState, createContext, useContext } from 'react'
import { cn } from '../utils/cn'

const TabsContext = createContext()

const Tabs = ({ 
  defaultValue,
  value,
  onValueChange,
  children,
  className,
  variant = 'default',
  size = 'default',
  ...props 
}) => {
  const [selectedTab, setSelectedTab] = useState(defaultValue)
  
  const isControlled = value !== undefined
  const currentValue = isControlled ? value : selectedTab
  
  const handleValueChange = (newValue) => {
    if (!isControlled) {
      setSelectedTab(newValue)
    }
    onValueChange?.(newValue)
  }
  
  return (
    <TabsContext.Provider value={{
      value: currentValue,
      onValueChange: handleValueChange,
      variant,
      size
    }}>
      <div className={cn('w-full', className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  )
}

const TabsList = React.forwardRef(({ 
  className,
  children,
  ...props 
}, ref) => {
  const { variant, size } = useContext(TabsContext)
  
  const variants = {
    default: 'bg-bg-secondary border border-border-custom rounded-lg p-1',
    underline: 'border-b border-border-custom',
    pills: 'bg-transparent gap-2'
  }
  
  const sizes = {
    sm: 'text-sm',
    default: 'text-base',
    lg: 'text-lg'
  }
  
  return (
    <div
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})

const TabsTrigger = React.forwardRef(({ 
  className,
  value,
  disabled = false,
  children,
  ...props 
}, ref) => {
  const { value: selectedValue, onValueChange, variant, size } = useContext(TabsContext)
  
  const isSelected = selectedValue === value
  
  const baseClasses = 'inline-flex items-center justify-center whitespace-nowrap rounded-md font-medium transition-all focus:outline-none focus:ring-2 focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    default: cn(
      'px-3 py-1.5',
      isSelected 
        ? 'bg-card text-text-primary shadow-sm' 
        : 'text-text-secondary hover:text-text-primary hover:bg-card-hover'
    ),
    underline: cn(
      'px-4 py-2 border-b-2 -mb-px',
      isSelected 
        ? 'border-accent text-accent' 
        : 'border-transparent text-text-secondary hover:text-text-primary hover:border-border-custom'
    ),
    pills: cn(
      'px-4 py-2 rounded-full',
      isSelected 
        ? 'bg-accent text-white' 
        : 'bg-bg-secondary text-text-secondary hover:bg-card hover:text-text-primary'
    )
  }
  
  const sizes = {
    sm: 'text-sm px-2 py-1',
    default: '',
    lg: 'text-lg px-6 py-3'
  }
  
  return (
    <button
      ref={ref}
      type="button"
      disabled={disabled}
      onClick={() => onValueChange(value)}
      className={cn(
        baseClasses,
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
})

const TabsContent = React.forwardRef(({ 
  className,
  value,
  children,
  ...props 
}, ref) => {
  const { value: selectedValue } = useContext(TabsContext)
  
  if (selectedValue !== value) return null
  
  return (
    <div
      ref={ref}
      className={cn('mt-4', className)}
      {...props}
    >
      {children}
    </div>
  )
})

Tabs.displayName = 'Tabs'
TabsList.displayName = 'TabsList'
TabsTrigger.displayName = 'TabsTrigger'
TabsContent.displayName = 'TabsContent'

export default Tabs
export { TabsList, TabsTrigger, TabsContent }
