import React from 'react'
import { cn } from '../utils/cn'

const Table = React.forwardRef(({ 
  className,
  variant = 'default',
  size = 'default',
  children,
  ...props 
}, ref) => {
  const baseClasses = 'w-full border-collapse'
  
  const variants = {
    default: 'border-border-custom',
    striped: 'border-border-custom',
    bordered: 'border border-border-custom'
  }
  
  const classes = cn(
    baseClasses,
    variants[variant],
    className
  )
  
  return (
    <div className="overflow-x-auto">
      <table
        ref={ref}
        className={classes}
        {...props}
      >
        {children}
      </table>
    </div>
  )
})

const TableHeader = React.forwardRef(({ 
  className,
  children,
  ...props 
}, ref) => {
  return (
    <thead
      ref={ref}
      className={cn('bg-bg-secondary', className)}
      {...props}
    >
      {children}
    </thead>
  )
})

const TableBody = React.forwardRef(({ 
  className,
  children,
  ...props 
}, ref) => {
  return (
    <tbody
      ref={ref}
      className={cn('divide-y divide-border-custom', className)}
      {...props}
    >
      {children}
    </tbody>
  )
})

const TableRow = React.forwardRef(({ 
  className,
  variant = 'default',
  children,
  ...props 
}, ref) => {
  const variants = {
    default: 'hover:bg-card-hover transition-colors',
    striped: 'odd:bg-bg-secondary hover:bg-card-hover transition-colors'
  }
  
  return (
    <tr
      ref={ref}
      className={cn(variants[variant], className)}
      {...props}
    >
      {children}
    </tr>
  )
})

const TableHead = React.forwardRef(({ 
  className,
  size = 'default',
  children,
  ...props 
}, ref) => {
  const sizes = {
    sm: 'px-2 py-2 text-xs',
    default: 'px-4 py-3 text-sm',
    lg: 'px-6 py-4 text-base'
  }
  
  return (
    <th
      ref={ref}
      className={cn(
        'text-left font-medium text-text-primary',
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </th>
  )
})

const TableCell = React.forwardRef(({ 
  className,
  size = 'default',
  children,
  ...props 
}, ref) => {
  const sizes = {
    sm: 'px-2 py-2 text-xs',
    default: 'px-4 py-3 text-sm',
    lg: 'px-6 py-4 text-base'
  }
  
  return (
    <td
      ref={ref}
      className={cn(
        'text-text-secondary',
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </td>
  )
})

Table.displayName = 'Table'
TableHeader.displayName = 'TableHeader'
TableBody.displayName = 'TableBody'
TableRow.displayName = 'TableRow'
TableHead.displayName = 'TableHead'
TableCell.displayName = 'TableCell'

export default Table
export { TableHeader, TableBody, TableRow, TableHead, TableCell }
