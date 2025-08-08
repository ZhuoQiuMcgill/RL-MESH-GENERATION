import React from 'react'
import { cn } from '../utils/cn'

const Skeleton = React.forwardRef(({ 
  className,
  variant = 'default',
  size = 'default',
  width,
  height,
  rounded = true,
  animation = 'pulse',
  ...props 
}, ref) => {
  const baseClasses = 'bg-bg-secondary'
  
  const variants = {
    default: 'bg-bg-secondary',
    light: 'bg-gray-200',
    dark: 'bg-gray-700'
  }

  const sizes = {
    sm: 'h-3',
    default: 'h-4',
    lg: 'h-5',
    xl: 'h-6'
  }

  const animations = {
    pulse: 'animate-pulse',
    wave: 'animate-pulse',
    none: ''
  }

  const roundedClasses = {
    true: 'rounded',
    false: '',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
    full: 'rounded-full'
  }

  const classes = cn(
    baseClasses,
    variants[variant],
    !width && !height && sizes[size],
    animations[animation],
    rounded && (typeof rounded === 'boolean' ? roundedClasses.true : roundedClasses[rounded]),
    className
  )

  const style = {
    ...(width && { width }),
    ...(height && { height }),
  }

  return (
    <div
      ref={ref}
      className={classes}
      style={style}
      {...props}
    />
  )
})

// Common skeleton patterns for convenience
const SkeletonText = ({ lines = 1, className, ...props }) => {
  if (lines === 1) {
    return <Skeleton className={cn('w-full h-4', className)} {...props} />
  }

  return (
    <div className="space-y-2">
      {Array.from({ length: lines - 1 }, (_, i) => (
        <Skeleton key={i} className="w-full h-4" {...props} />
      ))}
      <Skeleton className={cn('w-3/4 h-4', className)} {...props} />
    </div>
  )
}

const SkeletonCircle = ({ size = 'default', className, ...props }) => {
  const sizes = {
    sm: 'w-8 h-8',
    default: 'w-10 h-10',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  }

  return (
    <Skeleton
      className={cn(sizes[size], 'rounded-full', className)}
      {...props}
    />
  )
}

const SkeletonCard = ({ className, ...props }) => {
  return (
    <div className={cn('space-y-3', className)}>
      <SkeletonCircle className="mb-4" />
      <div className="space-y-2">
        <Skeleton className="w-full h-4" {...props} />
        <Skeleton className="w-full h-4" {...props} />
        <Skeleton className="w-3/4 h-4" {...props} />
      </div>
    </div>
  )
}

Skeleton.displayName = 'Skeleton'
SkeletonText.displayName = 'SkeletonText'
SkeletonCircle.displayName = 'SkeletonCircle'
SkeletonCard.displayName = 'SkeletonCard'

export default Skeleton
export { SkeletonText, SkeletonCircle, SkeletonCard }
