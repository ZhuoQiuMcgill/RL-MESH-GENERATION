import React from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../utils/cn'
import Toast from './Toast'
import { useToast, TOAST_POSITIONS } from './ToastContext'

const ToastContainer = ({ 
  className,
  position,
  ...props 
}) => {
  const { toasts, config, removeToast } = useToast()
  const finalPosition = position || config.position

  // Don't render anything if no toasts
  if (toasts.length === 0) {
    return null
  }

  // Get position-specific classes
  const getPositionClasses = (pos) => {
    const positions = {
      [TOAST_POSITIONS.TOP_LEFT]: 'top-4 left-4',
      [TOAST_POSITIONS.TOP_CENTER]: 'top-4 left-1/2 transform -translate-x-1/2',
      [TOAST_POSITIONS.TOP_RIGHT]: 'top-4 right-4',
      [TOAST_POSITIONS.BOTTOM_LEFT]: 'bottom-4 left-4',
      [TOAST_POSITIONS.BOTTOM_CENTER]: 'bottom-4 left-1/2 transform -translate-x-1/2',
      [TOAST_POSITIONS.BOTTOM_RIGHT]: 'bottom-4 right-4'
    }
    
    return positions[pos] || positions[TOAST_POSITIONS.TOP_RIGHT]
  }

  // Create the toast container
  const toastContainer = (
    <div
      className={cn(
        // Base positioning
        'fixed z-50 pointer-events-none',
        'flex flex-col',
        
        // Position-specific classes
        getPositionClasses(finalPosition),
        
        // Direction based on position (reverse for bottom positions)
        finalPosition.includes('bottom') ? 'flex-col-reverse' : 'flex-col',
        
        className
      )}
      role="region"
      aria-label="Toast notifications"
      {...props}
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="pointer-events-auto"
        >
          <Toast
            toast={toast}
            onRemove={removeToast}
          />
        </div>
      ))}
    </div>
  )

  // Render toasts in a portal to avoid z-index issues
  if (typeof document !== 'undefined') {
    return createPortal(toastContainer, document.body)
  }

  return null
}

ToastContainer.displayName = 'ToastContainer'

export default ToastContainer
