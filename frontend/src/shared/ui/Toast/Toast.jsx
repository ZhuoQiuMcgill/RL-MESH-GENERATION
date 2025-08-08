import React, { useState, useEffect, useRef } from 'react'
import { cn } from '../../utils/cn'
import { Button } from '../index'
import { TOAST_TYPES } from './ToastContext'

const Toast = ({ 
  toast, 
  onRemove, 
  className,
  ...props 
}) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isLeaving, setIsLeaving] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const timeoutRef = useRef(null)
  const startTimeRef = useRef(null)
  const remainingTimeRef = useRef(null)

  // Animation and auto-dismiss logic
  useEffect(() => {
    // Show toast with animation
    const showTimer = setTimeout(() => setIsVisible(true), 10)
    
    // Set up auto-dismiss timer
    if (toast.duration > 0) {
      startTimeRef.current = Date.now()
      remainingTimeRef.current = toast.duration
      
      const setupTimeout = () => {
        timeoutRef.current = setTimeout(() => {
          handleRemove()
        }, remainingTimeRef.current)
      }
      
      setupTimeout()
    }

    return () => {
      clearTimeout(showTimer)
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [toast.duration])

  // Handle pause/resume on hover
  const handleMouseEnter = () => {
    if (toast.pauseOnHover && toast.duration > 0) {
      setIsPaused(true)
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        const elapsedTime = Date.now() - startTimeRef.current
        remainingTimeRef.current = Math.max(0, toast.duration - elapsedTime)
      }
    }
  }

  const handleMouseLeave = () => {
    if (toast.pauseOnHover && toast.duration > 0) {
      setIsPaused(false)
      startTimeRef.current = Date.now()
      
      timeoutRef.current = setTimeout(() => {
        handleRemove()
      }, remainingTimeRef.current)
    }
  }

  // Handle toast removal with exit animation
  const handleRemove = () => {
    setIsLeaving(true)
    setTimeout(() => {
      onRemove(toast.id)
    }, 300) // Match exit animation duration
  }

  // Get toast variant styles
  const getVariantStyles = (type) => {
    const variants = {
      [TOAST_TYPES.INFO]: {
        container: 'bg-blue-50 border-blue-300 text-blue-800',
        icon: (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        )
      },
      [TOAST_TYPES.SUCCESS]: {
        container: 'bg-green-50 border-green-300 text-green-800',
        icon: (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        )
      },
      [TOAST_TYPES.WARNING]: {
        container: 'bg-yellow-50 border-yellow-300 text-yellow-800',
        icon: (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        )
      },
      [TOAST_TYPES.ERROR]: {
        container: 'bg-red-50 border-red-300 text-red-800',
        icon: (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        )
      }
    }

    return variants[type] || variants[TOAST_TYPES.INFO]
  }

  const variant = getVariantStyles(toast.type)

  return (
    <div
      className={cn(
        // Base styles
        'relative flex items-start p-4 mb-3 rounded-lg border shadow-lg transition-all duration-300 ease-out max-w-md',
        
        // Variant styles
        variant.container,
        
        // Animation states
        isVisible && !isLeaving && 'translate-x-0 opacity-100',
        !isVisible && 'translate-x-full opacity-0',
        isLeaving && 'translate-x-full opacity-0 scale-95',
        
        // Hover effects
        toast.pauseOnHover && 'hover:shadow-xl',
        isPaused && 'ring-2 ring-gray-300',
        
        className
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="alert"
      aria-live="polite"
      {...props}
    >
      {/* Progress bar for timed toasts */}
      {toast.duration > 0 && !isPaused && (
        <div 
          className="absolute bottom-0 left-0 h-1 bg-current opacity-30 transition-all ease-linear"
          style={{
            animation: `toast-progress ${toast.duration}ms linear`,
            animationPlayState: isPaused ? 'paused' : 'running'
          }}
        />
      )}

      {/* Toast content */}
      <div className="flex items-start w-full">
        {/* Icon */}
        {(toast.icon !== false) && (
          <div className="flex-shrink-0 mr-3">
            {toast.icon || variant.icon}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Title */}
          {toast.title && (
            <div className="font-semibold text-sm mb-1">
              {toast.title}
            </div>
          )}

          {/* Message */}
          <div className="text-sm">
            {typeof toast.message === 'string' ? (
              <p>{toast.message}</p>
            ) : (
              toast.message
            )}
          </div>

          {/* Description */}
          {toast.description && (
            <div className="text-xs mt-1 opacity-80">
              {toast.description}
            </div>
          )}

          {/* Action button */}
          {toast.action && (
            <div className="mt-3">
              <Button
                size="xs"
                variant="outline"
                onClick={() => {
                  toast.action.onClick?.()
                  if (toast.action.closeOnClick !== false) {
                    handleRemove()
                  }
                }}
                className="bg-transparent border-current text-current hover:bg-current/10"
              >
                {toast.action.label}
              </Button>
            </div>
          )}
        </div>

        {/* Close button */}
        {toast.showCloseButton && (
          <button
            type="button"
            onClick={handleRemove}
            className="flex-shrink-0 ml-3 -mr-1 -mt-1 p-1 rounded-md hover:bg-black/10 transition-colors"
            aria-label="Close notification"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        )}
      </div>

      {/* CSS for progress bar animation */}
      <style jsx>{`
        @keyframes toast-progress {
          from {
            width: 100%;
          }
          to {
            width: 0%;
          }
        }
      `}</style>
    </div>
  )
}

Toast.displayName = 'Toast'

export default Toast
