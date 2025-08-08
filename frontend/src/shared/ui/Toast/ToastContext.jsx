import React, { createContext, useContext, useReducer, useCallback } from 'react'

// Toast types
export const TOAST_TYPES = {
  INFO: 'info',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error'
}

// Toast positions
export const TOAST_POSITIONS = {
  TOP_LEFT: 'top-left',
  TOP_CENTER: 'top-center',
  TOP_RIGHT: 'top-right',
  BOTTOM_LEFT: 'bottom-left',
  BOTTOM_CENTER: 'bottom-center',
  BOTTOM_RIGHT: 'bottom-right'
}

// Initial state
const initialState = {
  toasts: []
}

// Action types
const ACTIONS = {
  ADD_TOAST: 'ADD_TOAST',
  REMOVE_TOAST: 'REMOVE_TOAST',
  CLEAR_TOASTS: 'CLEAR_TOASTS',
  UPDATE_TOAST: 'UPDATE_TOAST'
}

// Reducer
const toastReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.ADD_TOAST:
      return {
        ...state,
        toasts: [...state.toasts, action.payload]
      }
    
    case ACTIONS.REMOVE_TOAST:
      return {
        ...state,
        toasts: state.toasts.filter(toast => toast.id !== action.payload)
      }
    
    case ACTIONS.CLEAR_TOASTS:
      return {
        ...state,
        toasts: []
      }
    
    case ACTIONS.UPDATE_TOAST:
      return {
        ...state,
        toasts: state.toasts.map(toast =>
          toast.id === action.payload.id
            ? { ...toast, ...action.payload.updates }
            : toast
        )
      }
    
    default:
      return state
  }
}

// Create context
const ToastContext = createContext(null)

// Default configuration
const defaultConfig = {
  position: TOAST_POSITIONS.TOP_RIGHT,
  duration: 5000,
  maxToasts: 5,
  showCloseButton: true,
  pauseOnHover: true
}

// Provider component
export const ToastProvider = ({ 
  children, 
  config = {}, 
  position = TOAST_POSITIONS.TOP_RIGHT 
}) => {
  const [state, dispatch] = useReducer(toastReducer, initialState)
  const finalConfig = { ...defaultConfig, ...config, position }

  // Generate unique ID
  const generateId = useCallback(() => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2)
  }, [])

  // Add toast
  const addToast = useCallback((toast) => {
    const id = generateId()
    const newToast = {
      id,
      type: TOAST_TYPES.INFO,
      duration: finalConfig.duration,
      showCloseButton: finalConfig.showCloseButton,
      pauseOnHover: finalConfig.pauseOnHover,
      createdAt: Date.now(),
      ...toast
    }

    dispatch({ type: ACTIONS.ADD_TOAST, payload: newToast })

    // Auto-remove toast after duration (if duration > 0)
    if (newToast.duration > 0) {
      setTimeout(() => {
        removeToast(id)
      }, newToast.duration)
    }

    // Limit max toasts
    if (state.toasts.length >= finalConfig.maxToasts) {
      // Remove oldest toast
      const oldestToast = state.toasts[0]
      if (oldestToast) {
        removeToast(oldestToast.id)
      }
    }

    return id
  }, [finalConfig, state.toasts.length])

  // Remove toast
  const removeToast = useCallback((id) => {
    dispatch({ type: ACTIONS.REMOVE_TOAST, payload: id })
  }, [])

  // Clear all toasts
  const clearToasts = useCallback(() => {
    dispatch({ type: ACTIONS.CLEAR_TOASTS })
  }, [])

  // Update toast
  const updateToast = useCallback((id, updates) => {
    dispatch({ type: ACTIONS.UPDATE_TOAST, payload: { id, updates } })
  }, [])

  // Convenience methods for different toast types
  const toast = useCallback((message, options = {}) => {
    return addToast({ ...options, message, type: TOAST_TYPES.INFO })
  }, [addToast])

  const success = useCallback((message, options = {}) => {
    return addToast({ ...options, message, type: TOAST_TYPES.SUCCESS })
  }, [addToast])

  const error = useCallback((message, options = {}) => {
    return addToast({ 
      ...options, 
      message, 
      type: TOAST_TYPES.ERROR,
      duration: options.duration ?? 7000 // Longer duration for errors
    })
  }, [addToast])

  const warning = useCallback((message, options = {}) => {
    return addToast({ ...options, message, type: TOAST_TYPES.WARNING })
  }, [addToast])

  const info = useCallback((message, options = {}) => {
    return addToast({ ...options, message, type: TOAST_TYPES.INFO })
  }, [addToast])

  // Promise-based toast for async operations
  const promise = useCallback(async (promise, messages = {}, options = {}) => {
    const loadingToastId = addToast({
      ...options,
      message: messages.loading || 'Loading...',
      type: TOAST_TYPES.INFO,
      duration: 0, // Don't auto-dismiss loading toast
      showCloseButton: false
    })

    try {
      const result = await promise
      
      // Remove loading toast
      removeToast(loadingToastId)
      
      // Show success toast
      if (messages.success) {
        success(messages.success, options)
      }
      
      return result
    } catch (error) {
      // Remove loading toast
      removeToast(loadingToastId)
      
      // Show error toast
      const errorMessage = messages.error || error.message || 'An error occurred'
      error(errorMessage, {
        ...options,
        action: messages.retry ? {
          label: 'Retry',
          onClick: messages.retry
        } : undefined
      })
      
      throw error
    }
  }, [addToast, removeToast, success, error])

  const contextValue = {
    // State
    toasts: state.toasts,
    config: finalConfig,
    
    // Actions
    addToast,
    removeToast,
    clearToasts,
    updateToast,
    
    // Convenience methods
    toast,
    success,
    error,
    warning,
    info,
    promise
  }

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
    </ToastContext.Provider>
  )
}

// Custom hook to use toast context
export const useToast = () => {
  const context = useContext(ToastContext)
  
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  
  return context
}

export default ToastContext
