/**
 * Hook for using the enhanced API client with toast integration
 */

import { useEffect } from 'react'
import { useToast } from '../../shared/ui'
import { enhancedApiClient } from '../api/enhancedClient'

/**
 * Custom hook that provides the enhanced API client with toast context
 * @returns {Object} Enhanced API client with toast integration
 */
export const useApiWithToast = () => {
  const toastContext = useToast()

  // Set the toast context on the enhanced API client
  useEffect(() => {
    enhancedApiClient.setToastContext(toastContext)
  }, [toastContext])

  return enhancedApiClient
}

/**
 * Hook for API operations with toast notifications and error handling
 * @param {Object} options - Configuration options
 * @returns {Object} API utilities with toast integration
 */
export const useApiOperations = (options = {}) => {
  const api = useApiWithToast()
  const toast = useToast()

  const executeOperation = async (operation, operationName, toastOptions = {}) => {
    try {
      const result = await api.executeWithToast(
        operationName,
        operation,
        {
          ...options,
          ...toastOptions
        }
      )
      return result
    } catch (error) {
      // Error is already handled by the enhanced client
      throw error
    }
  }

  const showSuccessToast = (message, options = {}) => {
    toast.success(message, options)
  }

  const showErrorToast = (message, options = {}) => {
    toast.error(message, options)
  }

  const showWarningToast = (message, options = {}) => {
    toast.warning(message, options)
  }

  const showInfoToast = (message, options = {}) => {
    toast.info(message, options)
  }

  return {
    api,
    executeOperation,
    showSuccessToast,
    showErrorToast,
    showWarningToast,
    showInfoToast,
    toast
  }
}

export default useApiWithToast
