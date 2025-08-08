/**
 * Enhanced API Client with Toast Integration
 * 
 * This module extends the base API client with toast notifications
 * and improved error handling including retry affordances.
 */

import { apiClientInstance, withRetry } from './client'

// Enhanced API client that integrates with toast system
export class EnhancedApiClient {
  constructor(apiClient, toastContext) {
    this.apiClient = apiClient
    this.toast = toastContext
    this.retryAttempts = new Map() // Track retry attempts per operation
  }

  /**
   * Set toast context (used when context is available)
   */
  setToastContext(toastContext) {
    this.toast = toastContext
  }

  /**
   * Enhanced error handling with toast notifications
   */
  handleApiError(error, operation = 'operation', options = {}) {
    const {
      showToast = true,
      enableRetry = true,
      retryAction = null,
      customMessage = null
    } = options

    if (!showToast || !this.toast) {
      throw error
    }

    // Determine error message
    let errorMessage = customMessage
    if (!errorMessage) {
      if (error.message.includes('timeout')) {
        errorMessage = `${operation} timed out. Please check your network connection.`
      } else if (error.message.includes('Failed to fetch')) {
        errorMessage = `Unable to connect to server. Please check if the server is running.`
      } else if (error.message.includes('HTTP error! status: 5')) {
        errorMessage = `Server error occurred during ${operation}. Please try again.`
      } else if (error.message.includes('HTTP error! status: 4')) {
        errorMessage = `Request failed: ${error.message}`
      } else {
        errorMessage = `${operation} failed: ${error.message}`
      }
    }

    // Show error toast with optional retry button
    const toastOptions = {
      title: 'Error',
      description: enableRetry ? 'Click retry to try again' : undefined,
      action: enableRetry && retryAction ? {
        label: 'Retry',
        onClick: () => {
          const currentAttempts = this.retryAttempts.get(operation) || 0
          if (currentAttempts < 3) {
            this.retryAttempts.set(operation, currentAttempts + 1)
            retryAction()
          } else {
            this.toast.warning('Maximum retry attempts reached', {
              title: 'Retry Limit'
            })
          }
        }
      } : undefined
    }

    this.toast.error(errorMessage, toastOptions)
    throw error
  }

  /**
   * Execute API call with enhanced error handling
   */
  async executeWithToast(operation, apiCall, options = {}) {
    const {
      loadingMessage = `Loading...`,
      successMessage = null,
      showLoading = true,
      showSuccess = false,
      enableRetry = true,
      retryCount = 1
    } = options

    try {
      // Show loading toast if enabled
      if (showLoading && this.toast) {
        return await this.toast.promise(
          withRetry(apiCall, retryCount),
          {
            loading: loadingMessage,
            success: successMessage,
            error: null, // We'll handle errors ourselves
            retry: enableRetry ? () => this.executeWithToast(operation, apiCall, options) : null
          }
        )
      } else {
        // Execute without loading toast
        const result = await withRetry(apiCall, retryCount)
        
        // Show success toast if enabled
        if (showSuccess && successMessage && this.toast) {
          this.toast.success(successMessage)
        }
        
        return result
      }
    } catch (error) {
      this.handleApiError(error, operation, {
        ...options,
        retryAction: () => this.executeWithToast(operation, apiCall, options)
      })
    }
  }

  // Enhanced API methods with toast integration

  async getTrainingStatus(options = {}) {
    return this.executeWithToast(
      'fetching training status',
      () => this.apiClient.getTrainingStatus(),
      {
        showLoading: false,
        ...options
      }
    )
  }

  async startTraining(config, options = {}) {
    return this.executeWithToast(
      'starting training',
      () => this.apiClient.startTraining(config),
      {
        loadingMessage: 'Starting training session...',
        successMessage: 'Training started successfully!',
        showLoading: true,
        showSuccess: true,
        ...options
      }
    )
  }

  async stopTraining(options = {}) {
    return this.executeWithToast(
      'stopping training',
      () => this.apiClient.stopTraining(),
      {
        loadingMessage: 'Stopping training session...',
        successMessage: 'Training stopped successfully',
        showLoading: true,
        showSuccess: true,
        ...options
      }
    )
  }

  async getMeshList(subfolder = 'mesh', options = {}) {
    return this.executeWithToast(
      'loading mesh list',
      () => this.apiClient.getMeshList(subfolder),
      {
        loadingMessage: 'Loading available meshes...',
        ...options
      }
    )
  }

  async getMeshInfo(meshName, subfolder = 'mesh', options = {}) {
    return this.executeWithToast(
      'loading mesh information',
      () => this.apiClient.getMeshInfo(meshName, subfolder),
      {
        loadingMessage: `Loading ${meshName} information...`,
        ...options
      }
    )
  }

  async getMeshBoundary(meshName, subfolder = 'mesh', options = {}) {
    return this.executeWithToast(
      'loading mesh boundary',
      () => this.apiClient.getMeshBoundary(meshName, subfolder),
      {
        loadingMessage: `Loading ${meshName} boundary data...`,
        ...options
      }
    )
  }

  async getMeshData(meshName, options = {}) {
    return this.executeWithToast(
      'loading mesh data',
      () => this.apiClient.getMeshData(meshName),
      {
        loadingMessage: `Loading ${meshName} mesh data...`,
        ...options
      }
    )
  }

  async getCheckpointList(options = {}) {
    return this.executeWithToast(
      'loading checkpoint list',
      () => this.apiClient.getCheckpointList(),
      {
        loadingMessage: 'Loading available checkpoints...',
        ...options
      }
    )
  }

  async deleteCheckpoint(checkpointName, options = {}) {
    return this.executeWithToast(
      'deleting checkpoint',
      () => this.apiClient.deleteCheckpoint(checkpointName),
      {
        loadingMessage: `Deleting checkpoint ${checkpointName}...`,
        successMessage: `Checkpoint ${checkpointName} deleted successfully`,
        showLoading: true,
        showSuccess: true,
        ...options
      }
    )
  }

  async executeAction(actionData, options = {}) {
    return this.executeWithToast(
      'executing action',
      () => this.apiClient.executeAction(actionData),
      {
        loadingMessage: 'Executing action...',
        successMessage: 'Action executed successfully',
        showLoading: true,
        showSuccess: true,
        ...options
      }
    )
  }

  async validateAction(actionType, actionData, options = {}) {
    return this.executeWithToast(
      'validating action',
      () => this.apiClient.validateAction(actionType, actionData),
      {
        loadingMessage: 'Validating action...',
        ...options
      }
    )
  }

  // Health check methods
  async checkConnection(options = {}) {
    return this.executeWithToast(
      'checking connection',
      () => this.apiClient.checkConnection(),
      {
        showLoading: false,
        enableRetry: true,
        retryCount: 2,
        ...options
      }
    )
  }

  async checkMeshHealth(options = {}) {
    return this.executeWithToast(
      'checking mesh API health',
      () => this.apiClient.checkMeshHealth(),
      {
        showLoading: false,
        ...options
      }
    )
  }

  async checkTrainingHealth(options = {}) {
    return this.executeWithToast(
      'checking training API health',
      () => this.apiClient.checkTrainingHealth(),
      {
        showLoading: false,
        ...options
      }
    )
  }

  // Direct access to base client for cases where toast integration is not needed
  get baseClient() {
    return this.apiClient
  }

  // Reset retry attempts (useful for cleanup)
  resetRetryAttempts() {
    this.retryAttempts.clear()
  }
}

// Create and export enhanced client instance
export const enhancedApiClient = new EnhancedApiClient(apiClientInstance, null)

// Export for use in contexts/hooks
export { enhancedApiClient as default }
