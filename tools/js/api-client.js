/**
 * API Client Module
 * Responsible for all communication with backend API
 */

import {CONSTANTS, delay} from './utils.js';

/**
 * API Error Handling Decorator
 * @param {Function} apiMethod - API method
 * @returns {Function} Wrapped method
 */
export function withErrorHandling(apiMethod) {
    return async function (...args) {
        try {
            return await apiMethod.apply(this, args);
        } catch (error) {
            console.error(`API call failed:`, error);

            // Return different error messages based on error type
            if (error.message.includes('timeout')) {
                throw new Error('Request timed out, please check network connection');
            } else if (error.message.includes('Failed to fetch')) {
                throw new Error('Network connection failed, please check server status');
            } else {
                throw error;
            }
        }
    };
}

/**
 * API Call with Retry Mechanism
 * @param {Function} apiCall - API call function
 * @param {number} maxRetries - Maximum retry attempts
 * @param {number} retryDelay - Retry delay time (milliseconds)
 * @returns {Promise<any>} API response
 */
export async function withRetry(apiCall, maxRetries = 1, retryDelay = 3000) {
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await apiCall();
        } catch (error) {
            lastError = error;

            if (attempt < maxRetries) {
                console.warn(`API call failed, retrying (${attempt + 1}/${maxRetries}):`, error.message);
                await delay(retryDelay * Math.pow(2, attempt)); // Exponential backoff
            }
        }
    }

    throw lastError;
}

export class ApiClient {
    constructor() {
        this.baseUrl = CONSTANTS.API_BASE_URL;
    }

    /**
     * Generic API Request Method
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @param {number} customTimeout - Custom timeout (milliseconds), uses default if not provided
     * @returns {Promise<any>} API response
     */
    async request(endpoint, options = {}, customTimeout = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const requestOptions = {...defaultOptions, ...options};

        // Determine timeout duration
        const timeout = customTimeout || CONSTANTS.CONNECTION_TIMEOUT;

        try {
            // Create timeout controller
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(url, {
                ...requestOptions,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error(`Request timed out (${timeout / 1000}s), please check network connection`);
            }
            throw error;
        }
    }

    /**
     * Check Backend Connection Status
     * @returns {Promise<boolean>} Connection status
     */
    async checkConnection() {
        try {
            await this.request('/training/health');
            return true;
        } catch (error) {
            console.error('Backend connection failed:', error);
            return false;
        }
    }

    /**
     * Get Training Status
     * @returns {Promise<Object>} Training status information
     */
    async getTrainingStatus() {
        return await this.request('/training/status');
    }

    /**
     * Start Training
     * @param {Object} config - Training configuration
     * @returns {Promise<Object>} Start result
     */
    async startTraining(config) {
        console.log('=== API Client Sending Request ===');
        console.log('Request Config:', config);
        console.log('Request Body:', JSON.stringify(config));
        console.log('==================================');

        return await this.request('/training/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * Stop Training - Uses longer timeout
     * @returns {Promise<Object>} Stop result
     */
    async stopTraining() {
        return await this.request('/training/stop', {
            method: 'POST'
        }, CONSTANTS.TRAINING_STOP_TIMEOUT); // Use 30 second timeout
    }

    /**
     * Get Available Mesh List
     * @param {string} subfolder - Subfolder name, defaults to 'mesh'
     * @returns {Promise<Object>} Mesh list
     */
    async getMeshList(subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/list?${params}`);
    }

    /**
     * Get Specified Mesh Information
     * @param {string} meshName - Mesh name
     * @param {string} subfolder - Subfolder name, defaults to 'mesh'
     * @returns {Promise<Object>} Mesh information
     */
    async getMeshInfo(meshName, subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/info/${meshName}?${params}`);
    }

    /**
     * Get Specified Mesh Boundary Data (New)
     * @param {string} meshName - Mesh name
     * @param {string} subfolder - Subfolder name, defaults to 'mesh'
     * @returns {Promise<Object>} Mesh boundary data
     */
    async getMeshBoundary(meshName, subfolder = 'mesh') {
        const params = new URLSearchParams({subfolder});
        return await this.request(`/mesh/boundary/${meshName}?${params}`);
    }

    /**
     * Check Mesh API Health Status
     * @returns {Promise<Object>} Health status
     */
    async checkMeshHealth() {
        return await this.request('/mesh/health');
    }

    /**
     * Check Training API Health Status
     * @returns {Promise<Object>} Health status
     */
    async checkTrainingHealth() {
        return await this.request('/training/health');
    }

    // ========== Checkpoint Related APIs ==========

    /**
     * Get Available Checkpoint List
     * @returns {Promise<Object>} Checkpoint list
     */
    async getCheckpointList() {
        return await this.request('/checkpoint/list');
    }

    /**
     * Get Specified Checkpoint Information
     * @param {string} checkpointName - Checkpoint name
     * @returns {Promise<Object>} Checkpoint information
     */
    async getCheckpointInfo(checkpointName) {
        return await this.request(`/checkpoint/info/${checkpointName}`);
    }

    /**
     * Validate Checkpoint Validity
     * @param {string} checkpointName - Checkpoint name
     * @returns {Promise<Object>} Validation result
     */
    async validateCheckpoint(checkpointName) {
        return await this.request(`/checkpoint/validate/${checkpointName}`);
    }

    /**
     * Delete Specified Checkpoint
     * @param {string} checkpointName - Checkpoint name
     * @returns {Promise<Object>} Deletion result
     */
    async deleteCheckpoint(checkpointName) {
        return await this.request(`/checkpoint/delete/${checkpointName}`, {
            method: 'DELETE'
        });
    }

    /**
     * Copy Checkpoint from Historical Training Directory
     * @param {string} trainingId - Training session ID
     * @param {string} checkpointName - Target checkpoint name (optional)
     * @returns {Promise<Object>} Copy result
     */
    async copyCheckpointFromHistory(trainingId, checkpointName = null) {
        const body = {training_id: trainingId};
        if (checkpointName) {
            body.checkpoint_name = checkpointName;
        }

        return await this.request('/checkpoint/copy', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    /**
     * Check Checkpoint API Health Status
     * @returns {Promise<Object>} Health status
     */
    async checkCheckpointHealth() {
        return await this.request('/checkpoint/health');
    }

    // ========== Action Related APIs ==========

    /**
     * Find reference point for a mesh
     * @param {string} meshName - Mesh name
     * @returns {Promise<Object>} Reference point information
     */
    async findReferencePoint(meshName) {
        return await this.request(`/action/find-ref-point/${meshName}`);
    }

    /**
     * Execute and validate an action
     * @param {Object} actionData - Action execution data
     * @returns {Promise<Object>} Execution result
     */
    async executeAction(actionData) {
        return await this.request('/action/execute', {
            method: 'POST',
            body: JSON.stringify(actionData)
        });
    }

    /**
     * Validate a specific action type
     * @param {string} actionType - Action type
     * @param {Object} actionData - Action validation data
     * @returns {Promise<Object>} Validation result
     */
    async validateAction(actionType, actionData) {
        return await this.request(`/action/validate/${actionType}`, {
            method: 'POST',
            body: JSON.stringify(actionData)
        });
    }

    /**
     * Get information about available actions
     * @returns {Promise<Object>} Action information
     */
    async getActionInfo() {
        return await this.request('/action/info');
    }

    /**
     * Check Action API Health Status
     * @returns {Promise<Object>} Health status
     */
    async checkActionHealth() {
        return await this.request('/action/health');
    }
}