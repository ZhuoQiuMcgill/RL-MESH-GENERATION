/**
 * UI Controller Module
 * Handles all UI updates and user interaction logic
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, getTimestamp, getLogStyle, safeGetElement} from './utils.js';

export class UIController {
    constructor() {
        this.elements = this.initializeElements();
        this.isTraining = false;
        this.meshData = null;
        this.boundaryData = null;
        this.refPointInfo = null;
        this.checkpoints = []; // Store checkpoint list
        
        // Time estimation tracking
        this.progressHistory = [];
        this.maxTimesteps = null;
        this.trainingStartTime = null;
    }

    /**
     * Initialize DOM element references
     * @returns {Object} Element reference object
     */
    initializeElements() {
        const elementIds = [
            'mesh-select', 'mesh-info', 'mesh-vertices', 'mesh-size', 
            'start-btn', 'stop-btn', 'refresh-btn', 'clear-log-btn', 
            'max-timesteps', 'max-steps', 'update-interval', 'description', 
            'current-episode', 'total-steps', 'avg-reward', 'buffer-size', 
            'episode-reward', 'episode-length', 'ref-point', 'avg-element-quality', 'click-coordinates', 
            'boundary-vertices', 'log-container', 'loading-overlay', 'estimated-finish-time',
            // Checkpoint-related elements
            'checkpoint-mode', 'checkpoint-select', 'checkpoint-info', 'checkpoint-details',
            // Training metrics elements
            'actor-loss', 'critic-loss', 'alpha-value',
            // Evaluation elements
            'last-eval-reward', 'best-eval-reward', 'eval-frequency',
            // Status bar elements (simplified)
            'compact-status-text', 'status-indicator-dot'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = safeGetElement(id);
        });

        return elements;
    }

    /**
     * Update status indicator
     * @param {string} status - Status value
     */
    updateStatusIndicator(status) {
        const compactIndicator = this.elements['status-indicator-dot'];
        const compactText = this.elements['compact-status-text'];

        const statusConfig = {
            [STATUS.RUNNING]: {class: 'status-running', text: 'Training...'},
            [STATUS.STOPPED]: {class: 'status-stopped', text: 'Stopped'},
            [STATUS.COMPLETED]: {class: 'status-success', text: 'Completed'},
            [STATUS.STOPPING]: {class: 'status-loading', text: 'Stopping...'},
            [STATUS.ERROR]: {class: 'status-stopped', text: 'Error'},
            [STATUS.IDLE]: {class: 'status-idle', text: 'Ready'}
        };

        const config = statusConfig[status] || statusConfig[STATUS.IDLE];
        
        if (compactIndicator) {
            compactIndicator.className = 'status-indicator-dot ' + config.class;
        }
        if (compactText) {
            compactText.textContent = config.text;
        }
    }

    /**
     * Update training statistics
     * @param {Object} stats - Statistics data
     */
    updateTrainingStats(stats) {
        if (!stats) return;

        // Update comprehensive training metrics display
        this.updateElement('current-episode', stats.episode || 0);
        this.updateElement('total-steps', stats.total_steps || 0);
        this.updateElement('avg-reward', formatNumber(stats.average_reward));
        this.updateElement('buffer-size', stats.buffer_size || 0);
        this.updateElement('episode-reward', formatNumber(stats.episode_reward));
        this.updateElement('episode-length', stats.episode_length || 0);
        this.updateElement('boundary-vertices', stats.boundary_vertices || 0);
        
        this.updateElement('actor-loss', formatNumber(stats.recent_actor_loss, 4));
        this.updateElement('critic-loss', formatNumber(stats.recent_critic_loss, 4));
        this.updateElement('alpha-value', formatNumber(stats.current_alpha, 4));

        // Update evaluation information
        this.updateElement('last-eval-reward', formatNumber(stats.last_eval_reward, 3));
        this.updateElement('best-eval-reward', formatNumber(stats.best_eval_reward, 3));
        this.updateElement('eval-frequency', `${stats.evaluation_frequency || 'N/A'} episodes`);

        // Update reference point information
        if (stats.reference_point_info && stats.reference_point_info.ref_vertex) {
            const [rx, ry] = stats.reference_point_info.ref_vertex;
            this.updateElement('ref-point', `(${formatNumber(rx)}, ${formatNumber(ry)})`);
            this.refPointInfo = stats.reference_point_info;
        } else {
            this.updateElement('ref-point', 'N/A');
        }

        // Update average element quality
        if (stats.avg_element_quality !== undefined && stats.avg_element_quality !== null) {
            this.updateElement('avg-element-quality', formatNumber(stats.avg_element_quality, 4));
        } else {
            this.updateElement('avg-element-quality', 'N/A');
        }

        // Update detailed statistics (if statsContainer exists)
        const statsContainer = document.getElementById('stats-container');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <span>Episode: ${stats.episode || 'N/A'}</span>
                <span>Total Steps: ${stats.total_steps || 'N/A'}</span>
                <span>Episode Reward: ${formatNumber(stats.episode_reward)}</span>
                <span>Average Reward: ${formatNumber(stats.average_reward)}</span>
                <span>Episode Length: ${stats.episode_length || 'N/A'}</span>
                <span>Boundary Vertices: ${stats.boundary_vertices || 'N/A'}</span>
                <span>Avg Element Quality: ${formatNumber(stats.avg_element_quality, 4)}</span>
                <span>Buffer Size: ${stats.buffer_size || 'N/A'}</span>
                <span>Actor Loss: ${formatNumber(stats.recent_actor_loss)}</span>
                <span>Critic Loss: ${formatNumber(stats.recent_critic_loss)}</span>
                <span>Alpha: ${formatNumber(stats.current_alpha)}</span>
                <span>Last Eval: ${formatNumber(stats.last_eval_reward, 3)}</span>
                <span>Best Eval: ${formatNumber(stats.best_eval_reward, 3)}</span>
                <span>Eval Freq: ${stats.evaluation_frequency || 'N/A'} episodes</span>
            `;
        }

        // Update mesh and boundary data
        if (stats.mesh_data) {
            this.meshData = stats.mesh_data;
        }
        if (stats.boundary_vertices_data) {
            this.boundaryData = stats.boundary_vertices_data;
        }
    }

    /**
     * Update progress information
     * @param {Object} progress - Progress data
     */
    updateProgressInfo(progress) {
        if (!progress) return;

        if (progress.current_episode !== undefined) {
            this.updateElement('current-episode', progress.current_episode);
        }

        if (progress.total_steps !== undefined) {
            this.updateElement('total-steps', progress.total_steps);
            
            // Update estimated finish time
            this.updateEstimatedFinishTime(progress.total_steps);
        }

        if (progress.average_reward !== undefined) {
            this.updateElement('avg-reward', formatNumber(progress.average_reward));
        }

        if (progress.buffer_utilization !== undefined) {
            this.updateElement('buffer-size', progress.buffer_utilization);
        }

        if (progress.latest_reward !== undefined) {
            this.updateElement('episode-reward', formatNumber(progress.latest_reward));
        }
        
        // Training metrics are updated via updateTrainingStats method
        // No need for compact versions as they don't exist in HTML
    }

    /**
     * Update estimated finish time based on current progress
     * @param {number} currentSteps - Current total steps completed
     */
    updateEstimatedFinishTime(currentSteps) {
        const now = Date.now();
        
        // Handle mid-training refresh scenario
        if (this.trainingStartTime === null && currentSteps > 0 && this.progressHistory.length === 0) {
            // This appears to be a mid-training refresh - estimate when training started
            this.adjustForMidTrainingRefresh(currentSteps, now);
        } else if (this.trainingStartTime === null && currentSteps > 0) {
            // Normal case - training just started from frontend
            this.trainingStartTime = now;
        }

        // Get max timesteps from configuration
        if (this.maxTimesteps === null) {
            this.maxTimesteps = this.getMaxTimestepsFromConfig();
        }

        // Update progress history for rate calculation
        this.progressHistory.push({
            timestamp: now,
            steps: currentSteps
        });

        // Keep only recent history (last 15 entries to get better trend data)
        if (this.progressHistory.length > 15) {
            this.progressHistory.shift();
        }

        const estimatedTime = this.calculateEstimatedFinishTime(currentSteps);
        this.updateElement('estimated-finish-time', estimatedTime);
    }

    /**
     * Calculate estimated finish time based on progress rate
     * @param {number} currentSteps - Current total steps
     * @returns {string} Formatted estimated finish time or status
     */
    calculateEstimatedFinishTime(currentSteps) {
        // Return N/A if we don't have enough data
        if (!this.maxTimesteps || currentSteps <= 0 || this.progressHistory.length < 1) {
            return 'N/A';
        }

        // If training is complete
        if (currentSteps >= this.maxTimesteps) {
            return 'Completed';
        }

        // For first update, just return "Calculating..."
        if (this.progressHistory.length < 2) {
            return 'Calculating...';
        }

        // Calculate rate using different strategies based on progress pattern
        let stepsPerSecond = 0;
        const now = Date.now();
        
        // Strategy 1: Try to use recent progress (last few data points)
        const recentProgress = this.progressHistory.slice(-Math.min(5, this.progressHistory.length));
        if (recentProgress.length >= 2) {
            const recentFirst = recentProgress[0];
            const recentLast = recentProgress[recentProgress.length - 1];
            const recentTimeElapsed = (recentLast.timestamp - recentFirst.timestamp) / 1000;
            const recentStepsProgress = recentLast.steps - recentFirst.steps;
            
            if (recentTimeElapsed > 0 && recentStepsProgress > 0) {
                stepsPerSecond = recentStepsProgress / recentTimeElapsed;
            }
        }
        
        // Strategy 2: If recent progress shows no step increase, use overall progress rate
        if (stepsPerSecond <= 0 && this.progressHistory.length >= 2) {
            const firstPoint = this.progressHistory[0];
            const lastPoint = this.progressHistory[this.progressHistory.length - 1];
            const totalTimeElapsed = (lastPoint.timestamp - firstPoint.timestamp) / 1000;
            const totalStepsProgress = lastPoint.steps - firstPoint.steps;
            
            if (totalTimeElapsed > 0 && totalStepsProgress > 0) {
                stepsPerSecond = totalStepsProgress / totalTimeElapsed;
            }
        }
        
        // Strategy 3: If we still have no progress but have training start time, calculate from beginning
        if (stepsPerSecond <= 0 && this.trainingStartTime !== null) {
            const totalTrainingTime = (now - this.trainingStartTime) / 1000;
            if (totalTrainingTime > 0 && currentSteps > 0) {
                stepsPerSecond = currentSteps / totalTrainingTime;
            }
        }
        
        // If still no valid rate can be calculated
        if (stepsPerSecond <= 0) {
            return 'Calculating...';
        }

        const remainingSteps = this.maxTimesteps - currentSteps;
        const estimatedSecondsRemaining = remainingSteps / stepsPerSecond;

        // Debug logging for timing calculations
        console.debug('Time estimation:', {
            currentSteps,
            maxTimesteps: this.maxTimesteps,
            remainingSteps,
            stepsPerSecond: stepsPerSecond.toFixed(4),
            estimatedSecondsRemaining: estimatedSecondsRemaining.toFixed(2),
            progressHistoryLength: this.progressHistory.length,
            trainingStartTime: this.trainingStartTime ? new Date(this.trainingStartTime).toLocaleTimeString() : 'null'
        });

        // Format the estimated time
        return this.formatDuration(estimatedSecondsRemaining);
    }

    /**
     * Format duration in seconds to human readable format
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration string
     */
    formatDuration(seconds) {
        if (isNaN(seconds) || seconds < 0) {
            return 'N/A';
        }

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    /**
     * Get max timesteps from the training configuration
     * @returns {number|null} Max timesteps or null if not available
     */
    getMaxTimestepsFromConfig() {
        const maxTimestepsInput = this.elements['max-timesteps'];
        if (maxTimestepsInput && maxTimestepsInput.value) {
            const value = parseInt(maxTimestepsInput.value);
            return isNaN(value) ? null : value;
        }
        return null;
    }

    /**
     * Reset estimation tracking (call when training starts)
     */
    resetEstimationTracking() {
        this.progressHistory = [];
        this.trainingStartTime = null;
        this.maxTimesteps = null;
        this.updateElement('estimated-finish-time', 'N/A');
    }
    
    /**
     * Handle mid-training refresh scenario by adjusting start time estimation
     * This method is called when we detect that the frontend was refreshed during training
     * @param {number} currentSteps - Current steps when frontend was refreshed
     * @param {number} refreshTime - Time when frontend was refreshed (timestamp)
     */
    adjustForMidTrainingRefresh(currentSteps, refreshTime) {
        // If we have a reasonable assumption about training speed, we can estimate when training actually started
        // This is a heuristic approach for when user refreshes frontend mid-training
        
        if (currentSteps > 0 && refreshTime) {
            // Assume a reasonable average training speed (steps per second) to backtrack start time
            // This is just an initial estimate that will be refined as we get more data points
            const assumedAverageStepsPerSecond = 2; // Conservative estimate
            const estimatedElapsedSeconds = currentSteps / assumedAverageStepsPerSecond;
            const estimatedStartTime = refreshTime - (estimatedElapsedSeconds * 1000);
            
            // Set the estimated training start time
            this.trainingStartTime = estimatedStartTime;
            
            // Add the current point as the first data point
            this.progressHistory = [{
                timestamp: refreshTime,
                steps: currentSteps
            }];
        }
    }

    /**
     * Update UI button states
     * @param {boolean} isTraining - Whether training is in progress
     */
    updateButtonStates(isTraining) {
        this.isTraining = isTraining;

        const buttonStates = {
            'start-btn': !isTraining,
            'stop-btn': isTraining,
            'mesh-select': !isTraining,
            'max-timesteps': !isTraining,
            'max-steps': !isTraining,
            'update-interval': !isTraining,
            'description': !isTraining,
            // New checkpoint-related controls
            'checkpoint-mode': !isTraining,
            'checkpoint-select': !isTraining
        };

        Object.entries(buttonStates).forEach(([elementId, enabled]) => {
            const element = this.elements[elementId];
            if (element) {
                element.disabled = !enabled;
            }
        });
    }

    /**
     * Show/hide loading indicator
     * @param {boolean} show - Whether to show
     */
    showLoading(show) {
        const overlay = this.elements['loading-overlay'];
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    /**
     * Populate mesh selection list
     * @param {Array} meshes - Mesh list
     */
    populateMeshList(meshes) {
        const select = this.elements['mesh-select'];
        if (!select) return;

        // Clear existing options
        select.innerHTML = '<option value="">Select a Mesh</option>';

        if (Array.isArray(meshes) && meshes.length > 0) {
            meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh;
                option.textContent = mesh;
                select.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No mesh files found';
            select.appendChild(option);
        }
    }

    /**
     * Populate checkpoint selection list
     * @param {Array} checkpoints - Checkpoint list
     */
    populateCheckpointList(checkpoints) {
        const select = this.elements['checkpoint-select'];
        if (!select) return;

        this.checkpoints = checkpoints || [];

        // Clear existing options
        select.innerHTML = '<option value="">Select a Checkpoint</option>';

        if (Array.isArray(checkpoints) && checkpoints.length > 0) {
            checkpoints.forEach(checkpoint => {
                const option = document.createElement('option');
                option.value = checkpoint.name;

                // Display checkpoint name and related information
                const displayText = `${checkpoint.name} (${checkpoint.modified_datetime}, ${checkpoint.file_size_mb}MB)`;
                option.textContent = displayText;

                // If checkpoint is invalid, disable the option
                if (!checkpoint.is_valid) {
                    option.disabled = true;
                    option.textContent += ' [Invalid]';
                }

                select.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No checkpoint files found';
            select.appendChild(option);
        }
    }

    /**
     * Show mesh information
     * @param {Object} info - Mesh information
     */
    showMeshInfo(info) {
        if (!info) return;

        this.updateElement('mesh-vertices', info.vertex_count || 0);
        this.updateElement('mesh-size', info.file_size || 0);

        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.remove('hidden');
        }
    }

    /**
     * Hide mesh information
     */
    hideMeshInfo() {
        const infoDiv = this.elements['mesh-info'];
        if (infoDiv) {
            infoDiv.classList.add('hidden');
        }
    }

    /**
     * Show checkpoint information
     * @param {Object} info - Checkpoint information
     */
    showCheckpointInfo(info) {
        if (!info) return;

        const infoDiv = this.elements['checkpoint-info'];
        const detailsDiv = this.elements['checkpoint-details'];

        if (infoDiv) {
            infoDiv.classList.remove('hidden');
        }

        if (detailsDiv) {
            detailsDiv.innerHTML = `
                <div class="text-xs text-gray-600 space-y-1">
                    <div>Training steps: ${info.training_timesteps.toLocaleString()}</div>
                    <div>Learning rate: ${info.learning_rate}</div>
                    <div>File size: ${info.file_size_mb} MB</div>
                    <div>Modified: ${info.modified_datetime}</div>
                    <div>Valid: ${info.is_valid ? '✓ Valid' : '✗ Invalid'}</div>
                    ${info.has_replay_buffer ? '<div>Contains replay buffer</div>' : ''}
                </div>
            `;
        }
    }

    /**
     * Hide checkpoint information
     */
    hideCheckpointInfo() {
        const infoDiv = this.elements['checkpoint-info'];
        if (infoDiv) {
            infoDiv.classList.add('hidden');
        }
    }

    /**
     * Control checkpoint selection area visibility
     * @param {boolean} show - Whether to show
     */
    showCheckpointSelection(show) {
        const checkpointSelect = this.elements['checkpoint-select'];
        const checkpointInfo = this.elements['checkpoint-info'];

        if (checkpointSelect) {
            checkpointSelect.style.display = show ? 'block' : 'none';
        }

        if (!show && checkpointInfo) {
            checkpointInfo.classList.add('hidden');
        }
    }

    /**
     * Log messages
     * @param {string} message - Message content
     * @param {string} type - Message type
     */
    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['log-container'];
        if (!container) return;

        const timestamp = getTimestamp();
        const style = getLogStyle(type);

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.style.color = style.color;
        logEntry.innerHTML = `<span style="color: #9CA3AF;">[${timestamp}]</span> ${style.icon} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // Limit log entries
        while (container.children.length > CONSTANTS.MAX_LOGS) {
            container.removeChild(container.firstChild);
        }
    }

    /**
     * Clear logs
     */
    clearLogs() {
        const container = this.elements['log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">Logs cleared</div>';
        }
    }

    /**
     * Update click coordinates display
     * @param {Array} coords - World coordinates [x, y]
     */
    updateClickCoordinates(coords) {
        if (!coords || !Array.isArray(coords) || coords.length !== 2) {
            this.updateElement('click-coordinates', 'No transform data');
            return;
        }

        const coordText = `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`;
        this.updateElement('click-coordinates', coordText);
    }

    /**
     * Get training configuration - based on timestep control, supports checkpoint
     * @returns {Object} Training configuration
     */
    getTrainingConfig() {
        // Get all input values
        const maxTimestepsValue = this.getElementValue('max-timesteps');
        const maxStepsValue = this.getElementValue('max-steps');
        const descriptionValue = this.getElementValue('description');

        // Get checkpoint-related configuration - fixed version
        const checkpointModeElement = document.getElementById('checkpoint-mode');
        const useCheckpoint = checkpointModeElement ? checkpointModeElement.checked : false;

        const rawName = this.getElementValue('checkpoint-select').trim();
        const checkpointName = rawName !== '' ? rawName : null;

        // Add debug logs
        console.log('Checkpoint mode element:', checkpointModeElement);
        console.log('Use checkpoint:', useCheckpoint);
        console.log('Selected checkpoint:', checkpointName);

        let maxTimesteps = null;
        let maxSteps = null;

        // Safely parse max_timesteps (main control parameter)
        if (maxTimestepsValue && maxTimestepsValue.trim() !== '') {
            const parsed = parseInt(maxTimestepsValue.trim());
            if (!isNaN(parsed) && parsed > 0) {
                maxTimesteps = parsed;
            }
        }

        // Safely parse max_steps
        if (maxStepsValue && maxStepsValue.trim() !== '') {
            const parsed = parseInt(maxStepsValue.trim());
            if (!isNaN(parsed) && parsed > 0) {
                maxSteps = parsed;
            }
        }

        const config = {
            mesh_name: this.getElementValue('mesh-select'),
            max_timesteps: maxTimesteps,
            max_steps: maxSteps,
            description: descriptionValue && descriptionValue.trim() !== '' ? descriptionValue.trim() : null
        };

        // If using checkpoint, add checkpoint configuration
        if (useCheckpoint && checkpointName) {
            config.checkpoint_name = checkpointName;
            config.from_checkpoint = !!useCheckpoint;
        }

        return config;
    }

    /**
     * Get update interval
     * @returns {number} Update interval (milliseconds)
     */
    getUpdateInterval() {
        const interval = parseInt(this.getElementValue('update-interval')) || 10;
        return interval * 1000; // Convert to milliseconds
    }

    /**
     * Validate training configuration - based on timestep control, supports checkpoint
     * @returns {Object} Validation result {valid: boolean, message: string}
     */
    validateTrainingConfig() {
        const config = this.getTrainingConfig();

        if (!config.mesh_name) {
            return {
                valid: false,
                message: 'Please select a mesh file first'
            };
        }

        // Validate checkpoint (if checkpoint is selected)
        const checkpointModeElement = document.getElementById('checkpoint-mode');
        const useCheckpoint = checkpointModeElement && checkpointModeElement.checked;

        if (useCheckpoint) {
            if (!config.checkpoint_name) {
                return {
                    valid: false,
                    message: 'A checkpoint must be selected when checkpoint mode is enabled'
                };
            }

            // Check if selected checkpoint is valid
            const selectedCheckpoint = this.checkpoints.find(cp => cp.name === config.checkpoint_name);
            if (!selectedCheckpoint || !selectedCheckpoint.is_valid) {
                return {
                    valid: false,
                    message: 'Selected checkpoint is invalid'
                };
            }
        }

        // Main validation: max_timesteps
        if (!config.max_timesteps) {
            return {
                valid: false,
                message: 'Please specify max training steps'
            };
        }

        if (config.max_timesteps && config.max_timesteps < 1000) {
            return {
                valid: false,
                message: 'Max training steps should be at least 1000'
            };
        }

        if (config.max_steps && config.max_steps < 1) {
            return {
                valid: false,
                message: 'Max steps per episode must be greater than 0'
            };
        }

        return {valid: true};
    }

    /**
     * Get mesh and boundary data
     * @returns {Object} Object containing mesh and boundary data
     */
    getRenderData() {
        return {
            meshData: this.meshData,
            boundaryData: this.boundaryData,
            refPointInfo: this.refPointInfo
        };
    }

    /**
     * Update text content of a single element
     * @param {string} elementId - Element ID
     * @param {any} value - Value
     */
    updateElement(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.textContent = value;
        } else {
            console.warn(`UI: Element with ID '${elementId}' not found`);
        }
    }

    /**
     * Get element value
     * @param {string} elementId - Element ID
     * @returns {string} Element value
     */
    getElementValue(elementId) {
        // If element wasn't found during initialization, try again and cache it
        let element = this.elements[elementId];
        if (!element) {
            element = document.getElementById(elementId);
            if (element) this.elements[elementId] = element;   // Add to cache
        }
        const value = element ? element.value : '';

        // Add debug logs
        if (elementId === 'checkbox-select') {
            console.log(`getElementValue(${elementId}):`, {
                element: element,
                value: value,
                directValue: document.getElementById(elementId)?.value
            });
        }

        return value;
    }

    /**
     * Set element value
     * @param {string} elementId - Element ID
     * @param {any} value - Value
     */
    setElementValue(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.value = value;
        }
    }

    /**
     * Show error state
     * @param {string} message - Error message
     */
    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
        this.updateStatusIndicator(STATUS.ERROR);
    }

    /**
     * Show success state
     * @param {string} message - Success message
     */
    showSuccess(message) {
        this.logMessage(message, LOG_TYPES.SUCCESS);
    }

    /**
     * Show warning state
     * @param {string} message - Warning message
     */
    showWarning(message) {
        this.logMessage(message, LOG_TYPES.WARNING);
    }

    /**
     * Reset UI to initial state
     */
    reset() {
        this.isTraining = false;
        this.meshData = null;
        this.boundaryData = null;
        this.refPointInfo = null;
        this.checkpoints = [];

        this.updateStatusIndicator(STATUS.IDLE);
        this.updateButtonStates(false);
        this.showLoading(false);
        this.clearLogs();
        this.updateClickCoordinates(null);
        this.hideCheckpointInfo();
        this.resetEstimationTracking();
    }
}