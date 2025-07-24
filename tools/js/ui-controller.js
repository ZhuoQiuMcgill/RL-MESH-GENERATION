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
    }

    /**
     * Initialize DOM element references
     * @returns {Object} Element reference object
     */
    initializeElements() {
        const elementIds = [
            'status-indicator', 'status-text', 'mesh-select', 'mesh-info',
            'mesh-vertices', 'mesh-size', 'start-btn', 'stop-btn',
            'refresh-btn', 'clear-log-btn', 'max-timesteps', 'max-steps',
            'update-interval', 'description', 'current-episode', 'total-steps', 'avg-reward',
            'buffer-size', 'episode-reward', 'episode-length', 'ref-point',
            'click-coordinates', 'display-episode', 'display-total-steps', 'boundary-vertices',
            'log-container', 'loading-overlay',
            // New checkpoint-related elements
            'checkpoint-mode', 'checkpoint-select', 'checkpoint-info', 'checkpoint-details',
            // Training metrics elements
            'actor-loss', 'critic-loss', 'alpha-value'
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
        const indicator = this.elements['status-indicator']?.querySelector('div');
        const text = this.elements['status-text'];

        if (!indicator || !text) return;

        // Remove all status classes
        indicator.className = 'w-2 h-2 rounded-full mr-2';

        const statusConfig = {
            [STATUS.RUNNING]: {class: 'status-running', text: 'Running'},
            [STATUS.STOPPED]: {class: 'status-stopped', text: 'Stopped'},
            [STATUS.COMPLETED]: {class: 'status-success', text: 'Completed'},
            [STATUS.STOPPING]: {class: 'status-loading', text: 'Stopping'},
            [STATUS.ERROR]: {class: 'status-stopped', text: 'Error'},
            [STATUS.IDLE]: {class: 'status-idle', text: 'Idle'}
        };

        const config = statusConfig[status] || statusConfig[STATUS.IDLE];
        indicator.classList.add(config.class);
        text.textContent = config.text;
    }

    /**
     * Update training statistics
     * @param {Object} stats - Statistics data
     */
    updateTrainingStats(stats) {
        if (!stats) return;

        // Update basic statistics
        this.updateElement('current-episode', stats.episode || 0);
        this.updateElement('display-episode', stats.episode || 0);
        this.updateElement('total-steps', stats.total_steps || 0);
        this.updateElement('display-total-steps', stats.total_steps || 0);
        this.updateElement('avg-reward', formatNumber(stats.average_reward));
        this.updateElement('buffer-size', stats.buffer_size || 0);
        this.updateElement('episode-reward', formatNumber(stats.episode_reward));
        this.updateElement('episode-length', stats.episode_length || 0);
        this.updateElement('boundary-vertices', stats.boundary_vertices || 0);

        // Update training metrics (new) - simplified for debugging
        console.log('DEBUG: Raw stats object:', stats);
        console.log('DEBUG: recent_actor_loss:', stats.recent_actor_loss);
        console.log('DEBUG: recent_critic_loss:', stats.recent_critic_loss);
        console.log('DEBUG: current_alpha:', stats.current_alpha);
        
        this.updateElement('actor-loss', formatNumber(stats.recent_actor_loss, 4));
        this.updateElement('critic-loss', formatNumber(stats.recent_critic_loss, 4));
        this.updateElement('alpha-value', formatNumber(stats.current_alpha, 4));

        // Update reference point information
        if (stats.reference_point_info && stats.reference_point_info.ref_vertex) {
            const [rx, ry] = stats.reference_point_info.ref_vertex;
            this.updateElement('ref-point', `(${formatNumber(rx)}, ${formatNumber(ry)})`);
            this.refPointInfo = stats.reference_point_info;
        } else {
            this.updateElement('ref-point', 'N/A');
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
                <span>Buffer Size: ${stats.buffer_size || 'N/A'}</span>
                <span>Actor Loss: ${formatNumber(stats.recent_actor_loss)}</span>
                <span>Critic Loss: ${formatNumber(stats.recent_critic_loss)}</span>
                <span>Alpha: ${formatNumber(stats.current_alpha)}</span>
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
            this.updateElement('display-episode', progress.current_episode);
        }

        if (progress.total_steps !== undefined) {
            this.updateElement('total-steps', progress.total_steps);
            this.updateElement('display-total-steps', progress.total_steps);
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
        console.log(`DEBUG updateElement: ${elementId} = ${value}, element found: ${!!element}`);
        if (element) {
            element.textContent = value;
            console.log(`DEBUG: Updated ${elementId} to "${value}"`);
        } else {
            console.error(`DEBUG: Element with ID '${elementId}' not found!`);
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
    }
}