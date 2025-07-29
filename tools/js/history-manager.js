/**
 * Training History Manager
 * Responsible for managing the viewing and visualization of historical training data
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {HistoryApiClient} from './history-api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';

export class HistoryManager {
    constructor() {
        // Initialize all modules
        this.apiClient = new HistoryApiClient();
        this.canvasRenderer = null; // Lazy initialization

        // State management
        this.trainingList = [];
        this.currentTrainingId = null;
        this.currentTrainingInfo = null;
        this.currentEpisodeIndex = null;
        this.currentEpisodeData = null;

        // State management to prevent duplicate requests
        this.isLoadingHistory = false;
        this.isLoadingTraining = false;
        this.isLoadingEpisode = false;

        // DOM element references
        this.elements = this.initializeElements();

        this.init();
    }

    /**
     * Initialize DOM element references
     */
    initializeElements() {
        const elementIds = [
            'training-list', 'current-training-info', 'episode-navigation',
            'training-id-display', 'detail-length-display', 'best-episode-display',
            'episode-index-input', 'current-episode-display', 'episode-meta-info',
            'episode-index-display', 'episode-reward-display', 'episode-length-display', 'episode-status-display',
            'actual-episode-number-display', 'actual-episode-number-display-header', 'boundary-vertices-count', 'mesh-vertices-count', 'ref-point-display', 'avg-element-quality-display',
            'click-coordinates-display', 'episode-data-container',
            'history-log-container', 'history-loading-overlay',
            // Buttons
            'refresh-history-btn', 'health-check-btn', 'goto-episode-btn',
            'goto-best-episode-btn', 'goto-last-episode-btn', 'prev-episode-btn',
            'next-episode-btn', 'clear-history-log-btn'
        ];

        const elements = {};
        elementIds.forEach(id => {
            elements[id] = document.getElementById(id);
        });

        return elements;
    }

    /**
     * Initialize application
     */
    async init() {
        try {
            this.setupCanvas();
            this.bindEvents();

            // Check backend connection
            const isConnected = await this.checkBackendConnection();
            if (isConnected) {
                await this.loadTrainingHistory();
            } else {
                this.logMessage('Cannot connect to history API server', LOG_TYPES.ERROR);
            }

            this.logMessage('History viewer initialized', LOG_TYPES.INFO);
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('System initialization failed: ' + error.message);
        }
    }

    /**
     * Setup Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('history-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
        } else {
            console.error('Canvas element not found');
        }
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Refresh history button
        const refreshBtn = this.elements['refresh-history-btn'];
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshTrainingHistory());
        }

        // Health check button
        const healthBtn = this.elements['health-check-btn'];
        if (healthBtn) {
            healthBtn.addEventListener('click', () => this.checkHealthStatus());
        }

        // Episode navigation buttons
        const gotoBtn = this.elements['goto-episode-btn'];
        if (gotoBtn) {
            gotoBtn.addEventListener('click', () => this.gotoEpisode());
        }

        const gotoBestBtn = this.elements['goto-best-episode-btn'];
        if (gotoBestBtn) {
            gotoBestBtn.addEventListener('click', () => this.gotoBestEpisode());
        }

        const gotoLastBtn = this.elements['goto-last-episode-btn'];
        if (gotoLastBtn) {
            gotoLastBtn.addEventListener('click', () => this.gotoLastEpisode());
        }

        const prevBtn = this.elements['prev-episode-btn'];
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.gotoPreviousEpisode());
        }

        const nextBtn = this.elements['next-episode-btn'];
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.gotoNextEpisode());
        }

        // Clear log button
        const clearLogBtn = this.elements['clear-history-log-btn'];
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.clearLogs());
        }

        // Episode input box enter event
        const episodeInput = this.elements['episode-index-input'];
        if (episodeInput) {
            episodeInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.gotoEpisode();
                }
            });
        }

        // Canvas click event
        const canvas = document.getElementById('history-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * Check backend connection status
     */
    async checkBackendConnection() {
        try {
            const response = await this.apiClient.checkHistoryHealth();
            if (response.success && response.status === 'healthy') {
                this.logMessage('History API connection successful', LOG_TYPES.SUCCESS);
                return true;
            } else {
                this.logMessage('History API status abnormal: ' + (response.error || 'Unknown error'), LOG_TYPES.WARNING);
                return false;
            }
        } catch (error) {
            this.logMessage('History API connection failed: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * Load training history list
     */
    async loadTrainingHistory() {
        if (this.isLoadingHistory) {
            this.logMessage('Loading training history, please wait...', LOG_TYPES.WARNING);
            return;
        }
        try {
            this.isLoadingHistory = true;
            this.showLoading(true);

            const response = await this.apiClient.getTrainingHistoryList();

            if (response.success && response.training_ids) {
                this.trainingList = response.training_ids;
                this.updateTrainingList();
                this.logMessage(`Loaded ${response.count} training history entries`, LOG_TYPES.SUCCESS);
            } else {
                this.logMessage('No training history found: ' + (response.error || 'Unknown error'), LOG_TYPES.WARNING);
                this.trainingList = [];
                this.updateTrainingList();
            }

        } catch (error) {
            console.error('Failed to load training history:', error);
            this.showError('Failed to load training history: ' + error.message);
        } finally {
            this.showLoading(false);
            this.isLoadingHistory = false;
        }
    }

    /**
     * Refresh training history
     */
    async refreshTrainingHistory() {
        if (this.isLoadingHistory) {
            this.logMessage('Refreshing, please wait...', LOG_TYPES.WARNING);
            return;
        }
        this.logMessage('Refreshing training history...', LOG_TYPES.INFO);
        await this.loadTrainingHistory();
    }

    /**
     * Check health status
     */
    async checkHealthStatus() {
        try {
            this.showLoading(true);
            const response = await this.apiClient.checkHistoryHealth();

            if (response.success && response.status === 'healthy') {
                this.logMessage(`Health check passed - available trainings: ${response.available_trainings}`, LOG_TYPES.SUCCESS);
                if (response.current_focus) {
                    this.logMessage(`Current focus: ${response.current_focus}`, LOG_TYPES.INFO);
                }
            } else {
                this.logMessage(`Service status abnormal: ${response.error || 'Unknown error'}`, LOG_TYPES.ERROR);
            }
        } catch (error) {
            this.logMessage(`Health check failed: ${error.message}`, LOG_TYPES.ERROR);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Update training list display
     */
    updateTrainingList() {
        const container = this.elements['training-list'];
        if (!container) return;

        if (!this.trainingList || this.trainingList.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <div class="text-sm">No training history records available</div>
                    <button class="mt-2 text-primary hover:text-blue-600 text-xs" onclick="window.historyManager.refreshTrainingHistory()">
                        Click to refresh
                    </button>
                </div>
            `;
            return;
        }

        // Sort training sessions by timestamp, newest first
        const sortedTrainingList = [...this.trainingList].sort((a, b) => {
            // Extract timestamp directly from training ID
            const aTimestamp = this.extractTimestampFromTrainingId(a);
            const bTimestamp = this.extractTimestampFromTrainingId(b);
            
            // If both have valid timestamps, sort in reverse chronological order (newest first)
            if (aTimestamp && bTimestamp) {
                return bTimestamp.localeCompare(aTimestamp); // Newer first
            }
            
            // If only one has timestamp, the one with timestamp comes first
            if (aTimestamp && !bTimestamp) return -1;
            if (!aTimestamp && bTimestamp) return 1;
            
            // If neither has timestamp, sort by ID string in reverse order (newer first)
            return b.localeCompare(a);
        });

        // Generate training item HTML
        const itemsHTML = sortedTrainingList.map(trainingId => {
            const isActive = trainingId === this.currentTrainingId;
            const displayName = this.formatTrainingDisplayName(trainingId);

            return `
                <div class="training-item ${isActive ? 'active' : ''}" data-training-id="${trainingId}">
                    <div class="training-name">${displayName.name}</div>
                    <div class="training-meta">
                        <div>${displayName.timestamp}</div>
                        <div>Mesh: ${displayName.mesh}</div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = itemsHTML;

        // Bind click events, ensure no duplicate binding causing multiple requests
        container.onclick = (e) => {
            const trainingItem = e.target.closest('.training-item');
            if (trainingItem) {
                const trainingId = trainingItem.dataset.trainingId;
                this.selectTraining(trainingId);
            }
        };
    }

    /**
     * Extract timestamp from training ID for sorting purposes
     * @param {string} trainingId - Training ID like "sac_20250729_143022_mesh1" or "continue_checkpoint_20250729_143022_mesh1"
     * @returns {string|null} Timestamp in format "YYYYMMDD_HHMMSS" or null if not found
     */
    extractTimestampFromTrainingId(trainingId) {
        if (!trainingId || typeof trainingId !== 'string') {
            return null;
        }

        const parts = trainingId.split('_');
        
        // Look for consecutive parts that match date (8 digits) and time (6 digits) pattern
        for (let i = 0; i < parts.length - 1; i++) {
            const datePart = parts[i];
            const timePart = parts[i + 1];
            
            // Check if current part is 8-digit date and next part is 6-digit time
            if (datePart && datePart.length === 8 && /^\d{8}$/.test(datePart) &&
                timePart && timePart.length === 6 && /^\d{6}$/.test(timePart)) {
                return `${datePart}_${timePart}`;
            }
        }
        
        return null;
    }

    /**
     * Format training display name
     */
    formatTrainingDisplayName(trainingId) {
        return this.apiClient.formatTrainingDisplayName(trainingId);
    }

    /**
     * Select training session
     */
    async selectTraining(trainingId) {
        if (trainingId === this.currentTrainingId) return;
        if (this.isLoadingTraining) {
            this.logMessage('Training is loading, please wait...', LOG_TYPES.WARNING);
            return;
        }

        try {
            this.isLoadingTraining = true;
            this.showLoading(true);
            this.logMessage(`Loading training: ${trainingId}`, LOG_TYPES.INFO);

            // Get training information
            const response = await this.apiClient.getTrainingInfo(trainingId);

            if (response.success) {
                this.currentTrainingId = trainingId;
                this.currentTrainingInfo = {
                    training_id: response.training_id,
                    detail_length: response.detail_length,
                    best_episode: response.best_episode
                };

                // Update UI
                this.updateTrainingList(); // Refresh list to show selected state
                this.updateTrainingInfo();
                this.showTrainingControls(true);

                // Load best episode by default
                await this.loadEpisode(this.currentTrainingInfo.best_episode);

                this.logMessage(`Training loaded successfully: ${response.detail_length} episodes`, LOG_TYPES.SUCCESS);
            } else {
                this.showError('Failed to load training info: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to select training:', error);
            this.showError('Failed to select training: ' + error.message);
        } finally {
            this.showLoading(false);
            this.isLoadingTraining = false;
        }
    }

    /**
     * Update training information display
     */
    updateTrainingInfo() {
        if (!this.currentTrainingInfo) return;

        const displayName = this.formatTrainingDisplayName(this.currentTrainingInfo.training_id);

        this.updateElement('training-id-display', displayName.name);
        this.updateElement('detail-length-display', this.currentTrainingInfo.detail_length);
        this.updateElement('best-episode-display', this.currentTrainingInfo.best_episode);

        // Update maximum value for episode input box
        const episodeInput = this.elements['episode-index-input'];
        if (episodeInput) {
            episodeInput.max = this.currentTrainingInfo.detail_length - 1;
        }
    }

    /**
     * Show/hide training control interface
     */
    showTrainingControls(show) {
        const infoDiv = this.elements['current-training-info'];
        const navDiv = this.elements['episode-navigation'];

        if (infoDiv) {
            if (show) {
                infoDiv.classList.remove('hidden');
            } else {
                infoDiv.classList.add('hidden');
            }
        }

        if (navDiv) {
            if (show) {
                navDiv.classList.remove('hidden');
            } else {
                navDiv.classList.add('hidden');
            }
        }
    }

    /**
     * Load specified episode data
     */
    async loadEpisode(episodeIndex) {
        if (!this.currentTrainingId || !this.currentTrainingInfo) {
            this.showError('Please select a training session first');
            return;
        }

        if (episodeIndex < 0 || episodeIndex >= this.currentTrainingInfo.detail_length) {
            this.showError(`Episode index out of range: ${episodeIndex}`);
            return;
        }

        try {
            this.showLoading(true);
            this.logMessage(`Loading Episode ${episodeIndex}...`, LOG_TYPES.INFO);

            const response = await this.apiClient.getEpisodeData(this.currentTrainingId, episodeIndex);

            if (response.success) {
                this.currentEpisodeIndex = episodeIndex;
                this.currentEpisodeData = response.episode_data;

                // Update UI
                this.updateEpisodeInfo();
                this.updateEpisodeData();
                this.updateVisualization();

                // Update episode input box value
                const episodeInput = this.elements['episode-index-input'];
                if (episodeInput) {
                    episodeInput.value = episodeIndex;
                }

                this.logMessage(`Episode ${episodeIndex} loaded successfully`, LOG_TYPES.SUCCESS);
            } else {
                this.showError('Failed to load episode: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to load episode:', error);
            this.showError('Failed to load episode: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Update episode information display
     */
    updateEpisodeInfo() {
        if (!this.currentEpisodeData) return;

        const {r: reward, l: length, is_completed, episode_number} = this.currentEpisodeData;

        // Debug: Log episode data to understand the structure
        console.log('Episode data structure:', this.currentEpisodeData);
        console.log('Episode number from data:', episode_number);

        this.updateElement('current-episode-display', `Episode ${this.currentEpisodeIndex}`);
        this.updateElement('episode-index-display', this.currentEpisodeIndex);
        this.updateElement('actual-episode-number-display', episode_number || 'N/A');
        this.updateElement('actual-episode-number-display-header', episode_number || 'N/A');
        this.updateElement('episode-reward-display', formatNumber(reward));
        this.updateElement('episode-length-display', length);
        this.updateElement('episode-status-display', is_completed ? 'Completed' : 'Incomplete');

        // Show meta information
        const metaInfo = this.elements['episode-meta-info'];
        if (metaInfo) {
            metaInfo.classList.remove('hidden');
        }

        // Update boundary and mesh vertex counts
        const boundaryVertices = this.currentEpisodeData.boundary_vertices_data || [];
        const meshData = this.currentEpisodeData.mesh_data || {};

        this.updateElement('boundary-vertices-count', boundaryVertices.length);
        this.updateElement('mesh-vertices-count', Object.keys(meshData).length);

        // Update reference point information
        const refPointInfo = this.currentEpisodeData.last_ref_point;
        if (refPointInfo && refPointInfo.ref_vertex) {
            const [rx, ry] = refPointInfo.ref_vertex;
            this.updateElement('ref-point-display', `(${formatNumber(rx)}, ${formatNumber(ry)})`);
        } else {
            this.updateElement('ref-point-display', 'N/A');
        }

        // Update average element quality
        const avgElementQuality = this.currentEpisodeData.avg_element_quality;
        if (avgElementQuality !== undefined && avgElementQuality !== null) {
            this.updateElement('avg-element-quality-display', formatNumber(avgElementQuality, 4));
        } else {
            this.updateElement('avg-element-quality-display', 'N/A');
        }
    }

    /**
     * Update episode detailed data display
     * Note: Basic episode data is now handled by updateEpisodeInfo()
     * This method is kept for any future additional data display needs
     */
    updateEpisodeData() {
        // Episode data is now integrated into the Episode Information section
        // and handled by updateEpisodeInfo(). This method is kept for future use.
        return;
    }

    /**
     * Update visualization
     */
    updateVisualization() {
        if (!this.canvasRenderer || !this.currentEpisodeData) return;

        const meshData = this.currentEpisodeData.mesh_data;
        const boundaryVertices = this.currentEpisodeData.boundary_vertices_data;
        const refPointInfo = this.currentEpisodeData.last_ref_point;

        this.canvasRenderer.renderScene(meshData, boundaryVertices, refPointInfo);
    }

    /**
     * Episode navigation methods
     */
    async gotoEpisode() {
        const episodeInput = this.elements['episode-index-input'];
        if (!episodeInput) return;

        const episodeIndex = parseInt(episodeInput.value);
        if (isNaN(episodeIndex)) {
            this.showError('Please enter a valid episode index');
            return;
        }

        await this.loadEpisode(episodeIndex);
    }

    async gotoBestEpisode() {
        if (!this.currentTrainingInfo) return;
        await this.loadEpisode(this.currentTrainingInfo.best_episode);
    }

    async gotoLastEpisode() {
        if (!this.currentTrainingInfo) return;
        await this.loadEpisode(this.currentTrainingInfo.detail_length - 1);
    }

    async gotoPreviousEpisode() {
        if (this.currentEpisodeIndex === null || this.currentEpisodeIndex <= 0) return;
        await this.loadEpisode(this.currentEpisodeIndex - 1);
    }

    async gotoNextEpisode() {
        if (!this.currentTrainingInfo || this.currentEpisodeIndex === null) return;
        if (this.currentEpisodeIndex >= this.currentTrainingInfo.detail_length - 1) return;
        await this.loadEpisode(this.currentEpisodeIndex + 1);
    }

    /**
     * Canvas click event handling
     */
    handleCanvasClick(event) {
        if (!this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) {
            this.updateElement('click-coordinates-display', 'No transform data');
            return;
        }

        const rect = event.target.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;

        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);
        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;

        this.updateElement('click-coordinates-display', coordText);
        this.logMessage(`Click coordinates: ${coordText}`, LOG_TYPES.INFO);
    }

    /**
     * Utility methods
     */
    showLoading(show) {
        const overlay = this.elements['history-loading-overlay'];
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
    }

    updateElement(elementId, value) {
        const element = this.elements[elementId];
        if (element) {
            element.textContent = value;
        }
    }

    logMessage(message, type = LOG_TYPES.INFO) {
        const container = this.elements['history-log-container'];
        if (!container) return;

        const timestamp = new Date().toLocaleTimeString();
        const colors = {
            [LOG_TYPES.SUCCESS]: '#059669',
            [LOG_TYPES.ERROR]: '#DC2626',
            [LOG_TYPES.WARNING]: '#D97706',
            [LOG_TYPES.INFO]: '#6B7280'
        };
        const icons = {
            [LOG_TYPES.SUCCESS]: '✓',
            [LOG_TYPES.ERROR]: '✗',
            [LOG_TYPES.WARNING]: '⚠',
            [LOG_TYPES.INFO]: 'ℹ'
        };

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.style.color = colors[type];
        logEntry.innerHTML = `<span style="color: #9CA3AF;">[${timestamp}]</span> ${icons[type]} ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // Limit number of log entries
        while (container.children.length > CONSTANTS.MAX_LOGS) {
            container.removeChild(container.firstChild);
        }
    }

    clearLogs() {
        const container = this.elements['history-log-container'];
        if (container) {
            container.innerHTML = '<div class="text-gray-500">Logs cleared</div>';
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        if (this.canvasRenderer) {
            this.canvasRenderer.onResize();
        }
        this.logMessage('Window resized', LOG_TYPES.INFO);
    }

    /**
     * Throttled version of canvas click event
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);

    /**
     * Destroy manager and clean up resources
     */
    destroy() {
        if (this.canvasRenderer) {
            this.canvasRenderer.destroy();
        }

        this.trainingList = [];
        this.currentTrainingId = null;
        this.currentTrainingInfo = null;
        this.currentEpisodeIndex = null;
        this.currentEpisodeData = null;

        console.log('HistoryManager destroyed');
    }
}