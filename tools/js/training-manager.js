/**
 * Reinforcement Learning Mesh Generation Training Management System - Checkpoint Support Version
 * Main TrainingManager class that integrates all functional modules
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {ApiClient, withErrorHandling, withRetry} from './api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';
import {UIController} from './ui-controller.js';

export class TrainingManager {
    constructor() {
        // Initialize all modules
        this.apiClient = new ApiClient();
        this.uiController = new UIController();
        this.canvasRenderer = null; // Delayed initialization

        // State management
        this.isTraining = false;
        this.updateInterval = null;
        this.immediateUpdateTimer = null; // New: immediate update timer

        // Create API methods with error handling
        this.safeApiCall = withErrorHandling.bind(this);

        this.init();
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
                await this.loadMeshList();
                await this.loadCheckpointList(); // New: load checkpoint list
            } else {
                this.uiController.logMessage('Cannot connect to backend server. Ensure the Flask app is running at http://localhost:5000', LOG_TYPES.ERROR);
            }

            this.uiController.updateButtonStates(false);
            this.uiController.logMessage('System initialization completed', LOG_TYPES.INFO);
        } catch (error) {
            console.error('Initialization failed:', error);
            this.uiController.showError('System initialization failed: ' + error.message);
        }
    }

    /**
     * Setup Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('mesh-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
            // Initially show empty state
            this.showEmptyState(true);
        } else {
            console.error('Canvas element not found');
        }
    }

    /**
     * Show or hide empty state overlay
     */
    showEmptyState(show = true) {
        const overlay = document.getElementById('empty-state-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Start training button
        const startBtn = document.getElementById('start-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startTraining());
        }

        // Stop training button
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopTraining());
        }

        // Refresh status button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshStatus());
        }

        // Clear log button
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.uiController.clearLogs());
        }

        // Mesh selection change - enhanced version with preview support
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', (e) => this.onMeshSelectionChange(e.target.value));
        }

        // New: Checkpoint mode toggle
        const checkpointMode = document.getElementById('checkpoint-mode');
        if (checkpointMode) {
            checkpointMode.addEventListener('change', (e) => this.onCheckpointModeChange(e.target.checked));
        }

        // New: Checkpoint selection change
        const checkpointSelect = document.getElementById('checkpoint-select');
        if (checkpointSelect) {
            checkpointSelect.addEventListener('change', (e) => this.onCheckpointSelectionChange(e.target.value));
        }

        // Canvas click event
        const canvas = document.getElementById('mesh-canvas');
        if (canvas && this.canvasRenderer) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * Check backend connection status
     * @returns {Promise<boolean>} Connection status
     */
    async checkBackendConnection() {
        try {
            const connected = await this.apiClient.checkConnection();
            if (connected) {
                this.uiController.logMessage('Backend connection successful', LOG_TYPES.SUCCESS);
            }
            return connected;
        } catch (error) {
            this.uiController.logMessage('Backend connection failed: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * Load available mesh list
     */
    async loadMeshList() {
        try {
            this.uiController.showLoading(true);

            const data = await withRetry(() => this.apiClient.getMeshList());

            this.uiController.populateMeshList(data.meshes || []);

            if (data.meshes && data.meshes.length > 0) {
                this.uiController.logMessage(`Loaded ${data.meshes.length} mesh files`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('No mesh files found', LOG_TYPES.WARNING);
            }

        } catch (error) {
            console.error('Failed to load mesh list:', error);
            this.uiController.showError('Failed to load mesh list: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * New: Load available checkpoint list
     */
    async loadCheckpointList() {
        try {
            const data = await withRetry(() => this.apiClient.getCheckpointList());

            this.uiController.populateCheckpointList(data.checkpoints || []);

            if (data.checkpoints && data.checkpoints.length > 0) {
                this.uiController.logMessage(`Loaded ${data.checkpoints.length} checkpoint files`, LOG_TYPES.SUCCESS);
            } else {
                this.uiController.logMessage('No checkpoint files found', LOG_TYPES.INFO);
            }

        } catch (error) {
            console.error('Failed to load checkpoint list:', error);
            this.uiController.logMessage('Failed to load checkpoint list: ' + error.message, LOG_TYPES.WARNING);
        }
    }

    /**
     * Mesh selection change event handler - enhanced version with preview support
     * @param {string} meshName - Selected mesh name
     */
    async onMeshSelectionChange(meshName) {
        if (!meshName) {
            this.uiController.hideMeshInfo();
            // Clear canvas and show default prompt
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
                // Show empty state when canvas is cleared
                this.showEmptyState(true);
            }
            return;
        }

        try {
            this.uiController.showLoading(true);

            // Get mesh info and boundary data simultaneously
            const [info, boundaryData] = await Promise.all([
                this.apiClient.getMeshInfo(meshName),
                this.apiClient.getMeshBoundary(meshName)
            ]);

            // Update UI info
            this.uiController.showMeshInfo(info);
            this.uiController.logMessage(`Selected mesh: ${meshName}`, LOG_TYPES.INFO);

            // Render boundary preview in canvas
            if (this.canvasRenderer && boundaryData.success) {
                this.canvasRenderer.renderBoundaryPreview(
                    boundaryData.boundary_vertices,
                    meshName
                );
                // Hide empty state when mesh data is rendered
                this.showEmptyState(false);
                this.uiController.logMessage(
                    `Loaded boundary preview: ${boundaryData.vertex_count} vertices`,
                    LOG_TYPES.SUCCESS
                );
            } else if (!boundaryData.success) {
                this.uiController.logMessage(
                    `Failed to load boundary data: ${boundaryData.error}`,
                    LOG_TYPES.WARNING
                );
            }

        } catch (error) {
            console.error('Failed to get mesh info:', error);
            this.uiController.showError('Failed to get mesh info: ' + error.message);
            this.uiController.hideMeshInfo();

            // Clear canvas
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
                // Show empty state when canvas is cleared due to error
                this.showEmptyState(true);
            }
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * New: Checkpoint mode toggle event handler
     * @param {boolean} useCheckpoint - Whether to use checkpoint
     */
    onCheckpointModeChange(useCheckpoint) {
        this.uiController.showCheckpointSelection(useCheckpoint);

        if (useCheckpoint) {
            this.uiController.logMessage('Checkpoint mode enabled', LOG_TYPES.INFO);
        } else {
            this.uiController.logMessage('Checkpoint mode disabled', LOG_TYPES.INFO);
            this.uiController.hideCheckpointInfo();
        }
    }

    /**
     * New: Checkpoint selection change event handler
     * @param {string} checkpointName - Selected checkpoint name
     */
    async onCheckpointSelectionChange(checkpointName) {
        if (!checkpointName) {
            this.uiController.hideCheckpointInfo();
            return;
        }

        try {
            this.uiController.showLoading(true);

            // Get checkpoint detailed information
            const response = await this.apiClient.getCheckpointInfo(checkpointName);

            if (response.success && response.checkpoint_info) {
                this.uiController.showCheckpointInfo(response.checkpoint_info);
                this.uiController.logMessage(`Selected checkpoint: ${checkpointName}`, LOG_TYPES.INFO);
            } else {
                this.uiController.showError('Failed to get checkpoint info');
            }

        } catch (error) {
            console.error('Failed to get checkpoint info:', error);
            this.uiController.showError('Failed to get checkpoint info: ' + error.message);
            this.uiController.hideCheckpointInfo();
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * Handle Canvas click event
     * @param {MouseEvent} event - Mouse event
     */
    handleCanvasClick(event) {
        if (!this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) {
            this.uiController.updateClickCoordinates(null);
            return;
        }

        // Get mouse position relative to canvas
        const rect = event.target.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;

        // Convert to world coordinates
        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);

        // Update display
        this.uiController.updateClickCoordinates(worldCoords);

        // Log to message log
        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
        this.uiController.logMessage(`Click coordinates: ${coordText}`, LOG_TYPES.INFO);
    }

    /**
     * Start training - Fixed version with checkpoint support and immediate status retrieval
     */
    async startTraining() {
        // Validate configuration
        const validation = this.uiController.validateTrainingConfig();
        if (!validation.valid) {
            this.uiController.showError(validation.message);
            return;
        }

        const config = this.uiController.getTrainingConfig();

        // Add debug logs
        console.log('=== Sending training request ===');
        console.log('Complete configuration:', config);
        console.log('checkpoint_name:', config.checkpoint_name);
        console.log('====================');

        try {
            this.uiController.showLoading(true);

            const result = await this.apiClient.startTraining(config);

            let successMessage = 'Training started: ' + result.message;
            if (result.from_checkpoint && result.checkpoint_name) {
                successMessage += ` (continued from checkpoint: ${result.checkpoint_name})`;
            }

            this.uiController.logMessage(successMessage, LOG_TYPES.SUCCESS);

            this.isTraining = true;
            this.uiController.updateButtonStates(true);
            this.uiController.updateStatusIndicator(STATUS.RUNNING);
            
            // Reset estimation tracking when training starts
            this.uiController.resetEstimationTracking();

            this.startPeriodicUpdate();
            this.scheduleImmediateUpdate();

        } catch (error) {
            console.error('Failed to start training:', error);
            this.uiController.showError('Failed to start training: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * New: Schedule immediate status update
     */
    scheduleImmediateUpdate() {
        // Clear previous immediate update timer
        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
        }

        // Get status immediately after 500ms, then again at 1s and 2s
        const immediateUpdates = [500, 1000, 2000];

        immediateUpdates.forEach((delay, index) => {
            setTimeout(async () => {
                if (this.isTraining) {
                    await this.updateTrainingStatus();
                    this.uiController.logMessage(`Training status update #${index + 1}`, LOG_TYPES.INFO);
                }
            }, delay);
        });
    }

    /**
     * Stop training
     */
    async stopTraining() {
        // Immediately stop polling and update UI
        this.stopPeriodicUpdate();
        this.isTraining = false;
        this.uiController.updateButtonStates(false);
        this.uiController.updateStatusIndicator(STATUS.STOPPING);

        // Clear immediate update timer
        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
        }

        try {
            this.uiController.showLoading(true);

            const result = await this.apiClient.stopTraining();
            this.uiController.logMessage('Training stop request sent: ' + result.message, LOG_TYPES.INFO);

        } catch (error) {
            console.error('Failed to stop training:', error);
            this.uiController.showError('Failed to stop training: ' + error.message);
        } finally {
            this.uiController.showLoading(false);
        }
    }

    /**
     * Refresh training status
     */
    async refreshStatus() {
        // When user manually refreshes status, we need to check if this is a mid-training scenario
        try {
            const status = await this.apiClient.getTrainingStatus();
            
            // If training is running and we don't have any progress history, this might be a mid-training refresh
            if (status.running && status.progress && status.progress.total_steps > 0) {
                // Check if we need to initialize timing for mid-training scenario
                if (this.uiController.progressHistory.length === 0 && this.uiController.trainingStartTime === null) {
                    // This is likely a mid-training refresh scenario
                    this.uiController.adjustForMidTrainingRefresh(status.progress.total_steps, Date.now());
                    this.uiController.logMessage('Detected mid-training refresh - adjusting time estimation', LOG_TYPES.INFO);
                }
                
                // Update training state
                this.isTraining = true;
                this.uiController.updateButtonStates(true);
                
                // Start periodic updates if not already running
                if (!this.updateInterval) {
                    this.startPeriodicUpdate();
                }
            }
            
            // Handle the status update normally
            this.handleStatusUpdate(status);
            
        } catch (error) {
            console.error('Failed to refresh training status:', error);
            this.uiController.logMessage('Failed to refresh training status: ' + error.message, LOG_TYPES.ERROR);
        }
    }

    /**
     * Start periodic updates
     */
    startPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        const interval = this.uiController.getUpdateInterval();

        this.updateInterval = setInterval(async () => {
            await this.updateTrainingStatus();
        }, interval);

        // Execute one update immediately
        this.updateTrainingStatus();
    }

    /**
     * Stop periodic updates
     */
    stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }

        if (this.immediateUpdateTimer) {
            clearTimeout(this.immediateUpdateTimer);
            this.immediateUpdateTimer = null;
        }
    }

    /**
     * Update training status
     */
    async updateTrainingStatus() {
        try {
            const status = await this.apiClient.getTrainingStatus();
            this.handleStatusUpdate(status);
        } catch (error) {
            console.error('Failed to get training status:', error);
            this.uiController.logMessage('Failed to get training status: ' + error.message, LOG_TYPES.ERROR);
        }
    }

    /**
     * Handle status update
     * @param {Object} status - Status data
     */
    handleStatusUpdate(status) {
        // Update running status
        this.isTraining = status.running;

        // Update status indicator
        this.uiController.updateStatusIndicator(status.status);

        // Update statistics
        if (status.stats) {
            this.uiController.updateTrainingStats(status.stats);
        }

        // Update progress information
        if (status.progress) {
            this.uiController.updateProgressInfo(status.progress);
        }

        // Update rendering
        this.updateRendering();

        // If training has clearly ended, stop periodic updates
        const isFinished = !status.running || [STATUS.STOPPED, STATUS.COMPLETED, STATUS.ERROR].includes(status.status);
        if (isFinished && this.updateInterval) {
            this.stopPeriodicUpdate();
        }

        this.uiController.updateButtonStates(this.isTraining);
    }

    /**
     * Update rendering
     */
    updateRendering() {
        if (!this.canvasRenderer) return;

        const renderData = this.uiController.getRenderData();

        if (renderData.meshData || renderData.boundaryData) {
            this.canvasRenderer.renderScene(
                renderData.meshData,
                renderData.boundaryData,
                renderData.refPointInfo,
                renderData.actionAttempted
            );
            // Hide empty state when mesh data is being rendered
            this.showEmptyState(false);
        } else {
            // Show empty state when no data to render
            this.showEmptyState(true);
        }
    }

    /**
     * Throttled version of Canvas click event handler
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);

    /**
     * Handle window resize - New method
     */
    handleResize() {
        if (this.canvasRenderer) {
            this.canvasRenderer.onResize();
        }
        this.uiController.logMessage('Window resized', LOG_TYPES.INFO);
    }

    /**
     * New: Refresh checkpoint list
     */
    async refreshCheckpointList() {
        await this.loadCheckpointList();
    }

    /**
     * Get application state
     * @returns {Object} Application state
     */
    getApplicationState() {
        return {
            isTraining: this.isTraining,
            hasUpdateInterval: !!this.updateInterval,
            canvasReady: !!this.canvasRenderer,
            uiReady: !!this.uiController
        };
    }

    /**
     * Destroy manager and clean up resources
     */
    destroy() {
        this.stopPeriodicUpdate();

        if (this.canvasRenderer) {
            this.canvasRenderer.destroy();
        }

        this.uiController.reset();

        console.log('TrainingManager destroyed');
    }
}