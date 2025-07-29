/**
 * Action Tester Module
 * Interactive tool for testing RL actions on mesh boundaries
 */

import {CONSTANTS, STATUS, LOG_TYPES, formatNumber, throttle} from './utils.js';
import {ApiClient, withErrorHandling, withRetry} from './api-client.js';
import {CanvasRenderer} from './canvas-renderer.js';
import {UIController} from './ui-controller.js';

export class ActionTester {
    constructor() {
        // Initialize all modules
        this.apiClient = new ApiClient();
        this.uiController = new UIController();
        this.canvasRenderer = null;

        // State management
        this.currentMesh = null;
        this.currentBoundary = null;
        this.referencePoint = null;
        this.selectedAction = null;
        this.clickedPoint = null;
        this.isType1Mode = false;

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
            } else {
                this.logMessage('Cannot connect to backend server. Ensure the Flask app is running at http://localhost:5000', LOG_TYPES.ERROR);
            }

            this.logMessage('Action Tester initialized successfully', LOG_TYPES.SUCCESS);
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('System initialization failed: ' + error.message);
        }
    }

    /**
     * Setup Canvas
     */
    setupCanvas() {
        const canvas = document.getElementById('mesh-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
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
        // Mesh selection
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', (e) => this.onMeshSelectionChange(e.target.value));
        }

        // Find reference point button
        const findRefPointBtn = document.getElementById('find-ref-point-btn');
        if (findRefPointBtn) {
            findRefPointBtn.addEventListener('click', () => this.findReferencePoint());
        }

        // Action buttons
        const actionType0Left = document.getElementById('action-type0-left');
        const actionType0Right = document.getElementById('action-type0-right');
        const actionType1 = document.getElementById('action-type1');

        if (actionType0Left) {
            actionType0Left.addEventListener('click', () => this.selectAction('type0_left'));
        }
        if (actionType0Right) {
            actionType0Right.addEventListener('click', () => this.selectAction('type0_right'));
        }
        if (actionType1) {
            actionType1.addEventListener('click', () => this.selectAction('type1'));
        }

        // Execute button
        const executeBtn = document.getElementById('execute-btn');
        if (executeBtn) {
            executeBtn.addEventListener('click', () => this.executeAction());
        }

        // Clear log button
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.clearLogs());
        }

        // Canvas click event
        const canvas = document.getElementById('mesh-canvas');
        if (canvas) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }
    }

    /**
     * Check backend connection status
     */
    async checkBackendConnection() {
        try {
            const connected = await this.apiClient.checkConnection();
            if (connected) {
                this.logMessage('Backend connection successful', LOG_TYPES.SUCCESS);
            }
            return connected;
        } catch (error) {
            this.logMessage('Backend connection failed: ' + error.message, LOG_TYPES.ERROR);
            return false;
        }
    }

    /**
     * Load available mesh list
     */
    async loadMeshList() {
        try {
            this.showLoading(true);

            const data = await withRetry(() => this.apiClient.getMeshList());
            this.populateMeshList(data.meshes || []);

            if (data.meshes && data.meshes.length > 0) {
                this.logMessage(`Loaded ${data.meshes.length} mesh files`, LOG_TYPES.SUCCESS);
            } else {
                this.logMessage('No mesh files found', LOG_TYPES.WARNING);
            }

        } catch (error) {
            console.error('Failed to load mesh list:', error);
            this.showError('Failed to load mesh list: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Populate mesh selection dropdown
     */
    populateMeshList(meshes) {
        const meshSelect = document.getElementById('mesh-select');
        if (!meshSelect) return;

        meshSelect.innerHTML = '<option value="">Select a mesh...</option>';
        
        if (Array.isArray(meshes) && meshes.length > 0) {
            meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh;
                option.textContent = mesh;
                meshSelect.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No meshes available';
            meshSelect.appendChild(option);
        }
    }

    /**
     * Handle mesh selection change
     */
    async onMeshSelectionChange(meshName) {
        if (!meshName) {
            this.resetState();
            return;
        }

        try {
            this.showLoading(true);

            // Get mesh info and boundary data
            const [info, boundaryData] = await Promise.all([
                this.apiClient.getMeshInfo(meshName),
                this.apiClient.getMeshBoundary(meshName)
            ]);

            // Update mesh info
            this.showMeshInfo(info);
            this.currentMesh = meshName;
            
            // Update status
            this.updateStatusValue('mesh-status', 'Loaded');
            this.updateStatusValue('boundary-vertices-count', boundaryData.vertex_count || 0);

            // Render boundary preview
            if (this.canvasRenderer && boundaryData.success) {
                this.currentBoundary = boundaryData.boundary_vertices;
                this.canvasRenderer.renderBoundaryPreview(
                    boundaryData.boundary_vertices,
                    meshName
                );
                this.showEmptyState(false);
                this.logMessage(`Loaded mesh: ${meshName} with ${boundaryData.vertex_count} boundary vertices`, LOG_TYPES.SUCCESS);
                
                // Enable find reference point button
                const findRefPointBtn = document.getElementById('find-ref-point-btn');
                if (findRefPointBtn) {
                    findRefPointBtn.disabled = false;
                }
            } else {
                this.logMessage(`Failed to load boundary data: ${boundaryData.error}`, LOG_TYPES.ERROR);
            }

        } catch (error) {
            console.error('Failed to get mesh info:', error);
            this.showError('Failed to get mesh info: ' + error.message);
            this.resetState();
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Find reference point
     */
    async findReferencePoint() {
        if (!this.currentMesh) {
            this.showError('Please select a mesh first');
            return;
        }

        try {
            this.showLoading(true);

            // Call backend API to find reference point
            const response = await this.apiClient.findReferencePoint(this.currentMesh);

            if (response.success) {
                this.referencePoint = response.reference_point;
                this.showReferencePointInfo(response.reference_point);
                this.updateCanvas();
                
                // Update status
                this.updateStatusValue('ref-point-status', `Index ${response.reference_point.index}`);
                
                // Show action buttons
                this.showActionButtons();
                
                this.logMessage(`Reference point found at index ${response.reference_point.index}`, LOG_TYPES.SUCCESS);
            } else {
                this.showError('Failed to find reference point: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to find reference point:', error);
            this.showError('Failed to find reference point: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Select action type
     */
    selectAction(actionType) {
        this.selectedAction = actionType;
        this.clickedPoint = null;
        this.isType1Mode = actionType === 'type1';
        
        // Update UI
        this.showActionSelectedInfo(actionType);
        this.updateStatusValue('action-status', `Selected: ${actionType}`);
        
        // Show type1 instruction if needed
        const type1Instruction = document.getElementById('type1-instruction');
        if (type1Instruction) {
            type1Instruction.classList.toggle('hidden', !this.isType1Mode);
        }
        
        // Enable execute button for type0 actions
        const executeBtn = document.getElementById('execute-btn');
        if (executeBtn) {
            executeBtn.disabled = this.isType1Mode; // Type1 needs click first
        }
        
        this.logMessage(`Selected action: ${actionType}`, LOG_TYPES.INFO);
    }

    /**
     * Handle canvas click
     */
    handleCanvasClick(event) {
        if (!this.isType1Mode || !this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) return;

        // Get mouse position relative to canvas with improved precision
        const rect = event.target.getBoundingClientRect();
        
        // Consider device pixel ratio for high DPI displays
        const devicePixelRatio = window.devicePixelRatio || 1;
        
        // Calculate screen coordinates relative to canvas, accounting for any scaling
        const screenX = (event.clientX - rect.left) * devicePixelRatio / devicePixelRatio;
        const screenY = (event.clientY - rect.top) * devicePixelRatio / devicePixelRatio;

        // Convert to world coordinates
        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);
        this.clickedPoint = worldCoords;

        // Update UI
        this.showClickCoordinates(worldCoords);
        
        // Enable execute button
        const executeBtn = document.getElementById('execute-btn');
        if (executeBtn) {
            executeBtn.disabled = false;
        }

        // Update status
        this.updateStatusValue('action-status', `Type1 - Point clicked`);
        this.updateStatusValue('action-status-compact', `Point Clicked`);

        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
        this.logMessage(`Type1 action: Clicked at ${coordText}`, LOG_TYPES.INFO);

        // Re-render canvas with clicked point
        this.updateCanvasWithClickedPoint();
    }

    /**
     * Update canvas to show clicked point
     */
    updateCanvasWithClickedPoint() {
        if (!this.canvasRenderer || !this.currentBoundary) return;

        // Create reference point info for rendering
        const refPointInfo = this.referencePoint ? {
            ref_vertex: this.referencePoint.coordinates,
            local_env_vertices: this.referencePoint.neighbor_vertices,
            clicked_point: this.clickedPoint // Add clicked point
        } : null;

        this.canvasRenderer.renderScene(
            null, // No mesh data for now
            this.currentBoundary,
            refPointInfo
        );
    }

    /**
     * Update canvas with execution result
     */
    updateCanvasWithExecutionResult(result) {
        if (!this.canvasRenderer || !this.currentBoundary) return;

        // Create reference point info for rendering
        const refPointInfo = this.referencePoint ? {
            ref_vertex: this.referencePoint.coordinates,
            local_env_vertices: this.referencePoint.neighbor_vertices,
            clicked_point: this.clickedPoint,
            new_element: result.generated_element // Use the actual element from backend!
        } : null;

        this.canvasRenderer.renderScene(
            null, // No mesh data for now
            this.currentBoundary,
            refPointInfo
        );
    }

    /**
     * Execute the selected action
     */
    async executeAction() {
        if (!this.selectedAction || !this.referencePoint) {
            this.showError('Please select an action and reference point first');
            return;
        }

        if (this.selectedAction === 'type1' && !this.clickedPoint) {
            this.showError('Please click on the canvas to place a point for Type1 action');
            return;
        }

        try {
            this.showLoading(true);

            const requestData = {
                mesh_name: this.currentMesh,
                action_type: this.selectedAction,
                reference_point_index: this.referencePoint.index,
                clicked_point: this.clickedPoint
            };

            const response = await this.apiClient.executeAction(requestData);

            if (response.success) {
                this.showExecutionResult(response.result);
                this.logMessage(`Action executed: ${this.selectedAction} - Valid: ${response.result.valid}`, 
                    response.result.valid ? LOG_TYPES.SUCCESS : LOG_TYPES.WARNING);
                
                // Update canvas with execution result
                this.updateCanvasWithExecutionResult(response.result);
            } else {
                this.showError('Failed to execute action: ' + response.error);
            }

        } catch (error) {
            console.error('Failed to execute action:', error);
            this.showError('Failed to execute action: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Show mesh information
     */
    showMeshInfo(info) {
        // Show in the right sidebar detail card
        const meshInfoCard = document.getElementById('mesh-info-card');
        const meshVerticesDetail = document.getElementById('mesh-vertices-detail');
        const meshSizeDetail = document.getElementById('mesh-size-detail');

        if (meshInfoCard) meshInfoCard.classList.remove('hidden');
        if (meshVerticesDetail) meshVerticesDetail.textContent = info.vertex_count || 0;
        if (meshSizeDetail) meshSizeDetail.textContent = info.file_size || 0;
    }

    /**
     * Show reference point information
     */
    showReferencePointInfo(refPoint) {
        // Show in the right sidebar detail card
        const refPointInfoCard = document.getElementById('ref-point-info-card');
        const refPointIndexDetail = document.getElementById('ref-point-index-detail');
        const refPointCoordsDetail = document.getElementById('ref-point-coords-detail');
        const refPointAngleDetail = document.getElementById('ref-point-angle-detail');

        if (refPointInfoCard) refPointInfoCard.classList.remove('hidden');
        if (refPointIndexDetail) refPointIndexDetail.textContent = refPoint.index;
        if (refPointCoordsDetail) refPointCoordsDetail.textContent = `(${refPoint.coordinates[0].toFixed(3)}, ${refPoint.coordinates[1].toFixed(3)})`;
        if (refPointAngleDetail) refPointAngleDetail.textContent = refPoint.interior_angle.toFixed(1);
    }

    /**
     * Show action buttons
     */
    showActionButtons() {
        const actionButtons = document.getElementById('action-buttons');
        if (actionButtons) {
            actionButtons.classList.remove('hidden');
        }
    }

    /**
     * Show action selected information
     */
    showActionSelectedInfo(actionType) {
        // Show in the right sidebar detail card
        const actionSelectedInfoCard = document.getElementById('action-selected-info-card');
        const selectedActionTypeDetail = document.getElementById('selected-action-type-detail');

        if (actionSelectedInfoCard) actionSelectedInfoCard.classList.remove('hidden');
        if (selectedActionTypeDetail) selectedActionTypeDetail.textContent = actionType;
    }

    /**
     * Show click coordinates
     */
    showClickCoordinates(coords) {
        // Show in the right sidebar detail card
        const clickCoordsInfoDetail = document.getElementById('click-coords-info-detail');
        const clickCoordinatesDetail = document.getElementById('click-coordinates-detail');

        if (clickCoordsInfoDetail) clickCoordsInfoDetail.classList.remove('hidden');
        if (clickCoordinatesDetail) clickCoordinatesDetail.textContent = `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`;
    }

    /**
     * Show execution result
     */
    showExecutionResult(result) {
        // Show in the right sidebar detail card
        const executionResultCard = document.getElementById('execution-result-card');
        const resultIsValidDetail = document.getElementById('result-is-valid-detail');
        const resultPolarCoordsDetail = document.getElementById('result-polar-coords-detail');
        const polarCoordsValueDetail = document.getElementById('polar-coords-value-detail');

        if (executionResultCard) {
            executionResultCard.classList.remove('hidden');
            // Add color coding for the card
            executionResultCard.className = `detail-card ${result.valid ? 'border-green-500' : 'border-red-500'}`;
        }
        if (resultIsValidDetail) resultIsValidDetail.textContent = result.valid ? 'Yes' : 'No';
        
        // Show polar coordinates for type1 actions
        if (this.selectedAction === 'type1' && result.polar_coordinates) {
            if (resultPolarCoordsDetail) resultPolarCoordsDetail.classList.remove('hidden');
            if (polarCoordsValueDetail) {
                const polar = result.polar_coordinates;
                polarCoordsValueDetail.textContent = `r=${polar.r.toFixed(3)}, θ=${polar.theta.toFixed(3)}`;
            }
        }
    }

    /**
     * Update canvas rendering
     */
    updateCanvas() {
        if (!this.canvasRenderer || !this.currentBoundary) return;

        // Create reference point info for rendering
        const refPointInfo = this.referencePoint ? {
            ref_vertex: this.referencePoint.coordinates,
            local_env_vertices: this.referencePoint.neighbor_vertices
        } : null;

        this.canvasRenderer.renderScene(
            null, // No mesh data for now
            this.currentBoundary,
            refPointInfo
        );
    }

    /**
     * Update status value
     */
    updateStatusValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
        
        // Also update compact status if applicable
        if (elementId === 'mesh-status') {
            const compactElement = document.getElementById('mesh-status-compact');
            if (compactElement) compactElement.textContent = value;
        } else if (elementId === 'ref-point-status') {
            const compactElement = document.getElementById('ref-point-status-compact');
            if (compactElement) compactElement.textContent = value;
        } else if (elementId === 'action-status') {
            const compactElement = document.getElementById('action-status-compact');
            if (compactElement) compactElement.textContent = value;
        }
    }

    /**
     * Reset application state
     */
    resetState() {
        this.currentMesh = null;
        this.currentBoundary = null;
        this.referencePoint = null;
        this.selectedAction = null;
        this.clickedPoint = null;
        this.isType1Mode = false;

        // Reset UI
        this.hideMeshInfo();
        this.hideReferencePointInfo();
        this.hideActionButtons();
        this.hideActionSelectedInfo();
        this.hideExecutionResult();

        // Reset status
        this.updateStatusValue('mesh-status', 'Not Selected');
        this.updateStatusValue('boundary-vertices-count', '0');
        this.updateStatusValue('ref-point-status', 'Not Selected');
        this.updateStatusValue('action-status', 'No Action');
        
        // Reset compact status
        this.updateStatusValue('mesh-status-compact', 'Not Selected');
        this.updateStatusValue('ref-point-status-compact', 'No Reference');
        this.updateStatusValue('action-status-compact', 'No Action');

        // Disable buttons
        const findRefPointBtn = document.getElementById('find-ref-point-btn');
        const executeBtn = document.getElementById('execute-btn');
        if (findRefPointBtn) findRefPointBtn.disabled = true;
        if (executeBtn) executeBtn.disabled = true;

        // Clear canvas
        if (this.canvasRenderer) {
            this.canvasRenderer.clearCanvas();
            this.showEmptyState(true);
        }
    }

    /**
     * Hide UI elements
     */
    hideMeshInfo() {
        const meshInfoCard = document.getElementById('mesh-info-card');
        if (meshInfoCard) meshInfoCard.classList.add('hidden');
    }

    hideReferencePointInfo() {
        const refPointInfoCard = document.getElementById('ref-point-info-card');
        if (refPointInfoCard) refPointInfoCard.classList.add('hidden');
    }

    hideActionButtons() {
        const actionButtons = document.getElementById('action-buttons');
        if (actionButtons) actionButtons.classList.add('hidden');
    }

    hideActionSelectedInfo() {
        const actionSelectedInfoCard = document.getElementById('action-selected-info-card');
        if (actionSelectedInfoCard) actionSelectedInfoCard.classList.add('hidden');
    }

    hideExecutionResult() {
        const executionResultCard = document.getElementById('execution-result-card');
        if (executionResultCard) executionResultCard.classList.add('hidden');
    }

    /**
     * Utility methods
     */
    showLoading(show = true) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.toggle('hidden', !show);
        }
    }

    showError(message) {
        this.logMessage(message, LOG_TYPES.ERROR);
        alert(message); // Simple error display for now
    }

    logMessage(message, type = LOG_TYPES.INFO) {
        const logContainer = document.getElementById('log-container');
        if (!logContainer) return;

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        logContainer.appendChild(logEntry);
        
        // Auto-scroll if enabled
        const autoScrollCheckbox = document.getElementById('auto-scroll-checkbox');
        if (autoScrollCheckbox && autoScrollCheckbox.checked) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }

    clearLogs() {
        const logContainer = document.getElementById('log-container');
        if (logContainer) {
            logContainer.innerHTML = '';
            this.logMessage('Action Tester logs cleared', LOG_TYPES.INFO);
        }
    }

    /**
     * Throttled version of Canvas click event handler
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);
}