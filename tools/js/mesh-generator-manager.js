/**
 * Mesh Generator Manager
 * Handles the mesh generation interface with prediction API integration
 */

import {CONSTANTS, formatNumber, throttle} from './utils.js';
import {CanvasRenderer} from './canvas-renderer.js';

export class MeshGeneratorManager {
    constructor() {
        // Core state
        this.sessionId = null;
        this.isSessionActive = false;
        this.components = null;
        this.currentStep = 0;
        this.actionStats = {
            totalAttempts: 0,
            successfulActions: 0,
            failedActions: 0,
            actionTypeCounts: {
                type0_left: { attempts: 0, successes: 0 },
                type0_right: { attempts: 0, successes: 0 },
                type1: { attempts: 0, successes: 0 }
            }
        };
        
        // Canvas renderer
        this.canvasRenderer = null;
        
        // Last action info for visualization
        this.lastActionInfo = null;
        this.lastGeneratedElement = null;
        this.currentReferencePoint = null;
        this.lastInvalidAction = null;
        
        // API client setup
        this.apiBaseUrl = 'http://127.0.0.1:5000/predict';
        
        this.init();
    }

    /**
     * Initialize the mesh generator
     */
    async init() {
        try {
            this.setupCanvas();
            this.bindEvents();
            
            // Load components from API
            await this.loadComponents();
            
            this.logMessage('Mesh Generator initialized successfully', 'info');
        } catch (error) {
            console.error('Failed to initialize Mesh Generator:', error);
            this.showError('Failed to initialize: ' + error.message);
        }
    }

    /**
     * Setup canvas renderer
     */
    setupCanvas() {
        const canvas = document.getElementById('mesh-generator-canvas');
        if (canvas) {
            this.canvasRenderer = new CanvasRenderer(canvas);
            this.showEmptyState(true);
        } else {
            console.error('Canvas element not found');
        }
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        // Mesh selection
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect) {
            meshSelect.addEventListener('change', (e) => this.onMeshChange(e.target.value));
        }

        // Predictor selection
        const predictorSelect = document.getElementById('predictor-select');
        if (predictorSelect) {
            predictorSelect.addEventListener('change', (e) => this.onPredictorChange(e.target.value));
        }

        // Reference selector selection
        const refSelectorSelect = document.getElementById('ref-selector-select');
        if (refSelectorSelect) {
            refSelectorSelect.addEventListener('change', (e) => this.onRefSelectorChange(e.target.value));
        }

        // Reselect reference point button
        const reselectBtn = document.getElementById('reselect-ref-point-btn');
        if (reselectBtn) {
            reselectBtn.addEventListener('click', () => this.reselectReferencePoint());
        }

        // Session controls
        this.bindSessionControls();

        // Canvas interaction
        const canvas = document.getElementById('mesh-generator-canvas');
        if (canvas) {
            canvas.addEventListener('click', this.handleCanvasClickThrottled);
        }

        // Clear log
        const clearLogBtn = document.getElementById('clear-log-btn');
        if (clearLogBtn) {
            clearLogBtn.addEventListener('click', () => this.clearLog());
        }
    }

    /**
     * Bind session control events
     */
    bindSessionControls() {
        const createSessionBtn = document.getElementById('create-session-btn');
        const nextStepBtn = document.getElementById('next-step-btn');
        const prevStepBtn = document.getElementById('prev-step-btn');
        const processAllBtn = document.getElementById('process-all-btn');
        const resetSessionBtn = document.getElementById('reset-session-btn');
        const deleteSessionBtn = document.getElementById('delete-session-btn');

        if (createSessionBtn) {
            createSessionBtn.addEventListener('click', () => this.createSession());
        }
        if (nextStepBtn) {
            nextStepBtn.addEventListener('click', () => this.executeNextStep());
        }
        if (prevStepBtn) {
            prevStepBtn.addEventListener('click', () => this.executePreviousStep());
        }
        if (processAllBtn) {
            processAllBtn.addEventListener('click', () => this.processAllSteps());
        }
        if (resetSessionBtn) {
            resetSessionBtn.addEventListener('click', () => this.resetSession());
        }
        if (deleteSessionBtn) {
            deleteSessionBtn.addEventListener('click', () => this.deleteSession());
        }
    }

    /**
     * Load available components from API
     */
    async loadComponents() {
        try {
            this.showLoading(true);
            
            const response = await this.apiRequest('/components', 'GET');
            
            this.components = response;
            
            this.populateComponentSelectors();
            this.logMessage('Components loaded successfully', 'success');
            
        } catch (error) {
            console.error('Failed to load components:', error);
            this.showError('Failed to load components: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Populate component selector dropdowns
     */
    populateComponentSelectors() {
        if (!this.components) return;

        // Populate mesh selector
        const meshSelect = document.getElementById('mesh-select');
        if (meshSelect && this.components.initial_meshes) {
            meshSelect.innerHTML = '<option value="">Select a mesh...</option>';
            this.components.initial_meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh;
                option.textContent = mesh;
                meshSelect.appendChild(option);
            });
        }

        // Populate predictor selector
        const predictorSelect = document.getElementById('predictor-select');
        if (predictorSelect && this.components.predictors) {
            predictorSelect.innerHTML = '<option value="">Select a predictor...</option>';
            Object.keys(this.components.predictors).forEach(key => {
                const predictor = this.components.predictors[key];
                const option = document.createElement('option');
                option.value = key;
                // Truncate long descriptions for better display
                const shortDesc = predictor.description.length > 40 ? 
                    predictor.description.substring(0, 40) + '...' : 
                    predictor.description;
                option.textContent = `${predictor.name} - ${shortDesc}`;
                option.title = `${predictor.name} - ${predictor.description}`; // Full text in tooltip
                predictorSelect.appendChild(option);
            });
        }

        // Populate reference selector
        const refSelectorSelect = document.getElementById('ref-selector-select');
        if (refSelectorSelect && this.components.reference_selectors) {
            refSelectorSelect.innerHTML = '<option value="">Select a reference selector...</option>';
            Object.keys(this.components.reference_selectors).forEach(key => {
                const selector = this.components.reference_selectors[key];
                const option = document.createElement('option');
                option.value = key;
                option.textContent = selector.name;
                option.title = selector.description; // Full description in tooltip
                refSelectorSelect.appendChild(option);
            });
        }

        // Populate model selector
        this.populateModelSelector();
    }

    /**
     * Populate model selector dropdown
     */
    populateModelSelector() {
        const modelSelect = document.getElementById('model-select');
        if (modelSelect && this.components && this.components.trained_models) {
            modelSelect.innerHTML = '<option value="">Select a trained model...</option>';
            this.components.trained_models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = `${model.name} (${this.formatFileSize(model.size)})`;
                modelSelect.appendChild(option);
            });
        }
    }

    /**
     * Handle mesh selection change
     */
    async onMeshChange(meshName) {
        if (!meshName) {
            this.hideMeshInfo();
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
                this.showEmptyState(true);
            }
            return;
        }

        try {
            this.showLoading(true);
            
            // Get mesh info (using training API since predict API doesn't have mesh info endpoint)
            const meshInfo = await this.trainingApiRequest(`/mesh/info/${meshName}`, 'GET');
            
            this.showMeshInfo(meshInfo);
            this.logMessage(`Selected mesh: ${meshName}`, 'info');
            
            // Load mesh boundary for preview
            await this.loadMeshPreview(meshName);
            
        } catch (error) {
            console.error('Failed to load mesh info:', error);
            this.showError('Failed to load mesh info: ' + error.message);
        } finally {
            this.showLoading(false);
        }
        
        this.validateConfiguration();
    }

    /**
     * Load mesh preview
     */
    async loadMeshPreview(meshName) {
        try {
            const boundaryData = await this.trainingApiRequest(`/mesh/boundary/${meshName}`, 'GET');
            
            if (boundaryData.success && this.canvasRenderer) {
                this.canvasRenderer.renderBoundaryPreview(
                    boundaryData.boundary_vertices,
                    meshName
                );
                this.showEmptyState(false);
                this.logMessage(`Loaded boundary preview: ${boundaryData.vertex_count} vertices`, 'success');
            }
        } catch (error) {
            console.error('Failed to load mesh preview:', error);
            this.logMessage('Failed to load mesh preview: ' + error.message, 'warning');
        }
    }

    /**
     * Handle predictor selection change
     */
    onPredictorChange(predictorType) {
        const configDiv = document.getElementById('predictor-config');
        
        if (!predictorType) {
            configDiv.classList.add('hidden');
            return;
        }
        
        configDiv.classList.remove('hidden');
        this.logMessage(`Selected predictor: ${predictorType}`, 'info');
        this.validateConfiguration();
    }

    /**
     * Handle reference selector change
     */
    async onRefSelectorChange(selectorType) {
        const configDiv = document.getElementById('ref-selector-config');
        
        if (!selectorType || selectorType === 'default') {
            configDiv.classList.add('hidden');
        } else {
            configDiv.classList.remove('hidden');
        }
        
        if (selectorType) {
            this.logMessage(`Selected reference selector: ${selectorType}`, 'info');

            // If the user is changing the selector, it's often to fix an invalid action.
            // Clear the invalid action flag to re-enable the "Next" button immediately.
            if (this.lastInvalidAction) {
                this.logMessage('Invalid action state cleared by changing reference selector.', 'info');
                this.lastInvalidAction = null;
            }
            
            if (this.isSessionActive) {
                // If session is active, update the session config and re-fetch the reference point
                await this.updateSessionRefSelector(selectorType);
            } else {
                // Otherwise, just preview the reference point on the selected mesh
                const meshName = document.getElementById('mesh-select').value;
                if (meshName) {
                    await this.previewReferencePoint();
                }
            }
        }
        
        this.validateConfiguration();
    }

    /**
     * Validate configuration and enable/disable create session button
     */
    validateConfiguration() {
        const meshSelect = document.getElementById('mesh-select');
        const predictorSelect = document.getElementById('predictor-select');
        const refSelectorSelect = document.getElementById('ref-selector-select');
        const modelSelect = document.getElementById('model-select');
        const createSessionBtn = document.getElementById('create-session-btn');

        const isValid = meshSelect.value && 
                       predictorSelect.value && 
                       refSelectorSelect.value && 
                       (predictorSelect.value !== 'RL' || modelSelect.value);

        if (createSessionBtn) {
            createSessionBtn.disabled = !isValid || this.isSessionActive;
        }
    }

    /**
     * Create prediction session
     */
    async createSession() {
        try {
            this.showLoading(true);
            
            const config = this.getSessionConfig();
            const response = await this.apiRequest('/session/create', 'POST', config);
            
            this.sessionId = response.session_id;
            this.isSessionActive = true;
            this.currentStep = 0;
            
            this.updateSessionStatus(response.initial_status);
            this.showSessionControls(true);
            this.logMessage(`Session created: ${this.sessionId}`, 'success');
            
            // Show reselect button
            this.showReselectButton(true);

            // Reset action statistics
            this.resetActionStats();
            
            // Get and display current reference point
            await this.updateCurrentReferencePoint();
            
        } catch (error) {
            console.error('Failed to create session:', error);
            this.showError('Failed to create session: ' + error.message);
        } finally {
            this.showLoading(false);
            this.validateConfiguration();
        }
    }

    /**
     * Get session configuration from form
     */
    getSessionConfig() {
        const meshName = document.getElementById('mesh-select').value;
        const predictorType = document.getElementById('predictor-select').value;
        const refSelectorType = document.getElementById('ref-selector-select').value;
        
        const config = {
            mesh_name: meshName,
            predictor_type: predictorType,
            ref_selector_type: refSelectorType
        };

        // Add predictor config
        if (predictorType === 'RL') {
            config.predictor_config = {
                model_path: document.getElementById('model-select').value,
                n: parseInt(document.getElementById('predictor-n').value) || 2,
                g: parseInt(document.getElementById('predictor-g').value) || 3,
                beta: parseInt(document.getElementById('predictor-beta').value) || 6
            };
        }

        // Add reference selector config
        if (refSelectorType === 'RL') {
            config.ref_selector_config = {
                n: parseInt(document.getElementById('ref-selector-n').value) || 2
            };
        }

        return config;
    }

    /**
     * Execute next prediction step
     */
    async executeNextStep() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            this.setButtonLoading('next-step-btn', true);
            
            const response = await this.apiRequest(`/session/${this.sessionId}/next`, 'POST');
            
            this.handleStepResult(response);
            this.logMessage('Next step executed', 'info');
            
        } catch (error) {
            console.error('Failed to execute next step:', error);
            this.showError('Failed to execute next step: ' + error.message);
        } finally {
            this.showLoading(false);
            this.setButtonLoading('next-step-btn', false);
        }
    }

    /**
     * Execute previous step (undo)
     */
    async executePreviousStep() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            this.setButtonLoading('prev-step-btn', true);
            
            const response = await this.apiRequest(`/session/${this.sessionId}/prev`, 'POST');
            
            if (response.undo_result.success) {
                this.logMessage('Previous step undone', 'success');

                // Clear stale data from the undone step to prevent re-rendering artifacts
                this.lastActionInfo = null;
                this.lastGeneratedElement = null;
                this.lastInvalidAction = null;

                // Refresh the session status, which will trigger a re-render of the mesh
                await this.refreshSessionStatus();
            } else {
                this.logMessage('Undo failed: ' + response.undo_result.message, 'warning');
            }
            
        } catch (error) {
            console.error('Failed to undo step:', error);
            this.showError('Failed to undo step: ' + error.message);
        } finally {
            this.showLoading(false);
            this.setButtonLoading('prev-step-btn', false);
            
            // Update reference point after undo
            if (this.sessionId) {
                setTimeout(() => this.updateCurrentReferencePoint(), 100);
            }
        }
    }

    /**
     * Process all remaining steps
     */
    async processAllSteps() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            this.setButtonLoading('process-all-btn', true);
            
            const response = await this.apiRequest(`/session/${this.sessionId}/process_all?max_steps=100`, 'POST');
            
            this.logMessage(`Processed ${response.steps_executed} steps`, 'success');
            
            // Handle each step result
            if (response.results) {
                response.results.forEach((result, index) => {
                    this.handleStepResult({
                        step_result: result,
                        status: index === response.results.length - 1 ? response.final_status : null
                    });
                });
            }
            
            await this.refreshSessionStatus();
            
        } catch (error) {
            console.error('Failed to process all steps:', error);
            this.showError('Failed to process all steps: ' + error.message);
        } finally {
            this.showLoading(false);
            this.setButtonLoading('process-all-btn', false);
        }
    }

    /**
     * Reset session to initial state
     */
    async resetSession() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            
            const response = await this.apiRequest(`/session/${this.sessionId}/reset`, 'POST');
            
            if (response.reset_result.success) {
                this.updateSessionStatus(response.status);
                this.resetActionStats();
                this.clearActionInfo();
                this.currentStep = 0;
                this.lastInvalidAction = null;
                
                // Clear visualization data
                this.lastActionInfo = null;
                this.lastGeneratedElement = null;
                this.currentReferencePoint = null;
                
                this.logMessage('Session reset to initial state', 'success');
                
                // Reload mesh preview and get a new reference point
                const meshName = document.getElementById('mesh-select').value;
                if (meshName) {
                    await this.loadMeshPreview(meshName);
                }
                await this.updateCurrentReferencePoint();
            }
        } catch (error) {
            console.error('Failed to reset session:', error);
            this.showError('Failed to reset session: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Delete current session
     */
    async deleteSession() {
        if (!this.sessionId) return;

        if (!confirm('Are you sure you want to delete this session?')) {
            return;
        }

        try {
            this.showLoading(true);
            
            await this.apiRequest(`/session/${this.sessionId}`, 'DELETE');
            
            this.sessionId = null;
            this.isSessionActive = false;
            this.currentStep = 0;
            this.lastInvalidAction = null;
            this.showSessionControls(false);
            this.clearSessionStatus();
            this.clearActionInfo();
            this.resetActionStats();
            this.showReselectButton(false);
            
            // Clear visualization data
            this.lastActionInfo = null;
            this.lastGeneratedElement = null;
            this.currentReferencePoint = null;
            
            this.logMessage('Session deleted successfully', 'success');
            
            // Show empty state
            if (this.canvasRenderer) {
                this.canvasRenderer.clearCanvas();
                this.showEmptyState(true);
            }
            
        } catch (error) {
            console.error('Failed to delete session:', error);
            this.showError('Failed to delete session: ' + error.message);
        } finally {
            this.showLoading(false);
            this.validateConfiguration();
        }
    }

    /**
     * Handle step execution result
     */
    handleStepResult(response) {
        const { step_result, status } = response;
        
        // Save action info and generated element for visualization
        if (step_result.action_info) {
            this.lastActionInfo = step_result.action_info;
            this.updateActionInfo(step_result.action_info);
            this.updateActionStats(step_result.action_info);
            
            // Track invalid actions for button state management
            if (!step_result.action_info.is_valid) {
                this.lastInvalidAction = step_result.action_info;
            } else {
                this.lastInvalidAction = null;
            }
            
            // Log the action attempt details
            this.logActionAttempt(step_result.action_info, step_result.success);
        }

        // Save generated element
        if (step_result.element) {
            this.lastGeneratedElement = step_result.element;
        }
        
        // Update session status and visualization
        if (status) {
            this.updateSessionStatus(status);
            
            // Update reference point after successful valid action
            if (step_result.success && step_result.action_info && step_result.action_info.is_valid) {
                setTimeout(() => this.updateCurrentReferencePoint(), 100);
            }
        }
        
        // Handle step result
        if (step_result.success) {
            this.logMessage('Step completed successfully', 'success');
            if (step_result.element) {
                this.logMessage(`Generated element with ${step_result.element.length} vertices`, 'info');
            }
        } else {
            const message = step_result.message || 'Step execution failed';
            this.logMessage(`Step failed: ${message}`, 'warning');
            
            // Show invalid action visualization
            if (step_result.action_info && !step_result.action_info.is_valid) {
                this.visualizeInvalidAction(step_result.action_info);
            }
        }
        
        // Refresh session status to get latest data
        setTimeout(() => this.refreshSessionStatus(), 200);
    }

    /**
     * Log action attempt details
     */
    logActionAttempt(actionInfo, success) {
        if (!actionInfo) return;
        
        const actionType = actionInfo.action_type;
        const refVertex = actionInfo.reference_vertex_idx;
        const coords = actionInfo.new_coords;
        const valid = actionInfo.is_valid;
        
        let message = `Action ${actionType} at ref vertex ${refVertex}`;
        
        if (actionType === 'type1' && coords && coords.length > 0) {
            const [x, y] = coords[0];
            message += ` -> new vertex (${x.toFixed(2)}, ${y.toFixed(2)})`;
        }
        
        if (!valid) {
            message += ` - INVALID: ${actionInfo.validation_message || 'Unknown error'}`;
            this.logMessage(message, 'error');
        } else if (success) {
            message += ' - SUCCESS';
            this.logMessage(message, 'success');
        } else {
            message += ' - FAILED';
            this.logMessage(message, 'warning');
        }
    }

    /**
     * Visualize invalid action attempt
     */
    visualizeInvalidAction(actionInfo) {
        if (!this.canvasRenderer) return;
        
        // Log detailed invalid action information
        this.logMessage(`Invalid action attempt details:`, 'error');
        this.logMessage(`  Type: ${actionInfo.action_type}`, 'error');
        this.logMessage(`  Reference vertex: ${actionInfo.reference_vertex_idx}`, 'error');
        
        if (actionInfo.new_coords && actionInfo.new_coords.length > 0) {
            const [x, y] = actionInfo.new_coords[0];
            this.logMessage(`  Attempted coordinates: (${x.toFixed(3)}, ${y.toFixed(3)})`, 'error');
        }
        
        if (actionInfo.validation_message) {
            this.logMessage(`  Validation error: ${actionInfo.validation_message}`, 'error');
            this.showError('Action validation failed: ' + actionInfo.validation_message, false);
        }
    }

    /**
     * Refresh session status
     */
    async refreshSessionStatus() {
        if (!this.sessionId) return;

        try {
            const response = await this.apiRequest(`/session/${this.sessionId}/status`, 'GET');
            this.updateSessionStatus(response.status);
        } catch (error) {
            console.error('Failed to refresh session status:', error);
        }
    }

    /**
     * Update session status display
     */
    updateSessionStatus(status) {
        if (!status) return;

        // Update status displays
        this.updateElement('session-id-display', this.sessionId || '-');
        this.updateElement('current-step-display', status.current_step || 0);
        this.updateElement('boundary-size-display', status.boundary_size || 0);
        this.updateElement('generated-elements-display', status.generated_elements_count || 0);
        this.updateElement('completion-status-display', status.is_completed ? 'Yes' : 'No');

        // Update step info
        this.updateElement('current-step-info', 
            `Step ${status.current_step || 0} - ${status.is_completed ? 'Completed' : 'Active'}`);

        // Update button states
        this.updateButtonStates(status);

        // Show session status panel
        const statusPanel = document.getElementById('session-status');
        if (statusPanel) {
            statusPanel.classList.remove('hidden');
        }

        // Update canvas visualization with current session data
        this.updateCanvasVisualization(status);
    }

    /**
     * Update canvas visualization with session data
     */
    updateCanvasVisualization(status) {
        if (!this.canvasRenderer || !status) return;

        try {
            // Render the mesh scene with the latest data, including the current reference point
            this.canvasRenderer.renderScene(
                status.mesh_data || null,
                status.boundary_vertices || null,
                this.currentReferencePoint // Pass the centrally managed reference point
            );
            
            // Hide empty state when we have data to render
            this.showEmptyState(false);
            
        } catch (error) {
            console.error('Failed to update canvas visualization:', error);
            this.logMessage('Failed to update visualization: ' + error.message, 'error');
        }
    }

    /**
     * Create reference point info from action info
     */
    createReferencePointInfo(actionInfo, boundaryVertices) {
        if (!actionInfo || !boundaryVertices || actionInfo.reference_vertex_idx === undefined) {
            return null;
        }

        const refVertexIdx = actionInfo.reference_vertex_idx;
        if (refVertexIdx < 0 || refVertexIdx >= boundaryVertices.length) {
            return null;
        }

        const refVertex = boundaryVertices[refVertexIdx];
        
        // Create local environment vertices (show neighboring vertices)
        const localEnvVertices = [];
        const numNeighbors = 2; // Show 2 vertices on each side
        
        for (let i = -numNeighbors; i <= numNeighbors; i++) {
            const idx = (refVertexIdx + i + boundaryVertices.length) % boundaryVertices.length;
            localEnvVertices.push(boundaryVertices[idx]);
        }

        const refPointInfo = {
            ref_vertex: refVertex,
            local_env_vertices: localEnvVertices
        };

        // Add clicked point for type1 actions
        if (actionInfo.action_type === 'type1' && actionInfo.new_coords && actionInfo.new_coords.length > 0) {
            refPointInfo.clicked_point = actionInfo.new_coords[0];
        }

        // Add new element if it was generated
        if (actionInfo.is_valid && this.lastGeneratedElement) {
            refPointInfo.new_element = this.lastGeneratedElement;
        }

        return refPointInfo;
    }

    /**
     * Update button states based on session status
     */
    updateButtonStates(status) {
        const nextBtn = document.getElementById('next-step-btn');
        const prevBtn = document.getElementById('prev-step-btn');
        const processAllBtn = document.getElementById('process-all-btn');
        
        // Update current step
        this.currentStep = status.current_step || 0;

        if (nextBtn) {
            // Disable next if session completed, OR if last action was invalid
            const disableNext = !this.isSessionActive || status.is_completed || 
                               (this.lastInvalidAction && this.currentStep > 0);
            nextBtn.disabled = disableNext;
        }
        if (prevBtn) {
            // Disable prev if no session, can't undo, OR at step 0
            const disablePrev = !this.isSessionActive || !status.can_undo || this.currentStep === 0;
            prevBtn.disabled = disablePrev;
        }
        if (processAllBtn) {
            processAllBtn.disabled = !this.isSessionActive || status.is_completed;
        }
    }

    /**
     * Update action information display
     */
    updateActionInfo(actionInfo) {
        this.updateElement('action-type-display', actionInfo.action_type || '-');
        this.updateElement('reference-vertex-display', actionInfo.reference_vertex_idx || '-');
        this.updateElement('action-status-display', actionInfo.is_valid ? 'Valid' : 'Invalid');
        
        // Add styling based on validity
        const statusDisplay = document.getElementById('action-status-display');
        if (statusDisplay) {
            statusDisplay.className = `stat-value ${actionInfo.is_valid ? 'valid' : 'invalid'}`;
        }

        // Update coordinates display
        if (actionInfo.new_coords && actionInfo.new_coords.length > 0) {
            const coords = actionInfo.new_coords[0];
            this.updateElement('new-coords-display', `(${coords[0].toFixed(3)}, ${coords[1].toFixed(3)})`);
        } else {
            this.updateElement('new-coords-display', '-');
        }

        // Show error details if invalid
        if (!actionInfo.is_valid) {
            this.showActionError(actionInfo.validation_message);
        } else {
            this.hideActionError();
        }
    }

    /**
     * Update action statistics
     */
    updateActionStats(actionInfo) {
        this.actionStats.totalAttempts++;
        
        if (actionInfo.is_valid) {
            this.actionStats.successfulActions++;
            this.actionStats.actionTypeCounts[actionInfo.action_type].successes++;
        } else {
            this.actionStats.failedActions++;
        }
        
        this.actionStats.actionTypeCounts[actionInfo.action_type].attempts++;
        
        // Update display
        this.updateElement('total-attempts-display', this.actionStats.totalAttempts);
        this.updateElement('successful-actions-display', this.actionStats.successfulActions);
        this.updateElement('failed-actions-display', this.actionStats.failedActions);
        
        const successRate = this.actionStats.totalAttempts > 0 ? 
            ((this.actionStats.successfulActions / this.actionStats.totalAttempts) * 100).toFixed(1) : 0;
        this.updateElement('success-rate-display', successRate + '%');
    }

    /**
     * Reset action statistics
     */
    resetActionStats() {
        this.actionStats = {
            totalAttempts: 0,
            successfulActions: 0,
            failedActions: 0,
            actionTypeCounts: {
                type0_left: { attempts: 0, successes: 0 },
                type0_right: { attempts: 0, successes: 0 },
                type1: { attempts: 0, successes: 0 }
            }
        };
        
        this.updateElement('total-attempts-display', 0);
        this.updateElement('successful-actions-display', 0);
        this.updateElement('failed-actions-display', 0);
        this.updateElement('success-rate-display', '0%');
    }

    /**
     * Show action error
     */
    showActionError(message) {
        const errorDisplay = document.getElementById('error-display');
        const errorMessage = document.getElementById('error-message');
        
        if (errorDisplay && errorMessage) {
            errorMessage.textContent = message || 'Unknown error';
            errorDisplay.classList.remove('hidden');
        }
    }

    /**
     * Hide action error
     */
    hideActionError() {
        const errorDisplay = document.getElementById('error-display');
        if (errorDisplay) {
            errorDisplay.classList.add('hidden');
        }
    }

    /**
     * Clear action information display
     */
    clearActionInfo() {
        this.updateElement('action-type-display', '-');
        this.updateElement('reference-vertex-display', '-');
        this.updateElement('action-status-display', '-');
        this.updateElement('new-coords-display', '-');
        this.hideActionError();
    }

    /**
     * Show/hide session controls
     */
    showSessionControls(show) {
        const controls = document.getElementById('session-controls');
        if (controls) {
            if (show) {
                controls.classList.remove('hidden');
            } else {
                controls.classList.add('hidden');
            }
        }
    }

    /**
     * Clear session status display
     */
    clearSessionStatus() {
        const statusPanel = document.getElementById('session-status');
        if (statusPanel) {
            statusPanel.classList.add('hidden');
        }
        
        this.updateElement('current-step-info', 'No Session Active');
    }

    /**
     * Handle canvas click event
     */
    handleCanvasClick(event) {
        // Canvas click handling for coordinate display
        if (!this.canvasRenderer) return;

        const transform = this.canvasRenderer.getCurrentTransform();
        if (!transform) return;

        const rect = event.target.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;

        const worldCoords = this.canvasRenderer.screenToWorld(screenX, screenY, transform);
        const coordText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
        
        this.logMessage(`Click coordinates: ${coordText}`, 'info');
    }

    /**
     * Throttled canvas click handler
     */
    handleCanvasClickThrottled = throttle((event) => {
        this.handleCanvasClick(event);
    }, 100);

    /**
     * Show/hide mesh info
     */
    showMeshInfo(info) {
        if (!info) return;

        this.updateElement('mesh-vertices', info.vertex_count || 0);
        this.updateElement('mesh-size', info.file_size || 0);

        const meshInfoDiv = document.getElementById('mesh-info');
        if (meshInfoDiv) {
            meshInfoDiv.classList.remove('hidden');
        }
    }

    /**
     * Hide mesh info
     */
    hideMeshInfo() {
        const meshInfoDiv = document.getElementById('mesh-info');
        if (meshInfoDiv) {
            meshInfoDiv.classList.add('hidden');
        }
    }

    /**
     * Show/hide empty state
     */
    showEmptyState(show) {
        const overlay = document.getElementById('empty-state-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    /**
     * Log message to operation log
     */
    logMessage(message, type = 'info') {
        const logContainer = document.getElementById('log-container');
        if (!logContainer) return;

        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;

        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;

        // Limit log entries
        const entries = logContainer.querySelectorAll('.log-entry');
        if (entries.length > 100) {
            entries[0].remove();
        }
    }

    /**
     * Clear operation log
     */
    clearLog() {
        const logContainer = document.getElementById('log-container');
        if (logContainer) {
            logContainer.innerHTML = '<div class="text-gray-500">Log cleared</div>';
        }
    }

    /**
     * Show/hide loading indicator
     */
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    /**
     * Set button loading state
     */
    setButtonLoading(buttonId, loading) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = loading;
            if (loading) {
                button.classList.add('loading');
            } else {
                button.classList.remove('loading');
            }
        }
    }

    /**
     * Show error message
     */
    showError(message, persistent = true) {
        this.logMessage(message, 'error');
        
        if (persistent) {
            alert('Error: ' + message);
        }
    }

    /**
     * Update element text content
     */
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Generic API request method for predict API
     */
    async apiRequest(endpoint, method = 'GET', body = null) {
        const url = endpoint.startsWith('http') ? endpoint : `${this.apiBaseUrl}${endpoint}`;
        
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Check API-level success flag for predict API responses
            if (data.hasOwnProperty('success') && data.success === false) {
                throw new Error(data.error || 'API request failed');
            }

            return data;
        } catch (error) {
            console.error('Predict API Request failed:', error);
            throw error;
        }
    }

    /**
     * Training API request method for mesh info and boundary data
     */
    async trainingApiRequest(endpoint, method = 'GET', body = null) {
        const url = `http://127.0.0.1:5000${endpoint}`;
        
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            return data;
        } catch (error) {
            console.error('Training API Request failed:', error);
            throw error;
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        if (this.canvasRenderer) {
            this.canvasRenderer.onResize();
        }
    }

    /**
     * Preview reference point for current selection
     */
    async previewReferencePoint() {
        const meshName = document.getElementById('mesh-select').value;
        const refSelectorType = document.getElementById('ref-selector-select').value;
        
        if (!meshName || !refSelectorType) return;
        
        try {
            // Always get the selector config if the input is visible
            const refSelectorConfig = {};
            const n_input = document.getElementById('ref-selector-n');
            if (n_input && !n_input.closest('.hidden')) {
                refSelectorConfig.n = parseInt(n_input.value) || 1;
            }
            
            const response = await this.apiRequest('/reference_point/preview', 'POST', {
                mesh_name: meshName.replace('.txt', ''),
                ref_selector_type: refSelectorType,
                ref_selector_config: refSelectorConfig
            });
            
            if (response.success && response.preview) {
                const preview = response.preview;
                this.currentReferencePoint = {
                    reference_vertex_idx: preview.reference_vertex_idx,
                    reference_vertex_coords: preview.reference_vertex_coords,
                    selector_info: preview.selector_info,
                    boundary_context: preview.boundary_context
                };
                
                // Update canvas with preview
                if (this.canvasRenderer && preview.boundary_vertices) {
                    this.canvasRenderer.renderBoundaryPreview(
                        preview.boundary_vertices,
                        meshName,
                        this.currentReferencePoint
                    );
                }
                
                this.logMessage(`Reference point preview: vertex ${preview.reference_vertex_idx} (angle: ${preview.boundary_context.interior_angle.toFixed(2)}°)`, 'info');
            }
        } catch (error) {
            console.error('Failed to preview reference point:', error);
            this.logMessage('Failed to preview reference point: ' + error.message, 'warning');
        }
    }
    
    /**
     * Update session reference selector configuration
     */
    async updateSessionRefSelector(selectorType) {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);

            // Always get the selector config if the input is visible
            const refSelectorConfig = {};
            const n_input = document.getElementById('ref-selector-n');
            if (n_input && !n_input.closest('.hidden')) {
                refSelectorConfig.n = parseInt(n_input.value) || 1;
            }

            const config = {
                ref_selector_type: selectorType,
                ref_selector_config: refSelectorConfig
            };

            const response = await this.apiRequest(`/session/${this.sessionId}/config`, 'PUT', config);
            this.logMessage(`Updated reference selector to: ${selectorType}`, 'success');

            // The response from the config update now contains the full, updated session status
            if (response.success && response.status) {
                // Update the current reference point from the response
                if (response.status.reference_point) {
                    this.currentReferencePoint = response.status.reference_point;
                    this.updateReferencePointDisplay(this.currentReferencePoint);
                }
                // Update the entire session status, which re-renders the canvas and buttons
                this.updateSessionStatus(response.status);
            }

        } catch (error) {
            console.error('Failed to update reference selector:', error);
            this.showError('Failed to update reference selector: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Show/hide reselect button
     */
    showReselectButton(show) {
        const container = document.getElementById('reselect-button-container');
        if (container) {
            if (show) {
                container.classList.remove('hidden');
            } else {
                container.classList.add('hidden');
            }
        }
    }

    /**
     * Trigger a re-selection of the reference point
     */
    async reselectReferencePoint() {
        if (!this.isSessionActive) return;

        this.logMessage('Requesting new reference point...', 'info');
        await this.updateCurrentReferencePoint();
    }

    /**
     * Update current reference point from session
     */
    async updateCurrentReferencePoint() {
        if (!this.sessionId) return;
        
        try {
            const response = await this.apiRequest(`/session/${this.sessionId}/reference_point`, 'GET');
            
            if (response.success && response.reference_point) {
                this.currentReferencePoint = response.reference_point;
                
                // Update reference point display in the UI
                this.updateReferencePointDisplay(response.reference_point);
                
                // Refresh the entire session status to ensure UI consistency, including button states
                this.updateSessionStatus(response.reference_point.session_status);
                
                this.logMessage(`Reference point updated: vertex ${response.reference_point.reference_vertex_idx}`, 'info');
            }
        } catch (error) {
            console.error('Failed to update reference point:', error);
            this.logMessage('Failed to update reference point: ' + error.message, 'warning');
        }
    }
    
    /**
     * Update reference point display in UI
     */
    updateReferencePointDisplay(refPoint) {
        if (!refPoint) return;
        
        this.updateElement('ref-vertex-idx-display', refPoint.reference_vertex_idx);
        
        if (refPoint.reference_vertex_coords) {
            const coords = refPoint.reference_vertex_coords;
            this.updateElement('ref-vertex-coords-display', `(${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`);
        }
        
        if (refPoint.selector_info) {
            this.updateElement('ref-selector-type-display', refPoint.selector_info.type);
        }
        
        if (refPoint.boundary_context) {
            this.updateElement('interior-angle-display', refPoint.boundary_context.interior_angle.toFixed(2) + '°');
        }
        
        // Show reference point panel
        const refPointPanel = document.getElementById('reference-point-info');
        if (refPointPanel) {
            refPointPanel.classList.remove('hidden');
        }
    }
    
    /**
     * Clear reference point display
     */
    clearReferencePointDisplay() {
        this.updateElement('ref-vertex-idx-display', '-');
        this.updateElement('ref-vertex-coords-display', '-');
        this.updateElement('ref-selector-type-display', '-');
        this.updateElement('interior-angle-display', '-');
        
        const refPointPanel = document.getElementById('reference-point-info');
        if (refPointPanel) {
            refPointPanel.classList.add('hidden');
        }
    }

    /**
     * Cleanup resources
     */
    destroy() {
        if (this.canvasRenderer) {
            this.canvasRenderer.destroy();
        }
    }
}