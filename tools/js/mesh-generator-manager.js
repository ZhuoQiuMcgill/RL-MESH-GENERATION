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
            
            // Load quality methods
            await this.loadQualityMethods();
            
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
        
        // Add async process all button if it exists
        const processAllAsyncBtn = document.getElementById('process-all-async-btn');
        if (processAllAsyncBtn) {
            processAllAsyncBtn.addEventListener('click', () => this.processAllStepsAsync());
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
        const qualityMethodSelect = document.getElementById('quality-method-select');
        const createSessionBtn = document.getElementById('create-session-btn');

        const isValid = meshSelect.value && 
                       predictorSelect.value && 
                       refSelectorSelect.value && 
                       qualityMethodSelect.value &&
                       (predictorSelect.value !== 'RL' || modelSelect.value);

        if (createSessionBtn) {
            createSessionBtn.disabled = !isValid || this.isSessionActive;
        }
    }

    /**
     * Load available quality methods
     */
    async loadQualityMethods() {
        try {
            const response = await this.apiRequest('/quality/methods', 'GET');
            this.qualityMethods = response.methods || [];
            
            // Populate quality method select
            const qualityMethodSelect = document.getElementById('quality-method-select');
            if (qualityMethodSelect && this.qualityMethods.length > 0) {
                qualityMethodSelect.innerHTML = '<option value="">Select Quality Method</option>';
                this.qualityMethods.forEach((method, index) => {
                    const option = document.createElement('option');
                    option.value = method;
                    option.textContent = method;
                    qualityMethodSelect.appendChild(option);
                    
                    // Select hybrid as default if available
                    if (method === 'hybrid') {
                        option.selected = true;
                    }
                });
                
                // Add change event listener for automatic quality updates
                qualityMethodSelect.addEventListener('change', () => {
                    this.validateConfiguration();
                    if (this.isSessionActive && qualityMethodSelect.value) {
                        this.updateQualityResults();
                    }
                });
            }
        } catch (error) {
            console.error('Failed to load quality methods:', error);
        }
    }

    /**
     * Update element quality results automatically
     */
    async updateQualityResults() {
        if (!this.sessionId) {
            return;
        }

        const qualityMethodSelect = document.getElementById('quality-method-select');
        const method = qualityMethodSelect?.value;

        if (!method) {
            this.hideElementQuality();
            return;
        }

        try {
            const response = await this.apiRequest(`/session/${this.sessionId}/quality?method=${method}`, 'GET');
            
            if (response.success) {
                this.displayElementQuality(response.average_quality, response.element_count, method);
            } else {
                this.displayElementQuality(null, response.element_count || 0, method, response.message);
            }
        } catch (error) {
            console.error('Failed to calculate quality:', error);
            this.displayElementQuality(null, 0, method, 'Calculation failed');
        }
    }

    /**
     * Display element quality information
     */
    displayElementQuality(averageQuality, elementCount, method, errorMessage = null) {
        const qualitySection = document.getElementById('element-quality');
        if (qualitySection) {
            qualitySection.classList.remove('hidden');
            
            this.updateElement('quality-method-display', method);
            this.updateElement('quality-element-count-display', elementCount);
            
            if (errorMessage) {
                this.updateElement('quality-average-display', '-');
                this.updateElement('quality-status-display', 'Error');
            } else if (averageQuality !== null) {
                this.updateElement('quality-average-display', averageQuality.toFixed(4));
                this.updateElement('quality-status-display', elementCount > 0 ? 'Ready' : 'No elements');
            } else {
                this.updateElement('quality-average-display', '-');
                this.updateElement('quality-status-display', elementCount > 0 ? 'Ready' : 'No elements');
            }
        }
    }

    /**
     * Hide element quality information
     */
    hideElementQuality() {
        const qualitySection = document.getElementById('element-quality');
        if (qualitySection) {
            qualitySection.classList.add('hidden');
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
            
            // Handle initial reference point if provided in response
            if (response.initial_status && response.initial_status.reference_point) {
                this.currentReferencePoint = response.initial_status.reference_point;
                this.updateReferencePointDisplay(this.currentReferencePoint);
            }
            
            this.updateSessionStatus(response.initial_status);
            this.showSessionControls(true);
            this.updateQualityResults(); // Initial quality calculation
            this.logMessage(`Session created: ${this.sessionId}`, 'success');
            
            // Show reselect button
            this.showReselectButton(true);

            
            // Get and display current reference point (if not already set from response)
            if (!this.currentReferencePoint) {
                await this.updateCurrentReferencePoint();
            }
            
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
            
            // Handle new response format with code and action_attempted fields
            if (response.code !== undefined) {
                this.logMessage(`Step execution code: ${response.code}`, 'info');
            }
            
            if (response.action_attempted !== undefined) {
                this.logMessage(`Action attempted: ${response.action_attempted}`, 'info');
            }
            
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

                // Update reference point FIRST to ensure correct local env display
                await this.updateCurrentReferencePoint();
                
                // Then refresh the session status, which will trigger a re-render with correct reference point
                await this.refreshSessionStatus(true);
                
                // Update quality results after going to previous step
                this.updateQualityResults();
            } else {
                this.logMessage('Undo failed: ' + response.undo_result.message, 'warning');
            }
            
        } catch (error) {
            console.error('Failed to undo step:', error);
            this.showError('Failed to undo step: ' + error.message);
        } finally {
            this.showLoading(false);
            this.setButtonLoading('prev-step-btn', false);
        }
    }

    /**
     * Process all remaining steps using async API with status polling
     */
    async processAllSteps() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            this.lockProcessingButtons(true);
            
            this.logMessage('Starting async mesh generation...', 'info');
            
            // Start the async process
            const startResponse = await this.apiRequest(`/session/${this.sessionId}/process_all`, 'POST');
            
            if (!startResponse.processing_started) {
                throw new Error('Failed to start async processing');
            }
            
            this.logMessage('Async processing started, monitoring progress...', 'info');
            
            // Start polling the status_async endpoint every second
            let pollIntervalId = null;
            let completed = false;
            let lastMeshUpdateStep = -1; // Track when we last updated mesh visualization
            
            const pollStatus = async () => {
                try {
                    // Always request with mesh data for real-time canvas updates
                    const response = await this.apiRequest(`/session/${this.sessionId}/status_async?include_mesh=true`, 'GET');
                    
                    // Update UI with current progress
                    if (response.status) {
                        this.updateSessionStatus(response.status);
                        this.updateQualityResults();
                        
                        // Update canvas visualization with fresh mesh data (but without local env during processing)
                        this.updateCanvasVisualizationAsync(response.status, response.processing?.is_processing || false);
                        
                        // Update progress information
                        const currentStep = response.status.current_step || 0;
                        if (currentStep !== lastMeshUpdateStep) {
                            this.logMessage(
                                `Progress: Step ${currentStep}, ` +
                                `Elements: ${response.status.generated_elements_count || 0}, ` +
                                `Boundary: ${response.status.boundary_size || 0}`,
                                'info'
                            );
                            lastMeshUpdateStep = currentStep;
                        }
                    }
                    
                    // During async processing, we do NOT render the local environment
                    // This prevents the stale local env display during processing
                    
                    // Check if processing is complete
                    if (response.processing && !response.processing.is_processing) {
                        completed = true;
                        clearInterval(pollIntervalId);
                        
                        // Handle completion
                        const reason = response.processing.completion_reason || 'unknown';
                        const stepsProcessed = response.processing.steps_processed || 0;
                        
                        this.logMessage(
                            `Process All completed: ${stepsProcessed} steps (${reason})`,
                            'success'
                        );
                        
                        // Log completion details
                        if (reason === 'mesh_completed') {
                            this.logMessage('Mesh generation completed successfully!', 'success');
                        } else if (reason === 'invalid_action') {
                            this.logMessage('Process stopped due to invalid action', 'warning');
                        } else if (reason === 'max_iterations_reached') {
                            this.logMessage('Process stopped: safety limit reached (10000 steps)', 'warning');
                        } else if (reason === 'error') {
                            this.logMessage('Process stopped due to error', 'error');
                        }
                        
                        this.showLoading(false);
                        this.lockProcessingButtons(false);
                        
                        // Clear any lingering invalid action state
                        this.lastInvalidAction = null;
                        
                        // Force immediate completion state - disable buttons directly
                        const nextBtn = document.getElementById('next-step-btn');
                        const processAllBtn = document.getElementById('process-all-btn');
                        const processAllAsyncBtn = document.getElementById('process-all-async-btn');
                        
                        if (nextBtn) {
                            nextBtn.disabled = true;
                            this.logMessage('Next Step button disabled - process completed', 'info');
                        }
                        if (processAllBtn) {
                            processAllBtn.disabled = true;
                            this.logMessage('Process All button disabled - process completed', 'info');
                        }
                        if (processAllAsyncBtn) {
                            processAllAsyncBtn.disabled = true;
                        }
                        
                        // Wait a moment then get final status for proper UI update
                        setTimeout(async () => {
                            try {
                                // Get final status with mesh data
                                const finalStatusResponse = await this.apiRequest(`/session/${this.sessionId}/status?include_mesh=true`, 'GET');
                                if (finalStatusResponse.status) {
                                    console.log('Final status retrieved:', {
                                        is_completed: finalStatusResponse.status.is_completed,
                                        current_step: finalStatusResponse.status.current_step,
                                        boundary_size: finalStatusResponse.status.boundary_size
                                    });
                                    
                                    // Update session status UI
                                    this.updateSessionStatus(finalStatusResponse.status);
                                    
                                    // Get final reference point to show correct local env position
                                    await this.updateCurrentReferencePoint();
                                    
                                    // If mesh is completed, don't show local env at all
                                    if (reason === 'mesh_completed' && finalStatusResponse.status.boundary_size <= 4) {
                                        this.logMessage('Mesh completed - hiding local environment (boundary ≤ 4 vertices)', 'info');
                                        // Render without local environment
                                        if (this.canvasRenderer) {
                                            this.canvasRenderer.renderScene(
                                                finalStatusResponse.status.mesh_data || null,
                                                finalStatusResponse.status.boundary_vertices || null,
                                                null // No local environment for completed mesh
                                            );
                                        }
                                    } else {
                                        // Regular final rendering with local env
                                        this.updateCanvasVisualization(finalStatusResponse.status);
                                    }
                                }
                            } catch (error) {
                                console.error('Failed to get final status:', error);
                            }
                        }, 300);
                        
                        // Inform user about undo capability if steps were processed
                        if (stepsProcessed > 0) {
                            this.logMessage(`You can now use "Previous Step" to review each of the ${stepsProcessed} steps`, 'info');
                        }
                    }
                    
                    // Check if there was an error
                    if (!response.success) {
                        completed = true;
                        clearInterval(pollIntervalId);
                        throw new Error(response.error || 'Status polling failed');
                    }
                    
                } catch (error) {
                    console.error('Error during async status polling:', error);
                    completed = true;
                    clearInterval(pollIntervalId);
                    
                    this.showError('Async processing error: ' + error.message);
                    this.showLoading(false);
                    this.lockProcessingButtons(false);
                }
            };
            
            // Start polling every second
            pollIntervalId = setInterval(pollStatus, 1000);
            
            // Initial poll
            await pollStatus();
            
            // If already completed in first poll, clean up
            if (completed) {
                clearInterval(pollIntervalId);
            }
            
        } catch (error) {
            console.error('Failed to process all steps:', error);
            this.showError('Failed to process all steps: ' + error.message);
            this.showLoading(false);
            this.lockProcessingButtons(false);
        }
    }

    /**
     * Reset session to initial state
     */
    async resetSession() {
        if (!this.sessionId) return;

        // Confirm with user before resetting
        if (!confirm('Are you sure you want to reset the session to the initial boundary? This will clear all progress.')) {
            return;
        }

        try {
            this.showLoading(true);
            this.setButtonLoading('reset-session-btn', true);
            
            const response = await this.apiRequest(`/session/${this.sessionId}/reset`, 'POST');
            
            if (response.reset_result.success) {
                // Handle initial reference point if provided in response
                if (response.status && response.status.reference_point) {
                    this.currentReferencePoint = response.status.reference_point;
                    this.updateReferencePointDisplay(this.currentReferencePoint);
                }
                
                // Update all UI state
                this.updateSessionStatus(response.status);
                this.updateQualityResults(); // Update quality after reset
                this.clearActionInfo();
                this.currentStep = 0;
                this.lastInvalidAction = null;
                
                // Clear visualization data
                this.lastActionInfo = null;
                this.lastGeneratedElement = null;
                
                this.logMessage('Session reset to initial boundary state', 'success');
                this.logMessage('All mesh generation progress has been cleared', 'info');
                this.logMessage('You can now start fresh with Next Step or Process All', 'info');
                
                // Get reference point if not already set from response
                if (!this.currentReferencePoint) {
                    await this.updateCurrentReferencePoint();
                }
                
                // Ensure canvas shows the clean initial boundary with reference point
                if (this.canvasRenderer && response.status) {
                    this.canvasRenderer.renderScene(
                        response.status.mesh_data || null,
                        response.status.boundary_vertices || null,
                        this.currentReferencePoint
                    );
                    this.showEmptyState(false);
                }
                
            } else {
                this.logMessage('Reset failed: ' + response.reset_result.message, 'error');
                this.showError('Reset failed: ' + response.reset_result.message);
            }
        } catch (error) {
            console.error('Failed to reset session:', error);
            this.showError('Failed to reset session: ' + error.message);
        } finally {
            this.showLoading(false);
            this.setButtonLoading('reset-session-btn', false);
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
            this.hideElementQuality();
            this.clearSessionStatus();
            this.clearActionInfo();
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
            
            // Update quality results after status update
            this.updateQualityResults();
            
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
        
        // Refresh session status to get latest data with mesh data
        setTimeout(() => this.refreshSessionStatus(true), 200);
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
    async refreshSessionStatus(includeMesh = false) {
        if (!this.sessionId) return;

        try {
            // Use lightweight status by default, optionally include mesh data
            const endpoint = includeMesh 
                ? `/session/${this.sessionId}/status?include_mesh=true`
                : `/session/${this.sessionId}/status`;
                
            const response = await this.apiRequest(endpoint, 'GET');
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
            // Check if we have mesh data to render
            const hasMeshData = status.mesh_data && Object.keys(status.mesh_data).length > 0;
            const hasBoundaryData = status.boundary_vertices && Array.isArray(status.boundary_vertices) && status.boundary_vertices.length > 0;
            
            // If we don't have sufficient rendering data, try to get it
            if (!hasMeshData && !hasBoundaryData) {
                console.debug('No mesh/boundary data in status, checking cached data or fetching with mesh');
                
                // Check if canvas has cached render data we can use
                if (this.canvasRenderer.lastRenderData && 
                    (this.canvasRenderer.lastRenderData.meshData || this.canvasRenderer.lastRenderData.boundaryVertices)) {
                    // Re-render using cached data with updated reference point
                    this.canvasRenderer.renderScene(
                        this.canvasRenderer.lastRenderData.meshData,
                        this.canvasRenderer.lastRenderData.boundaryVertices,
                        this.currentReferencePoint
                    );
                    this.showEmptyState(false);
                    return;
                }
                
                // If no cached data, fetch complete status with mesh data
                setTimeout(() => this.refreshSessionStatus(true), 100);
                return;
            }
            
            // Render the mesh scene with the latest data, including the current reference point
            this.canvasRenderer.renderScene(
                status.mesh_data || null,
                status.boundary_vertices || null,
                this.currentReferencePoint
            );
            
            // Hide empty state when we have data to render
            this.showEmptyState(false);
            
        } catch (error) {
            console.error('Failed to update canvas visualization:', error);
            this.logMessage('Failed to update visualization: ' + error.message, 'error');
        }
    }

    /**
     * Update canvas visualization during async processing (without local environment)
     */
    updateCanvasVisualizationAsync(status, isProcessing) {
        if (!this.canvasRenderer || !status) return;

        try {
            // Check if we have mesh data to render
            const hasMeshData = status.mesh_data && Object.keys(status.mesh_data).length > 0;
            const hasBoundaryData = status.boundary_vertices && Array.isArray(status.boundary_vertices) && status.boundary_vertices.length > 0;
            
            // If we don't have sufficient rendering data, try to get it
            if (!hasMeshData && !hasBoundaryData) {
                console.debug('No mesh/boundary data in status, checking cached data or fetching with mesh');
                
                // Check if canvas has cached render data we can use
                if (this.canvasRenderer.lastRenderData && 
                    (this.canvasRenderer.lastRenderData.meshData || this.canvasRenderer.lastRenderData.boundaryVertices)) {
                    // Re-render using cached data WITHOUT reference point during processing
                    this.canvasRenderer.renderScene(
                        this.canvasRenderer.lastRenderData.meshData,
                        this.canvasRenderer.lastRenderData.boundaryVertices,
                        isProcessing ? null : this.currentReferencePoint
                    );
                    this.showEmptyState(false);
                    return;
                }
                
                // If no cached data, fetch complete status with mesh data
                setTimeout(() => this.refreshSessionStatus(true), 100);
                return;
            }
            
            // During async processing, do NOT render local environment (reference point)
            // This prevents stale local env display that doesn't update with the processing
            this.canvasRenderer.renderScene(
                status.mesh_data || null,
                status.boundary_vertices || null,
                isProcessing ? null : this.currentReferencePoint
            );
            
            // Hide empty state when we have data to render
            this.showEmptyState(false);
            
        } catch (error) {
            console.error('Failed to update async canvas visualization:', error);
            this.logMessage('Failed to update async visualization: ' + error.message, 'error');
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
        const processAllAsyncBtn = document.getElementById('process-all-async-btn');
        const resetBtn = document.getElementById('reset-session-btn');
        
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
        if (processAllAsyncBtn) {
            processAllAsyncBtn.disabled = !this.isSessionActive || status.is_completed;
        }
        if (resetBtn) {
            // Enable reset when session is active and has made some progress
            const enableReset = this.isSessionActive && (this.currentStep > 0 || status.generated_elements_count > 0);
            resetBtn.disabled = !enableReset;
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
     * Enable/disable session controls (always visible)
     */
    showSessionControls(enable) {
        // Controls are always visible now, just enable/disable buttons
        const nextBtn = document.getElementById('next-step-btn');
        const prevBtn = document.getElementById('prev-step-btn');
        const processAllBtn = document.getElementById('process-all-btn');
        const resetBtn = document.getElementById('reset-session-btn');
        const deleteBtn = document.getElementById('delete-session-btn');

        if (nextBtn) nextBtn.disabled = !enable;
        if (prevBtn) prevBtn.disabled = !enable;
        if (processAllBtn) processAllBtn.disabled = !enable;
        if (resetBtn) resetBtn.disabled = !enable;
        if (deleteBtn) deleteBtn.disabled = !enable;
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
     * Lock/unlock processing buttons during async processing
     */
    lockProcessingButtons(lock) {
        const nextBtn = document.getElementById('next-step-btn');
        const prevBtn = document.getElementById('prev-step-btn');
        const processAllBtn = document.getElementById('process-all-btn');
        const processAllAsyncBtn = document.getElementById('process-all-async-btn');
        const resetBtn = document.getElementById('reset-session-btn');
        const deleteBtn = document.getElementById('delete-session-btn');
        
        // Lock all processing-related buttons
        if (nextBtn) {
            nextBtn.disabled = lock;
            if (lock) {
                nextBtn.classList.add('processing-locked');
            } else {
                nextBtn.classList.remove('processing-locked');
            }
        }
        if (prevBtn) {
            prevBtn.disabled = lock;
            if (lock) {
                prevBtn.classList.add('processing-locked');
            } else {
                prevBtn.classList.remove('processing-locked');
            }
        }
        if (processAllBtn) {
            processAllBtn.disabled = lock;
            if (lock) {
                processAllBtn.classList.add('processing-locked');
            } else {
                processAllBtn.classList.remove('processing-locked');
            }
        }
        if (processAllAsyncBtn) {
            processAllAsyncBtn.disabled = lock;
            if (lock) {
                processAllAsyncBtn.classList.add('processing-locked');
            } else {
                processAllAsyncBtn.classList.remove('processing-locked');
            }
        }
        if (resetBtn) {
            resetBtn.disabled = lock;
            if (lock) {
                resetBtn.classList.add('processing-locked');
            } else {
                resetBtn.classList.remove('processing-locked');
            }
        }
        if (deleteBtn) {
            deleteBtn.disabled = lock;
            if (lock) {
                deleteBtn.classList.add('processing-locked');
            } else {
                deleteBtn.classList.remove('processing-locked');
            }
        }
        
        // Log the locking state
        if (lock) {
            this.logMessage('Processing buttons locked during async operation', 'info');
        } else {
            this.logMessage('Processing buttons unlocked', 'info');
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

        if (this.lastInvalidAction) {
            this.lastInvalidAction = null;
            this.logMessage('Next step unlocked by reselecting reference point.', 'info');
        }

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
     * Process all remaining steps asynchronously with status polling
     */
    async processAllStepsAsync() {
        if (!this.sessionId) return;

        try {
            this.showLoading(true);
            this.setButtonLoading('process-all-async-btn', true);
            
            this.logMessage('Starting async mesh generation...', 'info');
            
            // Start polling the status_async endpoint
            let pollIntervalId = null;
            let completed = false;
            
            const pollStatus = async () => {
                try {
                    const response = await this.apiRequest(`/session/${this.sessionId}/status_async`, 'GET');
                    
                    // Update UI with current progress
                    if (response.status) {
                        this.updateSessionStatus(response.status);
                        this.updateQualityResults();
                        
                        // Update progress information
                        this.logMessage(
                            `Progress: Step ${response.status.current_step || 0}, ` +
                            `Elements: ${response.status.generated_elements_count || 0}`,
                            'info'
                        );
                    }
                    
                    // Check if processing is complete
                    if (response.completed) {
                        completed = true;
                        clearInterval(pollIntervalId);
                        
                        // Handle completion
                        const reason = response.completion_reason || 'unknown';
                        this.logMessage(
                            `Async processing completed: ${response.steps_executed || 0} steps (${reason})`,
                            'success'
                        );
                        
                        // Log completion details
                        if (reason === 'mesh_completed') {
                            this.logMessage('Mesh generation completed successfully!', 'success');
                        } else if (reason === 'invalid_action') {
                            this.logMessage('Process stopped due to invalid action', 'warning');
                        } else if (reason === 'max_iterations_reached') {
                            this.logMessage('Process stopped: safety limit reached (1000 steps)', 'warning');
                        }
                        
                        // Final status update with mesh data
                        await this.refreshSessionStatus(true);
                        await this.updateCurrentReferencePoint();
                        
                        this.showLoading(false);
                        this.setButtonLoading('process-all-async-btn', false);
                    }
                    
                    // Check if there was an error
                    if (response.error) {
                        completed = true;
                        clearInterval(pollIntervalId);
                        throw new Error(response.error);
                    }
                    
                } catch (error) {
                    console.error('Error during async status polling:', error);
                    completed = true;
                    clearInterval(pollIntervalId);
                    
                    this.showError('Async processing error: ' + error.message);
                    this.showLoading(false);
                    this.setButtonLoading('process-all-async-btn', false);
                }
            };
            
            // Start polling every second
            pollIntervalId = setInterval(pollStatus, 1000);
            
            // Initial poll
            await pollStatus();
            
            // If already completed, clean up immediately
            if (completed) {
                clearInterval(pollIntervalId);
            }
            
        } catch (error) {
            console.error('Failed to start async processing:', error);
            this.showError('Failed to start async processing: ' + error.message);
            this.showLoading(false);
            this.setButtonLoading('process-all-async-btn', false);
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
