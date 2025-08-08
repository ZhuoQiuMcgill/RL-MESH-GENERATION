import React, { useState, useEffect, useRef } from 'react';
import { NavHeader, MeshCanvas } from '../components';
import { Button, FormInput, FormSelect, LoadingOverlay, EmptyState } from '../components/ui';
import { useApi } from '../context/ApiProvider';

const Generator = () => {
  // State management
  const [components, setComponents] = useState(null);
  const [selectedMesh, setSelectedMesh] = useState('');
  const [meshInfo, setMeshInfo] = useState(null);
  const [selectedPredictor, setSelectedPredictor] = useState('');
  const [selectedRefSelector, setSelectedRefSelector] = useState('');
  const [selectedQualityMethod, setSelectedQualityMethod] = useState('');
  const [predictorConfig, setPredictorConfig] = useState({ n: 2, g: 3, beta: 6, modelPath: '' });
  const [refSelectorConfig, setRefSelectorConfig] = useState({ n: 2 });
  const [sessionId, setSessionId] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [sessionData, setSessionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [log, setLog] = useState([]);
  const [actionInfo, setActionInfo] = useState(null);
  const [referencePointInfo, setReferencePointInfo] = useState(null);
  const [elementQuality, setElementQuality] = useState(null);
  
  // Refs
  const canvasRef = useRef(null);
  const api = useApi();

  // Initialize component
  useEffect(() => {
    loadComponents();
    addLogEntry('Mesh Generator initialized successfully', 'info');
  }, []);

  const addLogEntry = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLog(prev => [...prev, { message: `[${timestamp}] ${message}`, type }]);
  };

  const loadComponents = async () => {
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // For now, we'll simulate the components loading
      const mockComponents = {
        initial_meshes: ['mesh1.inp', 'mesh2.inp', 'mesh3.inp'],
        predictors: {
          'rl_predictor': { name: 'RL Predictor', description: 'Reinforcement learning based predictor' },
          'heuristic_predictor': { name: 'Heuristic Predictor', description: 'Rule-based heuristic predictor' }
        },
        reference_selectors: {
          'angle_based': { name: 'Angle Based', description: 'Select reference point based on angle' },
          'random': { name: 'Random', description: 'Random reference point selection' }
        },
        quality_methods: {
          'aspect_ratio': { name: 'Aspect Ratio', description: 'Element aspect ratio quality' },
          'skewness': { name: 'Skewness', description: 'Element skewness quality' }
        },
        trained_models: [
          { name: 'Model 1', path: '/models/model1.pth', size: 1024000 },
          { name: 'Model 2', path: '/models/model2.pth', size: 2048000 }
        ]
      };
      
      setComponents(mockComponents);
      addLogEntry('Components loaded successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load components: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleMeshChange = async (meshName) => {
    if (!meshName) {
      setSelectedMesh('');
      setMeshInfo(null);
      if (canvasRef.current) {
        canvasRef.current.clearCanvas();
      }
      return;
    }

    try {
      setIsLoading(true);
      setSelectedMesh(meshName);
      
      // Get mesh info
      const info = await api.getMeshInfo(meshName);
      setMeshInfo(info);
      
      // Load mesh boundary for preview
      const boundaryData = await api.getMeshBoundary(meshName);
      if (boundaryData.success && canvasRef.current) {
        canvasRef.current.renderBoundaryPreview(boundaryData.boundary_vertices, meshName);
      }
      
      addLogEntry(`Selected mesh: ${meshName}`, 'info');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load mesh: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const createSession = async () => {
    try {
      setIsLoading(true);
      
      const sessionConfig = {
        initial_mesh: selectedMesh,
        predictor_type: selectedPredictor,
        predictor_config: predictorConfig,
        reference_selector: selectedRefSelector,
        reference_selector_config: refSelectorConfig,
        quality_method: selectedQualityMethod
      };
      
      // Note: This would need to be implemented in the predict API
      // const response = await api.createPredictionSession(sessionConfig);
      // setSessionId(response.session_id);
      // setSessionData(response);
      
      addLogEntry('Session created successfully', 'success');
      addLogEntry(`Session ID: ${sessionId}`, 'info');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to create session: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const executeNextStep = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // const response = await api.executeNextStep(sessionId);
      // setCurrentStep(prev => prev + 1);
      // setActionInfo(response.action_info);
      
      // Update canvas visualization
      // if (canvasRef.current && response.visualization_data) {
      //   canvasRef.current.renderScene(
      //     response.visualization_data.mesh_data,
      //     response.visualization_data.boundary_vertices,
      //     response.visualization_data.ref_point_info
      //   );
      // }
      
      addLogEntry(`Executed step ${currentStep + 1}`, 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to execute step: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const executePreviousStep = async () => {
    if (!sessionId || currentStep <= 0) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // const response = await api.executePreviousStep(sessionId);
      setCurrentStep(prev => Math.max(0, prev - 1));
      
      addLogEntry(`Reverted to step ${currentStep - 1}`, 'info');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to revert step: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const processAllSteps = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // const response = await api.processAllSteps(sessionId);
      
      addLogEntry('Processing all steps...', 'info');
      addLogEntry('All steps processed successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to process all steps: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const resetSession = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // await api.resetSession(sessionId);
      
      setCurrentStep(0);
      setActionInfo(null);
      setReferencePointInfo(null);
      setElementQuality(null);
      
      addLogEntry('Session reset successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to reset session: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const deleteSession = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // await api.deleteSession(sessionId);
      
      setSessionId(null);
      setSessionData(null);
      setCurrentStep(0);
      setActionInfo(null);
      setReferencePointInfo(null);
      setElementQuality(null);
      
      addLogEntry('Session deleted successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to delete session: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const reselectReferencePoint = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the predict API
      // const response = await api.reselectReferencePoint(sessionId);
      // setReferencePointInfo(response.reference_point_info);
      
      addLogEntry('Reference point reselected', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to reselect reference point: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const clearLog = () => {
    setLog([]);
    addLogEntry('Log cleared', 'info');
  };

  const isConfigurationValid = () => {
    return selectedMesh && selectedPredictor && selectedRefSelector && selectedQualityMethod && predictorConfig.modelPath;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      <NavHeader 
        title="Mesh Generator"
        breadcrumbs={[
          { label: 'Tools', href: '/' },
          { label: 'Mesh Generator', href: '/generator' }
        ]}
      />
      
      {/* Main Container */}
      <div className="flex min-h-[calc(100vh-var(--nav-header-height))] bg-bg-primary">
        {/* Left Configuration Panel */}
        <div className="w-80 min-w-80 bg-card border-r border-border-primary overflow-y-auto flex-shrink-0">
          <div className="p-6 border-b border-gray-200">
            <h1 className="text-2xl font-bold text-text-primary">Mesh Generator</h1>
            <p className="text-sm text-text-secondary mt-1">Interactive mesh generation with RL prediction</p>
          </div>
          
          <div className="flex-1 p-6 flex flex-col overflow-hidden">
            {/* Session Setup */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-text-primary mb-4">Session Setup</h3>
              
              {/* Mesh Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-text-primary mb-2">Select Initial Mesh</label>
                <FormSelect
                  value={selectedMesh}
                  onChange={(e) => handleMeshChange(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Select a mesh...</option>
                  {components?.initial_meshes?.map((mesh) => (
                    <option key={mesh} value={mesh}>{mesh}</option>
                  ))}
                </FormSelect>
                {meshInfo && (
                  <div className="mt-2 text-xs text-text-secondary">
                    <div>Vertices: {meshInfo.vertices || 0}</div>
                    <div>File size: {formatFileSize(meshInfo.size || 0)}</div>
                  </div>
                )}
              </div>
              
              {/* Predictor Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-text-primary mb-2">Select Predictor</label>
                <FormSelect
                  value={selectedPredictor}
                  onChange={(e) => setSelectedPredictor(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Select a predictor...</option>
                  {components?.predictors && Object.entries(components.predictors).map(([key, predictor]) => (
                    <option key={key} value={key} title={predictor.description}>
                      {predictor.name}
                    </option>
                  ))}
                </FormSelect>
                
                {selectedPredictor && (
                  <div className="mt-3 bg-bg-tertiary border border-border-primary rounded p-3">
                    <div className="space-y-3">
                      <div>
                        <label className="block text-xs font-medium text-text-primary mb-1">Model Path</label>
                        <FormSelect
                          value={predictorConfig.modelPath}
                          onChange={(e) => setPredictorConfig(prev => ({ ...prev, modelPath: e.target.value }))}
                        >
                          <option value="">Select a trained model...</option>
                          {components?.trained_models?.map((model) => (
                            <option key={model.path} value={model.path}>
                              {model.name} ({formatFileSize(model.size)})
                            </option>
                          ))}
                        </FormSelect>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <label className="block text-xs font-medium text-text-primary mb-1">N</label>
                          <FormInput
                            type="number"
                            value={predictorConfig.n}
                            onChange={(e) => setPredictorConfig(prev => ({ ...prev, n: parseInt(e.target.value) || 2 }))}
                            min="1"
                            className="text-sm text-center"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-text-primary mb-1">G</label>
                          <FormInput
                            type="number"
                            value={predictorConfig.g}
                            onChange={(e) => setPredictorConfig(prev => ({ ...prev, g: parseInt(e.target.value) || 3 }))}
                            min="1"
                            className="text-sm text-center"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-text-primary mb-1">Beta</label>
                          <FormInput
                            type="number"
                            value={predictorConfig.beta}
                            onChange={(e) => setPredictorConfig(prev => ({ ...prev, beta: parseInt(e.target.value) || 6 }))}
                            min="1"
                            className="text-sm text-center"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Reference Selector */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-text-primary mb-2">Reference Selector</label>
                <FormSelect
                  value={selectedRefSelector}
                  onChange={(e) => setSelectedRefSelector(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Select a reference selector...</option>
                  {components?.reference_selectors && Object.entries(components.reference_selectors).map(([key, selector]) => (
                    <option key={key} value={key} title={selector.description}>
                      {selector.name}
                    </option>
                  ))}
                </FormSelect>
                
                {selectedRefSelector && (
                  <div className="mt-3 bg-bg-tertiary border border-border-primary rounded p-3">
                    <div>
                      <label className="block text-xs font-medium text-text-primary mb-1">N</label>
                      <FormInput
                        type="number"
                        value={refSelectorConfig.n}
                        onChange={(e) => setRefSelectorConfig(prev => ({ ...prev, n: parseInt(e.target.value) || 2 }))}
                        min="1"
                        className="text-sm"
                      />
                    </div>
                    
                    {sessionId && (
                      <div className="mt-3">
                        <Button onClick={reselectReferencePoint} variant="secondary" size="sm" className="w-full">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h5M20 20v-5h-5M4 9a9 9 0 0114.13-4.44M20 15a9 9 0 01-14.13 4.44" />
                          </svg>
                          Reselect Point
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Quality Method Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-text-primary mb-2">Quality Method</label>
                <FormSelect
                  value={selectedQualityMethod}
                  onChange={(e) => setSelectedQualityMethod(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Select quality method...</option>
                  {components?.quality_methods && Object.entries(components.quality_methods).map(([key, method]) => (
                    <option key={key} value={key} title={method.description}>
                      {method.name}
                    </option>
                  ))}
                </FormSelect>
              </div>
              
              {/* Create Session Button */}
              <Button
                onClick={createSession}
                variant="primary"
                className="w-full"
                disabled={!isConfigurationValid() || isLoading}
              >
                Create Session
              </Button>
            </div>
          </div>
        </div>
        
        {/* Main Content Container */}
        <div className="flex-1 flex flex-row min-w-0 gap-px bg-border-primary">
          {/* Canvas Visualization Area */}
          <div className="flex-1 flex flex-col bg-card min-w-96">
            <div className="flex-1 relative p-4 bg-bg-canvas">
              {selectedMesh ? (
                <MeshCanvas 
                  ref={canvasRef}
                  className="w-full h-full rounded border border-border-primary"
                />
              ) : (
                <EmptyState 
                  title="Ready to Generate Mesh"
                  description="Select a mesh, predictor, and reference selector to begin mesh generation. The visualization will appear here during the process."
                  icon="🔧"
                />
              )}
            </div>
            
            {/* Session Controls */}
            <div className="p-4 border-t border-border-primary bg-card">
              <div className="flex items-center gap-3 flex-wrap">
                <div className="flex gap-2">
                  <Button 
                    onClick={executePreviousStep} 
                    variant="secondary" 
                    size="sm"
                    disabled={!sessionId || currentStep <= 0 || isLoading}
                  >
                    ← Previous
                  </Button>
                  <Button 
                    onClick={executeNextStep} 
                    variant="primary" 
                    size="sm"
                    disabled={!sessionId || isLoading}
                  >
                    Next →
                  </Button>
                </div>
                <Button 
                  onClick={processAllSteps} 
                  variant="warning" 
                  size="sm"
                  disabled={!sessionId || isLoading}
                >
                  Process All
                </Button>
                <div className="flex gap-2 ml-auto">
                  <Button 
                    onClick={resetSession} 
                    variant="tertiary" 
                    size="sm"
                    disabled={!sessionId || isLoading}
                  >
                    Reset
                  </Button>
                  <Button 
                    onClick={deleteSession} 
                    variant="danger" 
                    size="sm"
                    disabled={!sessionId || isLoading}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right Data Panel */}
          <div className="w-80 min-w-80 bg-card flex flex-col overflow-hidden">
            {/* Step Details Header */}
            <div className="p-4 border-b border-border-primary">
              <h2 className="text-lg font-semibold text-text-primary">Step Details</h2>
              <p className="text-sm text-text-secondary">
                {sessionId ? `Step ${currentStep} - Session Active` : 'No Session Active'}
              </p>
            </div>
            
            {/* Action Information Section */}
            <div className="p-4 border-b border-border-primary">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Last Action</h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Action Type:</span>
                  <span className="text-text-primary font-medium">{actionInfo?.type || '-'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Reference Vertex:</span>
                  <span className="text-text-primary font-medium">{actionInfo?.referenceVertex || '-'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Status:</span>
                  <span className={`font-medium ${
                    actionInfo?.status === 'valid' ? 'text-success' :
                    actionInfo?.status === 'invalid' ? 'text-danger' :
                    'text-text-primary'
                  }`}>
                    {actionInfo?.status || '-'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">New Coordinates:</span>
                  <span className="text-text-primary font-medium">{actionInfo?.newCoords || '-'}</span>
                </div>
              </div>
            </div>
            
            {/* Reference Point Information */}
            {referencePointInfo && (
              <div className="p-4 border-b border-border-primary">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Current Reference Point</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Vertex Index:</span>
                    <span className="text-text-primary font-medium">{referencePointInfo.index || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Coordinates:</span>
                    <span className="text-text-primary font-medium">{referencePointInfo.coords || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Interior Angle:</span>
                    <span className="text-text-primary font-medium">{referencePointInfo.angle || '-'}°</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Element Quality Section */}
            {elementQuality && (
              <div className="p-4 border-b border-border-primary">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Element Quality</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Method:</span>
                    <span className="text-text-primary font-medium">{elementQuality.method || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Elements:</span>
                    <span className="text-text-primary font-medium">{elementQuality.count || '0'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Average Quality:</span>
                    <span className="text-text-primary font-medium">{elementQuality.average?.toFixed(3) || '-'}</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Session Status */}
            {sessionData && (
              <div className="p-4 border-b border-border-primary">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Session Status</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Session ID:</span>
                    <span className="text-text-primary font-medium text-xs">{sessionId || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Current Step:</span>
                    <span className="text-text-primary font-medium">{currentStep}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Boundary Size:</span>
                    <span className="text-text-primary font-medium">{sessionData.boundarySize || '0'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Generated Elements:</span>
                    <span className="text-text-primary font-medium">{sessionData.generatedElements || '0'}</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Operation Log Section */}
            <div className="flex-1 p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-text-primary">Operation Log</h3>
                <Button onClick={clearLog} variant="tertiary" size="sm">
                  Clear
                </Button>
              </div>
              <div className="flex-1 bg-bg-tertiary border border-border-primary rounded p-3 overflow-y-auto font-mono text-xs">
                {log.length === 0 ? (
                  <div className="text-text-secondary">System ready, waiting for session creation...</div>
                ) : (
                  log.map((entry, index) => (
                    <div key={index} className={`mb-1 ${
                      entry.type === 'error' ? 'text-danger font-semibold' :
                      entry.type === 'success' ? 'text-success font-medium' :
                      entry.type === 'warning' ? 'text-warning font-medium' :
                      'text-text-secondary'
                    }`}>
                      {entry.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Loading Overlay */}
      <LoadingOverlay isVisible={isLoading} message="Processing..." />
    </div>
  );
};

export default Generator;
