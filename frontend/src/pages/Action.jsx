import React, { useState, useEffect, useRef } from 'react';
import { NavHeader, MeshCanvas } from '../components';
import { Button, FormSelect, LoadingOverlay, CompactStatusBar, PanelCard, EmptyState } from '../components/ui';
import { useApi } from '../context/ApiProvider';

const ActionTester = () => {
  // State management
  const [meshList, setMeshList] = useState([]);
  const [selectedMesh, setSelectedMesh] = useState('');
  const [meshInfo, setMeshInfo] = useState(null);
  const [referencePoint, setReferencePoint] = useState(null);
  const [selectedAction, setSelectedAction] = useState(null);
  const [clickCoordinates, setClickCoordinates] = useState(null);
  const [executionResult, setExecutionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [log, setLog] = useState([]);
  const [isWaitingForClick, setIsWaitingForClick] = useState(false);
  const [status, setStatus] = useState('Ready');
  
  // Refs
  const canvasRef = useRef(null);
  const api = useApi();

  // Initialize component
  useEffect(() => {
    loadMeshList();
    addLogEntry('Action Tester initialized. Select a mesh to begin testing.', 'info');
  }, []);

  const addLogEntry = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLog(prev => [...prev, { message: `[${timestamp}] ${message}`, type }]);
  };

  const loadMeshList = async () => {
    try {
      setIsLoading(true);
      const response = await api.getMeshList();
      setMeshList(response.meshes || []);
      addLogEntry('Mesh list loaded successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load mesh list: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleMeshChange = async (meshName) => {
    if (!meshName) {
      setSelectedMesh('');
      setMeshInfo(null);
      setReferencePoint(null);
      setSelectedAction(null);
      setClickCoordinates(null);
      setExecutionResult(null);
      if (canvasRef.current) {
        canvasRef.current.clearCanvas();
      }
      setStatus('Ready');
      return;
    }

    try {
      setIsLoading(true);
      setSelectedMesh(meshName);
      setStatus('Loading Mesh');
      
      // Get mesh info
      const info = await api.getMeshInfo(meshName);
      setMeshInfo(info);
      
      // Load mesh boundary for visualization
      const boundaryData = await api.getMeshBoundary(meshName);
      if (boundaryData.success && canvasRef.current) {
        canvasRef.current.renderBoundaryPreview(boundaryData.boundary_vertices, meshName);
      }
      
      addLogEntry(`Selected mesh: ${meshName}`, 'info');
      addLogEntry(`Vertices: ${info.vertices || 'N/A'}, Boundary vertices: ${boundaryData.vertex_count || 'N/A'}`, 'info');
      setStatus('Mesh Loaded');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load mesh: ${err.message}`, 'error');
      setStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  const findReferencePoint = async () => {
    if (!selectedMesh) return;
    
    try {
      setIsLoading(true);
      setStatus('Finding Reference Point');
      
      const response = await api.findReferencePoint(selectedMesh);
      setReferencePoint(response);
      
      // Update canvas to highlight reference point
      // if (canvasRef.current && response.reference_point) {
      //   canvasRef.current.highlightReferencePoint(response.reference_point);
      // }
      
      addLogEntry(`Reference point found at index ${response.vertex_index}`, 'success');
      addLogEntry(`Coordinates: (${response.coordinates?.join(', ') || 'N/A'}), Interior angle: ${response.interior_angle || 'N/A'}°`, 'info');
      setStatus('Reference Point Found');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to find reference point: ${err.message}`, 'error');
      setStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  const selectAction = (actionType) => {
    setSelectedAction(actionType);
    setClickCoordinates(null);
    setExecutionResult(null);
    
    if (actionType === 'type1') {
      setIsWaitingForClick(true);
      setStatus('Waiting for Canvas Click');
      addLogEntry('Type1 action selected. Click on canvas to place vertex.', 'info');
    } else {
      setIsWaitingForClick(false);
      setStatus('Action Selected');
      addLogEntry(`${actionType} action selected`, 'info');
    }
  };

  const handleCanvasClick = (worldCoords, event) => {
    if (!isWaitingForClick) return;
    
    if (worldCoords) {
      setClickCoordinates(worldCoords);
      setIsWaitingForClick(false);
      setStatus('Click Recorded');
      
      const coordsText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
      addLogEntry(`Canvas clicked at: ${coordsText}`, 'info');
    } else {
      addLogEntry('Invalid click coordinates', 'warning');
    }
  };

  const executeAction = async () => {
    if (!selectedMesh || !referencePoint || !selectedAction) return;
    
    try {
      setIsLoading(true);
      setStatus('Executing Action');
      
      const actionData = {
        mesh_name: selectedMesh,
        action_type: selectedAction,
        reference_point: referencePoint,
        click_coordinates: clickCoordinates
      };
      
      const response = await api.executeAction(actionData);
      setExecutionResult(response);
      
      // Update canvas visualization with result
      // if (canvasRef.current && response.visualization_data) {
      //   canvasRef.current.renderActionResult(response.visualization_data);
      // }
      
      const resultText = response.is_valid ? 'Valid' : 'Invalid';
      addLogEntry(`Action executed: ${resultText}`, response.is_valid ? 'success' : 'warning');
      
      if (response.polar_coordinates) {
        addLogEntry(`Polar coordinates: (${response.polar_coordinates.join(', ')})`, 'info');
      }
      
      setStatus(response.is_valid ? 'Action Valid' : 'Action Invalid');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to execute action: ${err.message}`, 'error');
      setStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  const clearLog = () => {
    setLog([]);
    addLogEntry('Log cleared', 'info');
  };

  const canExecute = () => {
    if (!selectedMesh || !referencePoint || !selectedAction) return false;
    if (selectedAction === 'type1' && !clickCoordinates) return false;
    return true;
  };

  const getStatusColor = () => {
    if (status.includes('Error')) return 'danger';
    if (status.includes('Loading') || status.includes('Finding') || status.includes('Executing')) return 'warning';
    if (status.includes('Found') || status.includes('Valid')) return 'success';
    return 'primary';
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      <NavHeader 
        title="Action Tester"
        breadcrumbs={[
          { label: 'Tools', href: '/' },
          { label: 'Action Tester', href: '/action' }
        ]}
      />
      
      {/* Compact Status Bar */}
      <CompactStatusBar
        status={status}
        statusColor={getStatusColor()}
        subtitle="Action Tester - Interactive RL Action Testing"
      />
      
      {/* Main Container */}
      <div className="flex min-h-[calc(100vh-var(--nav-header-height)-var(--status-bar-height))] bg-bg-primary">
        {/* Left Control Panel */}
        <div className="w-80 min-w-80 bg-card border-r border-border-primary overflow-y-auto flex-shrink-0">
          <div className="p-4 border-b border-gray-200">
            <h1 className="text-xl font-bold text-text-primary">Action Tester</h1>
            <p className="text-sm text-text-secondary mt-1">Test RL Actions Interactively</p>
          </div>
          
          <div className="flex-1 p-4 flex flex-col overflow-hidden">
            {/* Step 1: Mesh Selection */}
            <PanelCard title="Step 1: Select Mesh">
              <FormSelect
                value={selectedMesh}
                onChange={(e) => handleMeshChange(e.target.value)}
                disabled={isLoading}
              >
                <option value="">Select a mesh...</option>
                {meshList.map((mesh) => (
                  <option key={mesh} value={mesh}>{mesh}</option>
                ))}
              </FormSelect>
            </PanelCard>
            
            {/* Step 2: Find Reference Point */}
            <PanelCard title="Step 2: Reference Point">
              <Button 
                onClick={findReferencePoint}
                variant="primary"
                className="w-full"
                disabled={!selectedMesh || isLoading}
              >
                Find Reference Point
              </Button>
            </PanelCard>
            
            {/* Step 3: Choose Action */}
            <PanelCard title="Step 3: Choose Action">
              {referencePoint ? (
                <div className="space-y-2">
                  <Button
                    onClick={() => selectAction('type0-left')}
                    variant={selectedAction === 'type0-left' ? 'primary' : 'secondary'}
                    className="w-full"
                  >
                    Type0 Left
                  </Button>
                  <Button
                    onClick={() => selectAction('type0-right')}
                    variant={selectedAction === 'type0-right' ? 'primary' : 'secondary'}
                    className="w-full"
                  >
                    Type0 Right
                  </Button>
                  <Button
                    onClick={() => selectAction('type1')}
                    variant={selectedAction === 'type1' ? 'primary' : 'secondary'}
                    className="w-full"
                  >
                    Type1 (Click to Draw)
                  </Button>
                  
                  {/* Type1 Instruction */}
                  {selectedAction === 'type1' && (
                    <div className="mt-3 p-2 bg-warning-light rounded text-sm text-warning-dark border border-warning">
                      💡 Click on canvas to place vertex
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-text-secondary py-4">
                  Find reference point first
                </div>
              )}
            </PanelCard>
            
            {/* Step 4: Execute */}
            <PanelCard title="Step 4: Execute">
              <Button
                onClick={executeAction}
                variant="success"
                className="w-full"
                disabled={!canExecute() || isLoading}
              >
                Execute Action
              </Button>
            </PanelCard>
          </div>
        </div>
        
        {/* Right Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Canvas and Info Side by Side */}
          <div className="flex-1 flex flex-row min-w-0 gap-px bg-border-primary">
            {/* Canvas Visualization Area */}
            <div className="flex-1 flex flex-col bg-card min-w-96">
              {/* Canvas Status Bar */}
              <div className="p-3 border-b border-border-primary bg-card">
                <h2 className="text-base font-semibold text-text-primary">Mesh Visualization</h2>
                <div className="flex gap-2 mt-1">
                  <span className={`text-xs px-2 py-1 rounded ${
                    selectedMesh ? 'bg-success text-success-fg' : 'bg-gray-100 text-text-secondary'
                  }`}>
                    {selectedMesh ? 'Mesh Selected' : 'Not Selected'}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    referencePoint ? 'bg-success text-success-fg' : 'bg-gray-100 text-text-secondary'
                  }`}>
                    {referencePoint ? 'Reference Found' : 'No Reference'}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    selectedAction ? 'bg-primary text-white' : 'bg-gray-100 text-text-secondary'
                  }`}>
                    {selectedAction ? `${selectedAction} Selected` : 'No Action'}
                  </span>
                </div>
              </div>
              
              <div className="flex-1 relative p-4 bg-bg-canvas">
                {selectedMesh ? (
                  <MeshCanvas 
                    ref={canvasRef}
                    onCanvasClick={handleCanvasClick}
                    className={`w-full h-full rounded border border-border-primary ${
                      isWaitingForClick ? 'cursor-crosshair' : 'cursor-default'
                    }`}
                  />
                ) : (
                  <EmptyState 
                    title="Select a Mesh to Begin"
                    description="Choose a mesh from the dropdown to start testing actions. The mesh visualization will appear here."
                    icon="⚙️"
                  />
                )}
              </div>
            </div>
            
            {/* Right Info Panel */}
            <div className="w-80 min-w-80 bg-card flex flex-col overflow-hidden">
              {/* Status Overview */}
              <div className="p-4 border-b border-border-primary">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Status Overview</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Mesh Status:</span>
                    <span className="text-text-primary font-medium">{selectedMesh ? 'Selected' : 'Not Selected'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Boundary Vertices:</span>
                    <span className="text-text-primary font-medium">{meshInfo?.boundary_vertices || 0}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Reference Point:</span>
                    <span className="text-text-primary font-medium">{referencePoint ? 'Found' : 'Not Selected'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Action Status:</span>
                    <span className="text-text-primary font-medium">{selectedAction || 'No Action'}</span>
                  </div>
                </div>
              </div>
              
              {/* Detailed Information Cards */}
              <div className="flex-1 overflow-y-auto">
                {/* Mesh Information Card */}
                {meshInfo && (
                  <div className="p-4 border-b border-border-primary">
                    <h4 className="text-sm font-semibold text-text-primary mb-2">Mesh Information</h4>
                    <div className="text-sm space-y-1">
                      <div>Vertices: <span className="font-medium">{meshInfo.vertices || 0}</span></div>
                      <div>File size: <span className="font-medium">{meshInfo.size || 0}</span> bytes</div>
                    </div>
                  </div>
                )}
                
                {/* Reference Point Information Card */}
                {referencePoint && (
                  <div className="p-4 border-b border-border-primary">
                    <h4 className="text-sm font-semibold text-text-primary mb-2">Reference Point Details</h4>
                    <div className="text-sm space-y-1">
                      <div>Index: <span className="font-medium">{referencePoint.vertex_index || '-'}</span></div>
                      <div>Coordinates: <span className="font-medium">{referencePoint.coordinates?.join(', ') || '-'}</span></div>
                      <div>Interior Angle: <span className="font-medium">{referencePoint.interior_angle || '-'}°</span></div>
                    </div>
                  </div>
                )}
                
                {/* Action Selection Information Card */}
                {selectedAction && (
                  <div className="p-4 border-b border-border-primary">
                    <h4 className="text-sm font-semibold text-text-primary mb-2">Selected Action</h4>
                    <div className="text-sm space-y-1">
                      <div>Type: <span className="font-medium">{selectedAction}</span></div>
                      {clickCoordinates && (
                        <div>Clicked Point: <span className="font-medium">({clickCoordinates[0].toFixed(3)}, {clickCoordinates[1].toFixed(3)})</span></div>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Execution Result Information Card */}
                {executionResult && (
                  <div className="p-4 border-b border-border-primary">
                    <h4 className="text-sm font-semibold text-text-primary mb-2">Execution Result</h4>
                    <div className="text-sm space-y-1">
                      <div>Valid: <span className={`font-medium ${
                        executionResult.is_valid ? 'text-success' : 'text-danger'
                      }`}>{executionResult.is_valid ? 'Yes' : 'No'}</span></div>
                      {executionResult.polar_coordinates && (
                        <div>Polar Coordinates: <span className="font-medium">({executionResult.polar_coordinates.join(', ')})</span></div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Log Area */}
          <div className="h-48 bg-card border-t border-border-primary flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-border-primary">
              <h3 className="text-sm font-semibold text-text-primary">Action Log</h3>
              <div className="flex items-center gap-3">
                <label className="flex items-center text-xs text-text-secondary">
                  <input type="checkbox" className="mr-1" defaultChecked />
                  Auto-scroll
                </label>
                <Button onClick={clearLog} variant="tertiary" size="sm">
                  Clear
                </Button>
              </div>
            </div>
            <div className="flex-1 p-3 overflow-y-auto font-mono text-xs bg-bg-tertiary">
              {log.length === 0 ? (
                <div className="text-text-secondary">Action Tester initialized. Select a mesh to begin testing.</div>
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
      
      {/* Loading Overlay */}
      <LoadingOverlay isVisible={isLoading} message="Processing..." />
    </div>
  );
};

export default ActionTester;
