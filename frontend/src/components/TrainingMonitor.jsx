import React, { useRef, useEffect, useState, useCallback } from 'react';
import MeshCanvas from './MeshCanvas';
import { 
  useMeshBoundary, 
  useMeshData, 
  useReferencePoint, 
  useTrainingStatus 
} from '../hooks/useTrainingHooks';

/**
 * TrainingMonitor - Container component that composes hooks and passes data to presentational components
 * 
 * This component acts as a smart container that:
 * - Uses custom hooks to manage business logic
 * - Coordinates data flow between hooks
 * - Passes data and callbacks to presentational components
 * - Manages canvas interactions and visualization
 */
const TrainingMonitor = () => {
  // Ref to access MeshCanvas imperative methods
  const canvasRef = useRef(null);
  
  // Local UI state
  const [selectedMesh, setSelectedMesh] = useState('');
  const [clickCoordinates, setClickCoordinates] = useState(null);

  // Business logic hooks
  const meshBoundary = useMeshBoundary(selectedMesh);
  const meshData = useMeshData(selectedMesh);
  const referencePoint = useReferencePoint(selectedMesh);
  const trainingStatus = useTrainingStatus({
    polling: true,
    interval: 2000,
    onStatusChange: (newStatus, prevStatus) => {
      console.log('Training status changed:', prevStatus.status, '->', newStatus.status);
    }
  });

  // Derived loading state from all hooks
  const isLoading = meshBoundary.isLoading || meshData.isLoading || referencePoint.isLoading;

  // Handle canvas clicks
  const handleCanvasClick = useCallback((worldCoords, event) => {
    if (worldCoords) {
      setClickCoordinates({
        world: worldCoords,
        screen: [event.clientX, event.clientY],
        timestamp: Date.now()
      });
      console.log('Canvas clicked at world coordinates:', worldCoords);
    }
  }, []);

  // Handle mesh selection change - coordinates multiple hooks
  const handleMeshChange = useCallback((meshName) => {
    setSelectedMesh(meshName);
    setClickCoordinates(null);
    
    // Update training config with new mesh
    trainingStatus.updateConfig({ mesh: meshName });

    // Clear all data when changing mesh
    meshBoundary.clearBoundary();
    meshData.clearMeshData();
    referencePoint.clearReferencePoint();
    
    if (!meshName) {
      // Clear canvas when no mesh selected
      if (canvasRef.current) {
        canvasRef.current.clearCanvas();
      }
      return;
    }

    // Load boundary preview first
    meshBoundary.loadBoundary(meshName);
  }, [meshBoundary, meshData, referencePoint, trainingStatus]);

  // Handle mesh data loading with canvas update
  const handleLoadMeshData = useCallback(() => {
    if (!selectedMesh) return;
    
    meshData.loadMeshData(selectedMesh).then(() => {
      // Auto-update canvas after loading mesh data
      if (canvasRef.current && meshData.meshData) {
        canvasRef.current.renderScene(
          meshData.meshData, 
          meshBoundary.boundaryData, 
          referencePoint.refPointInfo
        );
      }
    });
  }, [selectedMesh, meshData, meshBoundary.boundaryData, referencePoint.refPointInfo]);

  // Handle reference point finding with canvas update
  const handleFindReferencePoint = useCallback(() => {
    if (!selectedMesh) return;
    
    referencePoint.findReferencePoint(selectedMesh).then(() => {
      // Auto-update canvas after finding reference point
      if (canvasRef.current) {
        canvasRef.current.renderScene(
          meshData.meshData, 
          meshBoundary.boundaryData, 
          referencePoint.refPointInfo
        );
      }
    });
  }, [selectedMesh, referencePoint, meshData.meshData, meshBoundary.boundaryData]);

  // Clear canvas and reset all state
  const handleClearCanvas = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
    meshData.clearMeshData();
    meshBoundary.clearBoundary();
    referencePoint.clearReferencePoint();
    setClickCoordinates(null);
  }, [meshData, meshBoundary, referencePoint]);

  // Auto-update canvas when boundary data changes
  useEffect(() => {
    if (canvasRef.current && meshBoundary.boundaryData && selectedMesh) {
      canvasRef.current.renderBoundaryPreview(meshBoundary.boundaryData, selectedMesh);
    }
  }, [meshBoundary.boundaryData, selectedMesh]);

  return (
    <div className="training-monitor max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-2">
          Training Monitor
        </h2>
        <p className="text-text-secondary">
          Monitor reinforcement learning training progress with mesh visualization
        </p>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Mesh Selection */}
        <div className="bg-card border border-border-custom rounded-lg p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-3">Mesh Selection</h3>
          <select
            value={selectedMesh}
            onChange={(e) => handleMeshChange(e.target.value)}
            className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2 text-text-primary"
            disabled={isLoading || trainingStatus.trainingStatus.is_training}
          >
            <option value="">Select a mesh...</option>
            <option value="simple_square">Simple Square</option>
            <option value="complex_polygon">Complex Polygon</option>
            <option value="curved_boundary">Curved Boundary</option>
          </select>
          
          <div className="mt-4 space-y-2">
            <button
              onClick={handleLoadMeshData}
              disabled={!selectedMesh || isLoading || trainingStatus.trainingStatus.is_training}
              className="w-full px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Load Full Mesh Data
            </button>
            <button
              onClick={handleFindReferencePoint}
              disabled={!selectedMesh || isLoading || trainingStatus.trainingStatus.is_training}
              className="w-full px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Find Reference Point
            </button>
          </div>
        </div>

        {/* Training Controls */}
        <div className="bg-card border border-border-custom rounded-lg p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-3">Training Controls</h3>
          <div className="space-y-2">
            <button
              onClick={() => trainingStatus.startTraining({ ...trainingStatus.trainingConfig, mesh: selectedMesh })}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!selectedMesh || isLoading || trainingStatus.trainingStatus.is_training}
            >
              {trainingStatus.trainingStatus.is_training ? 'Training Active...' : 'Start Training'}
            </button>
            <button
              onClick={() => trainingStatus.stopTraining()}
              className="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!trainingStatus.trainingStatus.is_training || isLoading}
            >
              Stop Training
            </button>
            <button
              onClick={handleClearCanvas}
              className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              Clear Canvas
            </button>
          </div>
        </div>

        {/* Status Info */}
        <div className="bg-card border border-border-custom rounded-lg p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-3">Status</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-secondary">Selected Mesh:</span>
              <span className="text-text-primary">{selectedMesh || 'None'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Training Status:</span>
              <span className={trainingStatus.trainingStatus.is_training ? 'text-green-500' : 'text-gray-500'}>
                {trainingStatus.trainingStatus.status}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Boundary Loaded:</span>
              <span className={meshBoundary.boundaryData ? 'text-green-500' : 'text-gray-500'}>
                {meshBoundary.boundaryData ? 'Yes' : 'No'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Mesh Data:</span>
              <span className={meshData.meshData ? 'text-green-500' : 'text-gray-500'}>
                {meshData.meshData ? 'Loaded' : 'None'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Ref Point:</span>
              <span className={referencePoint.refPointInfo ? 'text-green-500' : 'text-gray-500'}>
                {referencePoint.refPointInfo ? 'Found' : 'None'}
              </span>
            </div>
            {clickCoordinates && (
              <div className="mt-2 p-2 bg-bg-secondary rounded">
                <div className="text-text-secondary text-xs">Last Click:</div>
                <div className="text-text-primary text-xs">
                  World: ({clickCoordinates.world[0].toFixed(3)}, {clickCoordinates.world[1].toFixed(3)})
                </div>
                <div className="text-text-secondary text-xs">
                  Time: {new Date(clickCoordinates.timestamp).toLocaleTimeString()}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Canvas Container */}
      <div className="bg-card border border-border-custom rounded-lg overflow-hidden">
        <div className="p-4 border-b border-border-custom">
          <h3 className="text-lg font-semibold text-text-primary">
            Mesh Visualization
          </h3>
        </div>
        
        <div 
          className="relative"
          style={{ 
            height: '600px',
            background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
          }}
        >
          {isLoading && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
              <div className="text-white">Loading...</div>
            </div>
          )}
          
          <MeshCanvas
            ref={canvasRef}
            onCanvasClick={handleCanvasClick}
            className="w-full h-full"
            style={{
              cursor: selectedMesh ? 'crosshair' : 'default',
            }}
          />
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 text-sm text-text-secondary">
        <p>
          • Select a mesh from the dropdown to preview its boundary
        </p>
        <p>
          • Click "Load Full Mesh Data" to visualize the complete mesh structure
        </p>
        <p>
          • Click "Find Reference Point" to locate training reference points
        </p>
        <p>
          • Click on the canvas to get world coordinates for interaction
        </p>
      </div>
    </div>
  );
};

export default TrainingMonitor;
