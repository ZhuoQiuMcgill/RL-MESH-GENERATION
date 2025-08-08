import React, { useRef, useEffect, useState, useCallback } from 'react';
import { 
  Button, 
  PanelCard, 
  CompactStatusBar, 
  FormSelect,
  LoadingOverlay 
} from '../components/ui';
import MeshCanvas from '../components/MeshCanvas';
import { useApi, usePolling } from '../context/ApiProvider';

/**
 * TrainingMonitor - Full-featured training monitor page using UI primitives
 * 
 * Features:
 * - Left panel with mesh selection and training controls
 * - Status bar showing training status and metrics
 * - Interactive mesh canvas
 * - Live metrics panel with polling updates
 * - Proper React state management replacing DOM event listeners
 * - useEffect-based timers replacing vanilla JS timers
 */
const TrainingMonitor = () => {
  // Refs
  const canvasRef = useRef(null);
  const updateIntervalRef = useRef(null);
  const immediateUpdateTimerRef = useRef(null);

  // Component state
  const [selectedMesh, setSelectedMesh] = useState('');
  const [meshData, setMeshData] = useState(null);
  const [boundaryData, setBoundaryData] = useState(null);
  const [refPointInfo, setRefPointInfo] = useState(null);
  const [clickCoordinates, setClickCoordinates] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Training state
  const [trainingStatus, setTrainingStatus] = useState({
    is_training: false,
    status: 'idle',
    episode: 0,
    total_episodes: 0,
    current_reward: 0,
    best_reward: 0,
    elapsed_time: 0,
    last_updated: null
  });
  
  // Metrics state
  const [trainingMetrics, setTrainingMetrics] = useState({
    episode_rewards: [],
    average_reward: 0,
    loss_values: [],
    learning_rate: 0,
    exploration_rate: 0
  });

  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState({
    algorithm: 'PPO',
    episodes: 1000,
    learning_rate: 0.001,
    mesh: ''
  });

  // UI state
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [updateInterval, setUpdateInterval] = useState(2000); // 2 seconds
  const [showMetrics, setShowMetrics] = useState(true);

  // Get API client
  const api = useApi();

  // Polling for training status when training is active
  const {
    data: statusData,
    isPolling: isPollingStatus,
    startPolling: startStatusPolling,
    stopPolling: stopStatusPolling
  } = usePolling('getTrainingStatus', updateInterval, {
    enabled: autoUpdate && trainingStatus.is_training,
    onSuccess: (data) => {
      if (data.success) {
        setTrainingStatus(data.status);
      }
    },
    onError: (error) => {
      console.error('Failed to poll training status:', error);
    }
  });

  // Handle canvas clicks with React state
  const handleCanvasClick = useCallback((worldCoords, event) => {
    if (worldCoords) {
      setClickCoordinates({
        world: worldCoords,
        screen: [event.clientX, event.clientY],
        timestamp: Date.now()
      });
      console.log('Canvas clicked at world coordinates:', worldCoords);
      
      // Trigger immediate update if training is active
      if (trainingStatus.is_training) {
        triggerImmediateUpdate();
      }
    }
  }, [trainingStatus.is_training]);

  // Immediate update timer implementation with useEffect
  const triggerImmediateUpdate = useCallback(() => {
    // Clear existing immediate update timer
    if (immediateUpdateTimerRef.current) {
      clearTimeout(immediateUpdateTimerRef.current);
    }

    // Set new immediate update timer
    immediateUpdateTimerRef.current = setTimeout(async () => {
      if (trainingStatus.is_training) {
        try {
          const statusResponse = await api.getTrainingStatus();
          if (statusResponse.success) {
            setTrainingStatus(statusResponse.status);
          }
        } catch (error) {
          console.error('Immediate update failed:', error);
        }
      }
    }, 500); // Quick update after 500ms
  }, [api, trainingStatus.is_training]);

  // Update interval timer implementation with useEffect
  useEffect(() => {
    if (autoUpdate && trainingStatus.is_training) {
      updateIntervalRef.current = setInterval(async () => {
        try {
          // Get updated training metrics
          const metricsResponse = await api.getTrainingStatus();
          if (metricsResponse.success) {
            setTrainingMetrics(prev => ({
              ...prev,
              episode_rewards: metricsResponse.metrics?.episode_rewards || prev.episode_rewards,
              average_reward: metricsResponse.metrics?.average_reward || prev.average_reward,
              loss_values: metricsResponse.metrics?.loss_values || prev.loss_values,
              learning_rate: metricsResponse.metrics?.learning_rate || prev.learning_rate,
              exploration_rate: metricsResponse.metrics?.exploration_rate || prev.exploration_rate
            }));
          }
        } catch (error) {
          console.error('Metrics update failed:', error);
        }
      }, updateInterval);
    } else {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
        updateIntervalRef.current = null;
      }
    }

    // Cleanup function
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [autoUpdate, trainingStatus.is_training, updateInterval, api]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      if (immediateUpdateTimerRef.current) {
        clearTimeout(immediateUpdateTimerRef.current);
      }
    };
  }, []);

  // Load mesh boundary data
  const loadMeshBoundary = useCallback(async (meshName) => {
    if (!meshName) return;
    
    setIsLoading(true);
    try {
      const data = await api.getMeshBoundary(meshName);
      
      if (data.success) {
        setBoundaryData(data.boundary_vertices);
        
        // Use canvas imperative method to render boundary preview
        if (canvasRef.current) {
          canvasRef.current.renderBoundaryPreview(data.boundary_vertices, meshName);
        }
      }
    } catch (error) {
      console.error('Failed to load mesh boundary:', error);
    } finally {
      setIsLoading(false);
    }
  }, [api]);

  // Load mesh data for training visualization
  const loadMeshData = useCallback(async (meshName) => {
    if (!meshName) return;
    
    setIsLoading(true);
    try {
      const data = await api.getMeshData(meshName);
      
      if (data.success) {
        setMeshData(data.mesh_data);
        
        // Render full scene with mesh data
        if (canvasRef.current) {
          canvasRef.current.renderScene(data.mesh_data, boundaryData, refPointInfo);
        }
      }
    } catch (error) {
      console.error('Failed to load mesh data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [api, boundaryData, refPointInfo]);

  // Handle mesh selection change
  const handleMeshChange = useCallback((value) => {
    setSelectedMesh(value);
    setMeshData(null);
    setBoundaryData(null);
    setRefPointInfo(null);
    setClickCoordinates(null);
    
    // Update training config
    setTrainingConfig(prev => ({ ...prev, mesh: value }));

    if (!value) {
      // Clear canvas when no mesh selected
      if (canvasRef.current) {
        canvasRef.current.clearCanvas();
      }
      return;
    }

    // Load boundary preview first
    loadMeshBoundary(value);
  }, [loadMeshBoundary]);

  // Find reference point
  const findReferencePoint = useCallback(async () => {
    if (!selectedMesh) return;
    
    setIsLoading(true);
    try {
      const data = await api.getTrainingReferencePoint({ mesh: selectedMesh });
      
      if (data.success) {
        setRefPointInfo(data.reference_point);
        
        // Re-render scene with reference point
        if (canvasRef.current) {
          canvasRef.current.renderScene(meshData, boundaryData, data.reference_point);
        }
      }
    } catch (error) {
      console.error('Failed to find reference point:', error);
    } finally {
      setIsLoading(false);
    }
  }, [api, selectedMesh, meshData, boundaryData]);

  // Start training
  const handleStartTraining = useCallback(async () => {
    if (!selectedMesh) return;
    
    setIsLoading(true);
    try {
      const config = {
        ...trainingConfig,
        mesh: selectedMesh
      };
      
      const response = await api.startTraining(config);
      
      if (response.success) {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          status: 'training'
        }));
        
        // Start polling for updates
        if (autoUpdate) {
          startStatusPolling();
        }
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    } finally {
      setIsLoading(false);
    }
  }, [api, selectedMesh, trainingConfig, autoUpdate, startStatusPolling]);

  // Stop training
  const handleStopTraining = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await api.stopTraining();
      
      if (response.success) {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: false,
          status: 'stopped'
        }));
        
        // Stop polling
        stopStatusPolling();
      }
    } catch (error) {
      console.error('Failed to stop training:', error);
    } finally {
      setIsLoading(false);
    }
  }, [api, stopStatusPolling]);

  // Clear canvas and reset state
  const clearCanvas = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
    setMeshData(null);
    setBoundaryData(null);
    setRefPointInfo(null);
    setClickCoordinates(null);
  }, []);

  // Handle checkbox changes (replaces DOM event listeners)
  const handleAutoUpdateChange = useCallback((event) => {
    const checked = event.target.checked;
    setAutoUpdate(checked);
    
    if (checked && trainingStatus.is_training) {
      startStatusPolling();
    } else {
      stopStatusPolling();
    }
  }, [trainingStatus.is_training, startStatusPolling, stopStatusPolling]);

  // Handle select changes (replaces DOM event listeners)
  const handleUpdateIntervalChange = useCallback((event) => {
    const interval = parseInt(event.target.value, 10);
    setUpdateInterval(interval);
  }, []);

  const handleTrainingConfigChange = useCallback((field, value) => {
    setTrainingConfig(prev => ({ ...prev, [field]: value }));
  }, []);

  // Prepare status items for CompactStatusBar
  const statusItems = [
    {
      label: 'Selected Mesh',
      value: selectedMesh || 'None'
    },
    {
      label: 'Training Status',
      value: trainingStatus.status,
      color: trainingStatus.is_training ? 'green-500' : 'gray-500'
    },
    {
      label: 'Episode',
      value: trainingStatus.total_episodes > 0 
        ? `${trainingStatus.episode}/${trainingStatus.total_episodes}` 
        : 'N/A'
    },
    {
      label: 'Current Reward',
      value: trainingStatus.current_reward?.toFixed(2) || '0.00'
    },
    {
      label: 'Best Reward',
      value: trainingStatus.best_reward?.toFixed(2) || '0.00',
      color: trainingStatus.best_reward > 0 ? 'green-500' : 'gray-500'
    },
    {
      label: 'Boundary Loaded',
      value: boundaryData ? 'Yes' : 'No',
      color: boundaryData ? 'green-500' : 'gray-500'
    },
    {
      label: 'Mesh Data',
      value: meshData ? 'Loaded' : 'None',
      color: meshData ? 'green-500' : 'gray-500'
    },
    {
      label: 'Ref Point',
      value: refPointInfo ? 'Found' : 'None',
      color: refPointInfo ? 'green-500' : 'gray-500'
    }
  ];

  // Prepare metrics items
  const metricsItems = [
    {
      label: 'Average Reward',
      value: trainingMetrics.average_reward?.toFixed(2) || '0.00'
    },
    {
      label: 'Learning Rate',
      value: trainingMetrics.learning_rate?.toFixed(6) || '0.000000'
    },
    {
      label: 'Exploration Rate',
      value: trainingMetrics.exploration_rate?.toFixed(3) || '0.000'
    },
    {
      label: 'Recent Episodes',
      value: trainingMetrics.episode_rewards?.length || 0
    },
    {
      label: 'Loss Values',
      value: trainingMetrics.loss_values?.length || 0
    }
  ];

  const meshOptions = [
    { value: '', label: 'Select a mesh...' },
    { value: 'simple_square', label: 'Simple Square' },
    { value: 'complex_polygon', label: 'Complex Polygon' },
    { value: 'curved_boundary', label: 'Curved Boundary' },
    { value: 'triangular_mesh', label: 'Triangular Mesh' },
    { value: 'hexagonal_pattern', label: 'Hexagonal Pattern' }
  ];

  const algorithmOptions = [
    { value: 'PPO', label: 'PPO (Proximal Policy Optimization)' },
    { value: 'SAC', label: 'SAC (Soft Actor-Critic)' },
    { value: 'TD3', label: 'TD3 (Twin Delayed Deep Deterministic)' },
    { value: 'A2C', label: 'A2C (Advantage Actor-Critic)' }
  ];

  return (
    <div className="training-monitor max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-text-primary mb-2">
          Training Monitor
        </h1>
        <p className="text-text-secondary">
          Monitor and control reinforcement learning training sessions with real-time mesh visualization
        </p>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Left Panel - Controls */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Mesh Selection Panel */}
          <PanelCard title="Mesh Selection">
            <FormSelect
              value={selectedMesh}
              onChange={(e) => handleMeshChange(e.target.value)}
              options={meshOptions}
              disabled={isLoading || trainingStatus.is_training}
              placeholder="Select a mesh..."
            />
            
            <div className="mt-4 space-y-2">
              <Button
                onClick={() => loadMeshData(selectedMesh)}
                disabled={!selectedMesh || isLoading || trainingStatus.is_training}
                className="w-full"
                variant="primary"
              >
                Load Full Mesh Data
              </Button>
              <Button
                onClick={findReferencePoint}
                disabled={!selectedMesh || isLoading || trainingStatus.is_training}
                className="w-full"
                variant="secondary"
              >
                Find Reference Point
              </Button>
            </div>
          </PanelCard>

          {/* Training Configuration Panel */}
          <PanelCard title="Training Configuration">
            <div className="space-y-4">
              <FormSelect
                label="Algorithm"
                value={trainingConfig.algorithm}
                onChange={(e) => handleTrainingConfigChange('algorithm', e.target.value)}
                options={algorithmOptions}
                disabled={trainingStatus.is_training}
              />
              
              <div className="space-y-2">
                <label className="block text-text-secondary text-sm font-medium">
                  Episodes
                </label>
                <input
                  type="number"
                  value={trainingConfig.episodes}
                  onChange={(e) => handleTrainingConfigChange('episodes', parseInt(e.target.value, 10))}
                  disabled={trainingStatus.is_training}
                  className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2 text-text-primary"
                  min="1"
                  max="10000"
                />
              </div>
              
              <div className="space-y-2">
                <label className="block text-text-secondary text-sm font-medium">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => handleTrainingConfigChange('learning_rate', parseFloat(e.target.value))}
                  disabled={trainingStatus.is_training}
                  className="w-full bg-bg-secondary border border-border-custom rounded-lg px-3 py-2 text-text-primary"
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                />
              </div>
            </div>
          </PanelCard>

          {/* Training Controls Panel */}
          <PanelCard title="Training Controls">
            <div className="space-y-3">
              <Button
                onClick={handleStartTraining}
                disabled={!selectedMesh || isLoading || trainingStatus.is_training}
                className="w-full"
                variant="success"
              >
                {trainingStatus.is_training ? 'Training Active...' : 'Start Training'}
              </Button>
              
              <Button
                onClick={handleStopTraining}
                disabled={!trainingStatus.is_training || isLoading}
                className="w-full"
                variant="danger"
              >
                Stop Training
              </Button>
              
              <Button
                onClick={clearCanvas}
                className="w-full"
                variant="outline"
              >
                Clear Canvas
              </Button>
            </div>
          </PanelCard>

          {/* Update Settings Panel */}
          <PanelCard title="Update Settings">
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="auto-update"
                  checked={autoUpdate}
                  onChange={handleAutoUpdateChange}
                  className="rounded"
                />
                <label htmlFor="auto-update" className="text-text-primary text-sm">
                  Auto Update
                </label>
              </div>
              
              <FormSelect
                label="Update Interval"
                value={updateInterval}
                onChange={handleUpdateIntervalChange}
                options={[
                  { value: 1000, label: '1 second' },
                  { value: 2000, label: '2 seconds' },
                  { value: 5000, label: '5 seconds' },
                  { value: 10000, label: '10 seconds' }
                ]}
              />
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="show-metrics"
                  checked={showMetrics}
                  onChange={(e) => setShowMetrics(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="show-metrics" className="text-text-primary text-sm">
                  Show Metrics Panel
                </label>
              </div>
            </div>
          </PanelCard>
        </div>

        {/* Center Panel - Canvas and Status */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Status Bar */}
          <PanelCard title="System Status">
            <CompactStatusBar items={statusItems} />
            
            {clickCoordinates && (
              <div className="mt-4 p-3 bg-bg-secondary rounded-lg">
                <div className="text-text-secondary text-xs mb-1">Last Canvas Click:</div>
                <div className="text-text-primary text-sm">
                  World: ({clickCoordinates.world[0].toFixed(3)}, {clickCoordinates.world[1].toFixed(3)})
                </div>
                <div className="text-text-secondary text-xs">
                  Time: {new Date(clickCoordinates.timestamp).toLocaleTimeString()}
                </div>
              </div>
            )}
          </PanelCard>

          {/* Mesh Canvas Panel */}
          <PanelCard title="Mesh Visualization">
            <div 
              className="relative rounded-lg overflow-hidden"
              style={{ 
                height: '500px',
                background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
              }}
            >
              {isLoading && (
                <LoadingOverlay 
                  text="Loading mesh data..." 
                  className="absolute inset-0"
                  size="lg"
                />
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
          </PanelCard>
        </div>

        {/* Right Panel - Metrics */}
        {showMetrics && (
          <div className="lg:col-span-1">
            <PanelCard title="Training Metrics">
              <CompactStatusBar items={metricsItems} />
              
              {trainingStatus.is_training && (
                <div className="mt-4">
                  <div className="text-text-secondary text-xs mb-2">Training Progress</div>
                  <div className="w-full bg-bg-secondary rounded-full h-2">
                    <div 
                      className="bg-accent h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${trainingStatus.total_episodes > 0 
                          ? (trainingStatus.episode / trainingStatus.total_episodes) * 100 
                          : 0}%`
                      }}
                    />
                  </div>
                  <div className="text-text-secondary text-xs mt-1">
                    {trainingStatus.total_episodes > 0 
                      ? `${Math.round((trainingStatus.episode / trainingStatus.total_episodes) * 100)}% Complete`
                      : 'Starting...'}
                  </div>
                </div>
              )}
              
              {trainingMetrics.episode_rewards.length > 0 && (
                <div className="mt-4">
                  <div className="text-text-secondary text-xs mb-2">Recent Rewards</div>
                  <div className="space-y-1">
                    {trainingMetrics.episode_rewards.slice(-5).map((reward, index) => (
                      <div key={index} className="flex justify-between text-xs">
                        <span className="text-text-secondary">Episode {index + 1}:</span>
                        <span className="text-text-primary">{reward?.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </PanelCard>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="mt-6 text-sm text-text-secondary bg-card border border-border-custom rounded-lg p-4">
        <h4 className="font-semibold mb-2">Instructions:</h4>
        <ul className="space-y-1">
          <li>• Select a mesh from the dropdown to preview its boundary</li>
          <li>• Configure training parameters (algorithm, episodes, learning rate)</li>
          <li>• Click "Load Full Mesh Data" to visualize the complete mesh structure</li>
          <li>• Click "Find Reference Point" to locate training reference points</li>
          <li>• Click "Start Training" to begin a training session</li>
          <li>• Click on the canvas to interact and get world coordinates</li>
          <li>• Use the update settings to control real-time monitoring</li>
          <li>• Training metrics will update automatically during active training</li>
        </ul>
      </div>
    </div>
  );
};

export default TrainingMonitor;
