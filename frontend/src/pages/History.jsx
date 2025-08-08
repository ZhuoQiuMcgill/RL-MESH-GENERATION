import React, { useState, useEffect, useRef } from 'react';
import { NavHeader, MeshCanvas } from '../components';
import { Button, FormInput, FormSelect, LoadingOverlay } from '../components/ui';
import { useApi } from '../context/ApiProvider';

const History = () => {
  // State management
  const [trainingList, setTrainingList] = useState([]);
  const [selectedTraining, setSelectedTraining] = useState(null);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [episodeData, setEpisodeData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [log, setLog] = useState([]);
  const [clickCoordinates, setClickCoordinates] = useState('Not clicked');
  
  // Refs
  const canvasRef = useRef(null);
  const api = useApi();

  // Initialize component
  useEffect(() => {
    loadTrainingHistory();
    addLogEntry('History Viewer initialized', 'info');
  }, []);

  const addLogEntry = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLog(prev => [...prev, { message: `[${timestamp}] ${message}`, type }]);
  };

  const loadTrainingHistory = async () => {
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the API
      // For now, we'll simulate the training history loading
      addLogEntry('Loading training history...', 'info');
      // const response = await api.getTrainingHistory();
      // setTrainingList(response.trainings || []);
      addLogEntry('Training history loaded successfully', 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load training history: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const selectTraining = async (training) => {
    try {
      setIsLoading(true);
      setSelectedTraining(training);
      setCurrentEpisode(0);
      addLogEntry(`Selected training: ${training.id}`, 'info');
      
      // Load first episode
      await loadEpisode(0);
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to select training: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const loadEpisode = async (episodeIndex) => {
    if (!selectedTraining) return;
    
    try {
      setIsLoading(true);
      // Note: This would need to be implemented in the API
      // const response = await api.getEpisodeData(selectedTraining.id, episodeIndex);
      // setEpisodeData(response);
      setCurrentEpisode(episodeIndex);
      
      // Update canvas visualization
      // if (canvasRef.current && response) {
      //   canvasRef.current.renderScene(response.meshData, response.boundaryVertices, response.refPointInfo);
      // }
      
      addLogEntry(`Loaded episode ${episodeIndex}`, 'success');
    } catch (err) {
      setError(err.message);
      addLogEntry(`Failed to load episode: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCanvasClick = (worldCoords, event) => {
    if (worldCoords) {
      const coordsText = `(${worldCoords[0].toFixed(3)}, ${worldCoords[1].toFixed(3)})`;
      setClickCoordinates(coordsText);
      addLogEntry(`Canvas clicked at: ${coordsText}`, 'info');
    } else {
      setClickCoordinates('Invalid coordinates');
    }
  };

  const navigateEpisode = (direction) => {
    if (!selectedTraining) return;
    
    let newEpisode = currentEpisode;
    if (direction === 'next' && currentEpisode < selectedTraining.totalEpisodes - 1) {
      newEpisode = currentEpisode + 1;
    } else if (direction === 'prev' && currentEpisode > 0) {
      newEpisode = currentEpisode - 1;
    }
    
    if (newEpisode !== currentEpisode) {
      loadEpisode(newEpisode);
    }
  };

  const goToEpisode = () => {
    const episodeInput = document.getElementById('episode-index-input');
    const episodeIndex = parseInt(episodeInput?.value || 0);
    
    if (episodeIndex >= 0 && episodeIndex < (selectedTraining?.totalEpisodes || 0)) {
      loadEpisode(episodeIndex);
    }
  };

  const clearLog = () => {
    setLog([]);
    addLogEntry('Log cleared', 'info');
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      <NavHeader 
        title="History Viewer"
        breadcrumbs={[
          { label: 'Tools', href: '/' },
          { label: 'History Viewer', href: '/history' }
        ]}
      />
      
      {/* Main Container */}
      <div className="flex min-h-[calc(100vh-var(--nav-header-height))] bg-bg-primary">
        {/* Left Training List Panel */}
        <div className="w-80 min-w-80 bg-card border-r border-border-primary overflow-y-auto flex-shrink-0">
          <div className="p-6 border-b border-gray-200">
            <h1 className="text-2xl font-bold text-text-primary">Training History</h1>
            <p className="text-sm text-text-secondary mt-1">View past training sessions and details</p>
          </div>
          
          <div className="flex-1 p-6 flex flex-col overflow-hidden">
            {/* Action Buttons */}
            <div className="mb-6 space-y-3">
              <Button 
                onClick={loadTrainingHistory} 
                variant="primary" 
                className="w-full"
                disabled={isLoading}
              >
                Refresh List
              </Button>
              <Button 
                onClick={() => api.checkTrainingHealth()} 
                variant="secondary" 
                className="w-full"
                disabled={isLoading}
              >
                Check Service
              </Button>
            </div>
            
            {/* Training Session List */}
            <div className="mb-6 flex-1 overflow-y-auto">
              <h3 className="text-sm font-medium text-text-primary mb-3">Training Sessions</h3>
              <div className="space-y-2">
                {trainingList.length === 0 ? (
                  <div className="text-center text-text-secondary py-4">
                    {isLoading ? (
                      <div>
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto mb-2"></div>
                        <div>Loading...</div>
                      </div>
                    ) : (
                      'No training sessions found'
                    )}
                  </div>
                ) : (
                  trainingList.map((training) => (
                    <button
                      key={training.id}
                      onClick={() => selectTraining(training)}
                      className={`w-full text-left p-3 rounded-lg border transition-colors ${
                        selectedTraining?.id === training.id
                          ? 'bg-primary text-white border-primary'
                          : 'bg-card border-border-primary hover:bg-bg-hover'
                      }`}
                    >
                      <div className="text-sm font-medium">{training.name}</div>
                      <div className="text-xs opacity-75">Episodes: {training.totalEpisodes}</div>
                    </button>
                  ))
                )}
              </div>
            </div>
            
            {/* Current Selected Training Info */}
            {selectedTraining && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-text-primary mb-3">Training Summary</h3>
                <div className="rounded-lg p-4 border bg-card border-border-primary space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Training ID:</span>
                    <span className="font-medium text-text-primary text-xs">{selectedTraining.id}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Total Episodes:</span>
                    <span className="font-medium text-text-primary">{selectedTraining.totalEpisodes}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Best Episode:</span>
                    <span className="font-medium text-text-primary">{selectedTraining.bestEpisode || 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Episode Navigation */}
            {selectedTraining && (
              <div className="mt-6">
                <h3 className="text-sm font-medium text-text-primary mb-3">Episode Navigation</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <FormInput
                      id="episode-index-input"
                      type="number"
                      min="0"
                      max={selectedTraining.totalEpisodes - 1}
                      value={currentEpisode}
                      className="flex-1 text-sm"
                      onChange={(e) => setCurrentEpisode(parseInt(e.target.value) || 0)}
                    />
                    <Button onClick={goToEpisode} variant="secondary" size="sm">
                      View
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button onClick={() => loadEpisode(selectedTraining.bestEpisode || 0)} variant="success" size="sm">
                      Best Episode
                    </Button>
                    <Button onClick={() => loadEpisode(selectedTraining.totalEpisodes - 1)} variant="warning" size="sm">
                      Last Episode
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button 
                      onClick={() => navigateEpisode('prev')} 
                      variant="secondary" 
                      size="sm"
                      disabled={currentEpisode <= 0}
                    >
                      ← Previous
                    </Button>
                    <Button 
                      onClick={() => navigateEpisode('next')} 
                      variant="secondary" 
                      size="sm"
                      disabled={currentEpisode >= (selectedTraining.totalEpisodes - 1)}
                    >
                      Next →
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Main Content Container */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Canvas Visualization Area */}
          <div className="flex-1 bg-card">
            <div className="h-full rounded-lg shadow-sm border p-1 bg-bg-canvas border-border-primary">
              <div className="relative h-full">
                <MeshCanvas 
                  ref={canvasRef}
                  onCanvasClick={handleCanvasClick}
                  className="w-full h-full rounded border border-border-primary"
                />
              </div>
            </div>
          </div>
          
          {/* Right Data Panel */}
          <div className="w-80 min-w-80 bg-card border-l border-border-primary flex flex-col overflow-hidden">
            {/* Episode Details Header */}
            <div className="p-4 border-b border-border-primary">
              <h2 className="text-lg font-semibold text-text-primary">Episode Details</h2>
              <p className="text-sm text-text-secondary">
                {selectedTraining ? `Episode ${currentEpisode} of ${selectedTraining.totalEpisodes - 1}` : 'No Episode Selected'}
              </p>
            </div>
            
            {/* Episode Information Section */}
            <div className="p-4 border-b border-border-primary">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Episode Information</h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Index:</span>
                  <span className="text-text-primary font-medium">{currentEpisode}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Reward:</span>
                  <span className="text-text-primary font-medium">{episodeData?.reward?.toFixed(3) || '0.000'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Length:</span>
                  <span className="text-text-primary font-medium">{episodeData?.length || '0'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Status:</span>
                  <span className="text-text-primary font-medium">{episodeData?.status || 'Unknown'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Boundary Vertices:</span>
                  <span className="text-text-primary font-medium">{episodeData?.boundaryVerticesCount || '0'}</span>
                </div>
              </div>
            </div>
            
            {/* Canvas Interaction Section */}
            <div className="p-4 border-b border-border-primary">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Canvas Interaction</h3>
              <div className="text-sm">
                <span className="text-text-secondary">Click Coordinates: </span>
                <span className="font-semibold text-primary">{clickCoordinates}</span>
              </div>
            </div>
            
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
                  <div className="text-text-secondary">System ready, waiting for actions...</div>
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
      <LoadingOverlay isVisible={isLoading} message="Loading..." />
    </div>
  );
};

export default History;
