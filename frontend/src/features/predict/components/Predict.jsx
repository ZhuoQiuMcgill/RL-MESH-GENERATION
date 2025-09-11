import React, { useState, useRef } from 'react';
import { predictionService } from '../../../shared/api';
import { LoadingSpinner, ErrorMessage } from '../../../shared/components';
import Button from '../../../components/Button';
import './PredictLayout.css';

// Predict component with form handling and basic functionality
const Predict = () => {
  // State management
  const [formData, setFormData] = useState({
    modelId: 'rl-mesh-v1.2',
    inputDimensions: {
      width: 100,
      height: 100,
      depth: 50
    },
    quality: 'medium',
    iterations: 1000,
    learningRate: 0.001
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [predictionResults, setPredictionResults] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const fileInputRef = useRef(null);

  // Form handlers
  const handleInputChange = (field, value) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setFormData(prev => ({
        ...prev,
        [parent]: {
          ...prev[parent],
          [child]: value
        }
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [field]: value
      }));
    }
    setError(null); // Clear error when user makes changes
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);
    
    try {
      await predictionService.uploadMeshFile(file, setUploadProgress);
      setUploadedFile(file);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  const handleStartPrediction = async () => {
    setIsLoading(true);
    setError(null);
    setPredictionResults(null);

    // Validate form data
    const validation = predictionService.validateParams(formData);
    if (!validation.isValid) {
      setError(validation.errors.join(', '));
      setIsLoading(false);
      return;
    }

    try {
      const result = await predictionService.startPrediction(formData);
      setPredictionResults(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearError = () => {
    setError(null);
  };

  const handleReset = () => {
    setFormData({
      modelId: 'rl-mesh-v1.2',
      inputDimensions: { width: 100, height: 100, depth: 50 },
      quality: 'medium',
      iterations: 1000,
      learningRate: 0.001
    });
    setPredictionResults(null);
    setUploadedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="predict-layout">
      {/* Left Control Panel */}
      <div className="predict-control-panel theme-fade-in">
        <div className="predict-panel-content">
          <div className="space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Parameters</h2>
            
            {/* Error display */}
            {error && (
              <ErrorMessage
                message={error}
                type="error"
                onDismiss={handleClearError}
                className="mb-4"
              />
            )}
            
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model ID
              </label>
              <select
                value={formData.modelId}
                onChange={(e) => handleInputChange('modelId', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="rl-mesh-v1.0">RL Mesh v1.0</option>
                <option value="rl-mesh-v1.1">RL Mesh v1.1</option>
                <option value="rl-mesh-v1.2">RL Mesh v1.2 (Latest)</option>
              </select>
            </div>

            {/* Input Dimensions */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Input Dimensions
              </label>
              <div className="grid grid-cols-3 gap-2">
                <input
                  type="number"
                  placeholder="Width"
                  value={formData.inputDimensions.width}
                  onChange={(e) => handleInputChange('inputDimensions.width', parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="number"
                  placeholder="Height"
                  value={formData.inputDimensions.height}
                  onChange={(e) => handleInputChange('inputDimensions.height', parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="number"
                  placeholder="Depth"
                  value={formData.inputDimensions.depth}
                  onChange={(e) => handleInputChange('inputDimensions.depth', parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Quality Setting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Quality
              </label>
              <select
                value={formData.quality}
                onChange={(e) => handleInputChange('quality', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            {/* Iterations */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Iterations
              </label>
              <input
                type="number"
                value={formData.iterations}
                onChange={(e) => handleInputChange('iterations', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.001"
                value={formData.learningRate}
                onChange={(e) => handleInputChange('learningRate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Mesh File (Optional)
              </label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".obj,.stl,.ply"
                onChange={handleFileUpload}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {uploadProgress > 0 && uploadProgress < 100 && (
                <div className="mt-2">
                  <LoadingSpinner progress={uploadProgress} size="sm" message="Uploading..." />
                </div>
              )}
              {uploadedFile && (
                <p className="mt-2 text-sm text-green-600">✓ {uploadedFile.name} uploaded</p>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-3 pt-4">
              <Button
                onClick={handleStartPrediction}
                disabled={isLoading}
                className="flex-1"
              >
                {isLoading ? 'Processing...' : 'Start Prediction'}
              </Button>
              <Button
                onClick={handleReset}
                variant="secondary"
                disabled={isLoading}
              >
                Reset
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Center Canvas Area with Control Bar */}
      <div className="predict-center-area">
        <div className="predict-canvas-with-controls">
          <div className="predict-canvas-container theme-fade-in">
            <div className="predict-canvas-wrapper">
              {/* Canvas Placeholder */}
              <div className="w-full h-full bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                {isLoading ? (
                  <LoadingSpinner size="lg" message="Generating mesh prediction..." />
                ) : predictionResults ? (
                  <div className="text-center">
                    <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Prediction Results</h3>
                    <p className="text-sm text-gray-600 mb-4">3D mesh visualization will be displayed here</p>
                    <p className="text-xs text-gray-500">Status: {predictionResults.status}</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Canvas Preview</h3>
                    <p className="text-sm text-gray-600">3D mesh visualization will appear here after starting prediction</p>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Control Bar - same width as canvas */}
          <div className="predict-control-bar theme-fade-in">
            <div className="predict-control-bar-content">
              <div className="predict-bar-section">
                <span className="predict-bar-label">View Controls</span>
                <div className="flex space-x-2 ml-4">
                  <button className="px-3 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200" disabled={!predictionResults}>Rotate</button>
                  <button className="px-3 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200" disabled={!predictionResults}>Zoom</button>
                  <button className="px-3 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200" disabled={!predictionResults}>Pan</button>
                </div>
              </div>
              <div className="predict-bar-section">
                <span className="predict-bar-label">Interaction Tools</span>
                <div className="flex space-x-2 ml-4">
                  <button className="px-3 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200" disabled={!predictionResults}>Measure</button>
                  <button className="px-3 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200" disabled={!predictionResults}>Analyze</button>
                </div>
              </div>
              <div className="predict-bar-section">
                <span className="predict-bar-label">Export Options</span>
                <div className="flex space-x-2 ml-4">
                  <button className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200" disabled={!predictionResults}>Export OBJ</button>
                  <button className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200" disabled={!predictionResults}>Export STL</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Data Display Panel */}
      <div className="predict-data-panel theme-fade-in">
        <div className="predict-panel-content">
          <div className="space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Data</h2>
            
            {predictionResults ? (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-green-800 mb-2">Status</h3>
                  <p className="text-sm text-green-700">{predictionResults.status}</p>
                </div>
                
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-blue-800 mb-2">Prediction ID</h3>
                  <p className="text-sm font-mono text-blue-700">{predictionResults.id}</p>
                </div>
                
                {predictionResults.estimated_time && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-yellow-800 mb-2">Estimated Time</h3>
                    <p className="text-sm text-yellow-700">{predictionResults.estimated_time} seconds</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <p className="text-sm">Prediction data will appear here after starting a prediction</p>
              </div>
            )}
            
            {/* Current Parameters Display */}
            <div className="mt-8">
              <h3 className="text-md font-medium text-gray-900 mb-3">Current Parameters</h3>
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs">
                <pre className="text-gray-700">{JSON.stringify(formData, null, 2)}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
