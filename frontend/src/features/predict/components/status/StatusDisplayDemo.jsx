import React, { useEffect } from 'react';
import { PredictSessionProvider, usePredictSession } from '../../contexts/PredictSessionContext';
import StatusDisplay from './StatusDisplay';

// Demo component that simulates prediction data
const StatusDisplayDemo = () => {
  const { actions } = usePredictSession();

  useEffect(() => {
    // Simulate session creation
    actions.createSession('demo-session-123');
    
    // Add some demo data
    setTimeout(() => {
      actions.setRefPoint({
        index: 10,
        coordinates: [600.0, 300.0],
        interior_angle: 89.99,
        neighbor_vertices: [
          [593.338, 206.662],
          [693.338, 393.338],
          [600.0, 300.0],
          [506.662, 206.662],
          [400.0, 400.0]
        ]
      });
    }, 1000);

    setTimeout(() => {
      actions.startPrediction({ totalSteps: 100 });
    }, 1500);

    setTimeout(() => {
      actions.addLog({
        level: 'info',
        message: 'Action executed',
        data: {
          action_type: 'type1',
          reference_vertex_idx: 10,
          clicked_point: [0.5, 0.5],
          decoded_coords: [[0.5, 0.5]],
          generated_element: [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
          polar_coordinates: { r: 0.707, theta: 0.785 },
          valid: true,
          quality: 0.85
        }
      });
    }, 2000);

    setTimeout(() => {
      actions.nextStep({
        meshData: {
          elements: [
            { quality: 0.92 },
            { quality: 0.78 },
            { quality: 0.65 },
            { quality: 0.88 }
          ],
          quality: 0.81,
          qualityMetrics: {
            average: 0.81,
            min: 0.65,
            max: 0.92,
            median: 0.83
          }
        }
      });
    }, 2500);

    // Simulate continuous updates
    const interval = setInterval(() => {
      actions.nextStep({
        meshData: {
          elements: Array.from({ length: Math.floor(Math.random() * 10) + 5 }, () => ({
            quality: Math.random() * 0.4 + 0.6 // Quality between 0.6 and 1.0
          })),
          quality: Math.random() * 0.3 + 0.7,
          qualityMetrics: {
            average: Math.random() * 0.3 + 0.7,
            min: Math.random() * 0.3 + 0.4,
            max: Math.random() * 0.1 + 0.9,
            median: Math.random() * 0.3 + 0.65
          }
        }
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [actions]);

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            StatusDisplay Demo
          </h1>
          <p className="text-gray-600">
            Demonstrating the StatusDisplay component group with simulated prediction data.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <StatusDisplay />
        </div>

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Alternative Layouts</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-lg font-medium mb-3">Column Layout</h3>
              <StatusDisplay layout="column" className="max-w-md" />
            </div>
            
            <div>
              <h3 className="text-lg font-medium mb-3">Row Layout</h3>
              <StatusDisplay layout="row" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main demo component wrapped with provider
const StatusDisplayDemoWrapper = () => {
  return (
    <PredictSessionProvider>
      <StatusDisplayDemo />
    </PredictSessionProvider>
  );
};

export default StatusDisplayDemoWrapper;
