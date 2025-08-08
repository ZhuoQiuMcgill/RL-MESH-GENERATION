import React, { useState, useRef, useCallback, useEffect } from 'react'
import MeshCanvas from '../components/MeshCanvas'
import { PageHeader, Card, Button, Select, Skeleton } from '../shared/ui'
import { 
  useMeshBoundary, 
  useMeshData, 
  useReferencePoint, 
  useTrainingStatus 
} from '../hooks/useTrainingHooks'

const Train = () => {
  // Refs and local state
  const canvasRef = useRef(null)
  const [selectedMesh, setSelectedMesh] = useState('')
  const [clickCoordinates, setClickCoordinates] = useState(null)

  // Business logic hooks
  const meshBoundary = useMeshBoundary(selectedMesh)
  const meshData = useMeshData(selectedMesh)
  const referencePoint = useReferencePoint(selectedMesh)
  const trainingStatus = useTrainingStatus({
    polling: true,
    interval: 2000,
    onStatusChange: (newStatus, prevStatus) => {
      console.log('Training status changed:', prevStatus.status, '->', newStatus.status)
    }
  })

  // Derived loading state
  const isLoading = meshBoundary.isLoading || meshData.isLoading || referencePoint.isLoading
  const isTraining = trainingStatus.trainingStatus.is_training

  // Handle canvas clicks
  const handleCanvasClick = useCallback((worldCoords, event) => {
    if (worldCoords) {
      setClickCoordinates({
        world: worldCoords,
        screen: [event.clientX, event.clientY],
        timestamp: Date.now()
      })
      console.log('Canvas clicked at world coordinates:', worldCoords)
    }
  }, [])

  // Handle mesh selection change
  const handleMeshChange = useCallback((meshName) => {
    setSelectedMesh(meshName)
    setClickCoordinates(null)
    
    trainingStatus.updateConfig({ mesh: meshName })
    meshBoundary.clearBoundary()
    meshData.clearMeshData()
    referencePoint.clearReferencePoint()
    
    if (!meshName) {
      if (canvasRef.current) {
        canvasRef.current.clearCanvas()
      }
      return
    }

    meshBoundary.loadBoundary(meshName)
  }, [meshBoundary, meshData, referencePoint, trainingStatus])

  // Handle mesh data loading
  const handleLoadMeshData = useCallback(() => {
    if (!selectedMesh) return
    
    meshData.loadMeshData(selectedMesh).then(() => {
      if (canvasRef.current && meshData.meshData) {
        canvasRef.current.renderScene(
          meshData.meshData, 
          meshBoundary.boundaryData, 
          referencePoint.refPointInfo
        )
      }
    })
  }, [selectedMesh, meshData, meshBoundary.boundaryData, referencePoint.refPointInfo])

  // Handle reference point finding
  const handleFindReferencePoint = useCallback(() => {
    if (!selectedMesh) return
    
    referencePoint.findReferencePoint(selectedMesh).then(() => {
      if (canvasRef.current) {
        canvasRef.current.renderScene(
          meshData.meshData, 
          meshBoundary.boundaryData, 
          referencePoint.refPointInfo
        )
      }
    })
  }, [selectedMesh, referencePoint, meshData.meshData, meshBoundary.boundaryData])

  // Clear canvas and reset state
  const handleClearCanvas = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas()
    }
    meshData.clearMeshData()
    meshBoundary.clearBoundary()
    referencePoint.clearReferencePoint()
    setClickCoordinates(null)
  }, [meshData, meshBoundary, referencePoint])

  // Auto-update canvas when boundary data changes
  useEffect(() => {
    if (canvasRef.current && meshBoundary.boundaryData && selectedMesh) {
      canvasRef.current.renderBoundaryPreview(meshBoundary.boundaryData, selectedMesh)
    }
  }, [meshBoundary.boundaryData, selectedMesh])

  const StatusIndicator = ({ status }) => {
    const getStatusConfig = () => {
      switch (status) {
        case 'running':
          return { color: 'text-green-500', bg: 'bg-green-500/10', pulse: true, icon: '●' }
        case 'stopping':
          return { color: 'text-yellow-500', bg: 'bg-yellow-500/10', pulse: true, icon: '●' }
        case 'error':
          return { color: 'text-red-500', bg: 'bg-red-500/10', pulse: false, icon: '●' }
        default:
          return { color: 'text-gray-500', bg: 'bg-gray-500/10', pulse: false, icon: '●' }
      }
    }

    const config = getStatusConfig()

    return (
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg}`}>
        <span className={`${config.color} ${config.pulse ? 'animate-pulse' : ''}`}>
          {config.icon}
        </span>
        <span className={`text-xs font-medium ${config.color}`}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-bg-primary">
      <div className="p-6 pb-0">
        <PageHeader
          title="Training"
          subtitle="Start or monitor reinforcement learning training sessions for mesh generation."
          icon="🚂"
          backLink={{ href: '/', label: 'Back to Dashboard' }}
          size="sm"
        />
      </div>

      {/* Status Header */}
      <div className="px-6 mb-6">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-text-secondary">Status:</span>
                <StatusIndicator status={trainingStatus.trainingStatus.status} />
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-text-secondary">Model:</span>
                <span className="text-sm text-text-primary">
                  {trainingStatus.trainingConfig.model || 'Not selected'}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-text-secondary">Mesh:</span>
                <span className="text-sm text-text-primary">
                  {selectedMesh || 'Not selected'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={() => trainingStatus.startTraining({ ...trainingStatus.trainingConfig, mesh: selectedMesh })}
                disabled={!selectedMesh || isLoading || isTraining}
                size="sm"
                variant="primary"
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                {isTraining ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Training...
                  </>
                ) : (
                  'Start Training'
                )}
              </Button>
              <Button
                onClick={() => trainingStatus.stopTraining()}
                disabled={!isTraining || isLoading}
                size="sm"
                variant="danger"
              >
                Stop
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Main Content - Split Layout */}
      <div className="flex-1 px-6 pb-6 min-h-0">
        <div className="flex gap-6 h-full">
          {/* Left Controls Panel */}
          <div className="w-80 flex-shrink-0 space-y-6 overflow-y-auto">
            {/* Mesh Selection */}
            <Card title="Mesh Selection" className="p-4">
              <div className="space-y-4">
                <Select
                  value={selectedMesh}
                  onChange={(e) => handleMeshChange(e.target.value)}
                  disabled={isLoading || isTraining}
                  className="w-full"
                >
                  <option value="">Select a mesh...</option>
                  <option value="simple_square">Simple Square</option>
                  <option value="complex_polygon">Complex Polygon</option>
                  <option value="curved_boundary">Curved Boundary</option>
                </Select>
                
                <div className="space-y-2">
                  <Button
                    onClick={handleLoadMeshData}
                    disabled={!selectedMesh || isLoading || isTraining}
                    className="w-full"
                    size="sm"
                  >
                    {meshData.isLoading ? 'Loading...' : 'Load Full Mesh Data'}
                  </Button>
                  <Button
                    onClick={handleFindReferencePoint}
                    disabled={!selectedMesh || isLoading || isTraining}
                    variant="secondary"
                    className="w-full"
                    size="sm"
                  >
                    {referencePoint.isLoading ? 'Finding...' : 'Find Reference Point'}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Training Parameters */}
            <Card title="Training Parameters" className="p-4">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-1">Model</label>
                  <Select className="w-full" disabled={isTraining}>
                    <option value="ppo">PPO</option>
                    <option value="sac">SAC</option>
                    <option value="ddpg">DDPG</option>
                  </Select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-1">Episodes</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 border border-border-custom rounded-lg bg-bg-secondary text-text-primary"
                    defaultValue={1000}
                    disabled={isTraining}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    className="w-full px-3 py-2 border border-border-custom rounded-lg bg-bg-secondary text-text-primary"
                    defaultValue={0.001}
                    disabled={isTraining}
                  />
                </div>
              </div>
            </Card>

            {/* Status Info */}
            <Card title="Status Information" className="p-4">
              <div className="space-y-3 text-sm">
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
                  <span className="text-text-secondary">Reference Point:</span>
                  <span className={referencePoint.refPointInfo ? 'text-green-500' : 'text-gray-500'}>
                    {referencePoint.refPointInfo ? 'Found' : 'None'}
                  </span>
                </div>
                {clickCoordinates && (
                  <div className="mt-3 p-2 bg-bg-secondary rounded">
                    <div className="text-text-secondary text-xs mb-1">Last Click:</div>
                    <div className="text-text-primary text-xs">
                      ({clickCoordinates.world[0].toFixed(3)}, {clickCoordinates.world[1].toFixed(3)})
                    </div>
                  </div>
                )}
              </div>
            </Card>

            {/* Canvas Controls */}
            <Card title="Canvas Controls" className="p-4">
              <div className="space-y-2">
                <Button
                  onClick={handleClearCanvas}
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  Clear Canvas
                </Button>
              </div>
            </Card>
          </div>

          {/* Right MeshCanvas Panel */}
          <div className="flex-1 min-w-0">
            <Card className="h-full p-0 overflow-hidden">
              <div className="p-4 border-b border-border-custom">
                <h3 className="text-lg font-semibold text-text-primary">
                  Mesh Visualization
                </h3>
              </div>
              
              <div className="relative h-full min-h-[600px] bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
                {isLoading && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
                    <div className="flex flex-col items-center">
                      <Skeleton className="w-16 h-16 rounded mb-4" />
                      <div className="text-white text-sm">Loading mesh data...</div>
                    </div>
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
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Train
