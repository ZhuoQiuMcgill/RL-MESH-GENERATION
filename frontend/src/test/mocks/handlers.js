import { http, HttpResponse } from 'msw'

// Golden response fixtures
export const fixtures = {
  trainingStatus: {
    success: true,
    status: {
      is_training: false,
      status: 'idle',
      episode: 0,
      total_episodes: 0,
      current_reward: 0,
      best_reward: 0,
      elapsed_time: 0,
      last_updated: new Date().toISOString()
    }
  },
  
  trainingStatusRunning: {
    success: true,
    status: {
      is_training: true,
      status: 'running',
      episode: 45,
      total_episodes: 1000,
      current_reward: 125.5,
      best_reward: 200.3,
      elapsed_time: 3600,
      last_updated: new Date().toISOString()
    }
  },

  meshList: {
    success: true,
    meshes: [
      'simple_square.obj',
      'complex_geometry.obj',
      'test_mesh.obj'
    ]
  },

  meshBoundary: {
    success: true,
    boundary_vertices: [
      [0, 0], [1, 0], [1, 1], [0, 1]
    ]
  },

  meshData: {
    success: true,
    mesh_data: {
      vertices: [[0, 0], [1, 0], [1, 1], [0, 1]],
      triangles: [[0, 1, 2], [0, 2, 3]],
      quality_metrics: {
        avg_quality: 0.85,
        min_quality: 0.72,
        max_quality: 0.95
      }
    }
  },

  referencePoint: {
    success: true,
    reference_point: {
      x: 0.5,
      y: 0.5,
      description: 'Center point of mesh'
    }
  },

  startTraining: {
    success: true,
    message: 'Training started successfully',
    session_id: 'training_session_123'
  },

  stopTraining: {
    success: true,
    message: 'Training stopped successfully'
  },

  apiError: {
    success: false,
    error: 'Internal server error',
    details: 'Connection timeout'
  },

  networkError: {
    success: false,
    error: 'Network error',
    details: 'Failed to fetch'
  }
}

// Request handlers
export const handlers = [
  // Training status endpoints
  http.get('http://127.0.0.1:5000/training/status', () => {
    return HttpResponse.json(fixtures.trainingStatus)
  }),

  http.post('http://127.0.0.1:5000/training/start', async ({ request }) => {
    const body = await request.json()
    
    // Validate required fields
    if (!body.mesh || !body.algorithm) {
      return HttpResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      )
    }
    
    return HttpResponse.json(fixtures.startTraining)
  }),

  http.post('http://127.0.0.1:5000/training/stop', () => {
    return HttpResponse.json(fixtures.stopTraining)
  }),

  // Mesh endpoints
  http.get('http://127.0.0.1:5000/meshes', () => {
    return HttpResponse.json(fixtures.meshList)
  }),

  http.get('http://127.0.0.1:5000/mesh/:meshName/boundary', ({ params }) => {
    if (params.meshName === 'invalid_mesh.obj') {
      return HttpResponse.json(
        { success: false, error: 'Mesh not found' },
        { status: 404 }
      )
    }
    return HttpResponse.json(fixtures.meshBoundary)
  }),

  http.get('http://127.0.0.1:5000/mesh/:meshName/data', ({ params }) => {
    if (params.meshName === 'invalid_mesh.obj') {
      return HttpResponse.json(
        { success: false, error: 'Mesh not found' },
        { status: 404 }
      )
    }
    return HttpResponse.json(fixtures.meshData)
  }),

  http.post('http://127.0.0.1:5000/training/reference-point', async ({ request }) => {
    const body = await request.json()
    
    if (!body.mesh) {
      return HttpResponse.json(
        { success: false, error: 'Mesh name required' },
        { status: 400 }
      )
    }
    
    return HttpResponse.json(fixtures.referencePoint)
  }),

  // Error simulation endpoints
  http.get('http://127.0.0.1:5000/test/server-error', () => {
    return HttpResponse.json(fixtures.apiError, { status: 500 })
  }),

  http.get('http://127.0.0.1:5000/test/network-error', () => {
    return HttpResponse.error()
  }),

  http.get('http://127.0.0.1:5000/test/timeout', async () => {
    // Simulate timeout
    await new Promise(resolve => setTimeout(resolve, 10000))
    return HttpResponse.json(fixtures.trainingStatus)
  }),

  // Health check
  http.get('http://127.0.0.1:5000/health', () => {
    return HttpResponse.json({ status: 'healthy', timestamp: new Date().toISOString() })
  })
]

// Helper functions for dynamic responses
export const createTrainingStatusHandler = (overrides = {}) => {
  return http.get('http://127.0.0.1:5000/training/status', () => {
    return HttpResponse.json({
      ...fixtures.trainingStatus,
      status: { ...fixtures.trainingStatus.status, ...overrides }
    })
  })
}

export const createErrorHandler = (endpoint, statusCode = 500, errorMessage = 'Server error') => {
  return http.get(endpoint, () => {
    return HttpResponse.json(
      { success: false, error: errorMessage },
      { status: statusCode }
    )
  })
}
