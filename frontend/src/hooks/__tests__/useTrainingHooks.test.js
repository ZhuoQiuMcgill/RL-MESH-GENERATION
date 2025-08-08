import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { 
  useMeshBoundary, 
  useMeshData, 
  useReferencePoint, 
  useTrainingStatus 
} from '../useTrainingHooks'
import { createHookWrapper } from '../../test/utils/test-utils'
import { server } from '../../test/mocks/server'
import { fixtures } from '../../test/mocks/handlers'

const HookWrapper = createHookWrapper(['api'])

describe('useTrainingHooks', () => {
  describe('useMeshBoundary', () => {
    it('should load mesh boundary data successfully', async () => {
      const { result } = renderHook(() => useMeshBoundary('simple_square.obj'), {
        wrapper: HookWrapper
      })

      expect(result.current.isLoading).toBe(false)
      expect(result.current.boundaryData).toBeNull()
      expect(result.current.error).toBeNull()

      // Load boundary data
      await waitFor(() => {
        result.current.loadBoundary()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.boundaryData).toEqual(fixtures.meshBoundary.boundary_vertices)
      expect(result.current.error).toBeNull()
    })

    it('should handle mesh boundary load error', async () => {
      const { result } = renderHook(() => useMeshBoundary('invalid_mesh.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.loadBoundary()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.boundaryData).toBeNull()
      expect(result.current.error).toBeTruthy()
    })

    it('should clear boundary data', async () => {
      const { result } = renderHook(() => useMeshBoundary('simple_square.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.loadBoundary()
      })

      await waitFor(() => {
        expect(result.current.boundaryData).toBeTruthy()
      })

      result.current.clearBoundary()

      expect(result.current.boundaryData).toBeNull()
      expect(result.current.error).toBeNull()
    })

    it('should handle empty mesh name', async () => {
      const { result } = renderHook(() => useMeshBoundary(''), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.loadBoundary()
      })

      expect(result.current.boundaryData).toBeNull()
      expect(result.current.isLoading).toBe(false)
    })
  })

  describe('useMeshData', () => {
    it('should load mesh data successfully', async () => {
      const { result } = renderHook(() => useMeshData('simple_square.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.loadMeshData()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.meshData).toEqual(fixtures.meshData.mesh_data)
      expect(result.current.error).toBeNull()
    })

    it('should handle mesh data load error', async () => {
      const { result } = renderHook(() => useMeshData('invalid_mesh.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.loadMeshData()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.meshData).toBeNull()
      expect(result.current.error).toBeTruthy()
    })
  })

  describe('useReferencePoint', () => {
    it('should find reference point successfully', async () => {
      const { result } = renderHook(() => useReferencePoint('simple_square.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.findReferencePoint()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.refPointInfo).toEqual(fixtures.referencePoint.reference_point)
      expect(result.current.error).toBeNull()
    })

    it('should handle reference point error', async () => {
      // Mock error response
      server.use(
        http.post('http://127.0.0.1:5000/training/reference-point', () => {
          return HttpResponse.json(
            { success: false, error: 'Reference point not found' },
            { status: 404 }
          )
        })
      )

      const { result } = renderHook(() => useReferencePoint('invalid_mesh.obj'), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.findReferencePoint()
      })

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })

      expect(result.current.refPointInfo).toBeNull()
      expect(result.current.error).toBeTruthy()
    })
  })

  describe('useTrainingStatus', () => {
    it('should initialize with default status', () => {
      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      expect(result.current.trainingStatus.is_training).toBe(false)
      expect(result.current.trainingStatus.status).toBe('idle')
      expect(result.current.trainingStatus.episode).toBe(0)
    })

    it('should get training status', async () => {
      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      await waitFor(() => {
        result.current.getStatus()
      })

      await waitFor(() => {
        expect(result.current.trainingStatus).toEqual(fixtures.trainingStatus.status)
      })
    })

    it('should start training successfully', async () => {
      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      const config = {
        mesh: 'simple_square.obj',
        algorithm: 'PPO',
        episodes: 1000
      }

      await waitFor(() => {
        result.current.startTraining(config)
      })

      await waitFor(() => {
        expect(result.current.trainingStatus.is_training).toBe(true)
        expect(result.current.trainingStatus.status).toBe('training')
      })
    })

    it('should stop training successfully', async () => {
      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      // First start training
      await waitFor(() => {
        result.current.startTraining({
          mesh: 'simple_square.obj',
          algorithm: 'PPO'
        })
      })

      // Then stop training
      await waitFor(() => {
        result.current.stopTraining()
      })

      await waitFor(() => {
        expect(result.current.trainingStatus.is_training).toBe(false)
        expect(result.current.trainingStatus.status).toBe('stopped')
      })
    })

    it('should update training configuration', () => {
      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      const updates = {
        mesh: 'complex_geometry.obj',
        episodes: 2000,
        learning_rate: 0.01
      }

      result.current.updateConfig(updates)

      expect(result.current.trainingConfig.mesh).toBe('complex_geometry.obj')
      expect(result.current.trainingConfig.episodes).toBe(2000)
      expect(result.current.trainingConfig.learning_rate).toBe(0.01)
      expect(result.current.trainingConfig.algorithm).toBe('PPO') // Should keep existing values
    })

    it('should handle training start error', async () => {
      // Mock error response
      server.use(
        http.post('http://127.0.0.1:5000/training/start', () => {
          return HttpResponse.json(
            { success: false, error: 'Missing required fields' },
            { status: 400 }
          )
        })
      )

      const { result } = renderHook(() => useTrainingStatus(), {
        wrapper: HookWrapper
      })

      await expect(
        result.current.startTraining({ mesh: '' }) // Invalid config
      ).rejects.toThrow()
    })

    it('should handle status callback on change', async () => {
      const onStatusChange = vi.fn()
      const { result } = renderHook(() => 
        useTrainingStatus({ onStatusChange }), {
          wrapper: HookWrapper
        }
      )

      await waitFor(() => {
        result.current.getStatus()
      })

      // Simulate status change by getting status again with different data
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          return HttpResponse.json({
            ...fixtures.trainingStatus,
            status: { ...fixtures.trainingStatus.status, status: 'running' }
          })
        })
      )

      await waitFor(() => {
        result.current.getStatus()
      })

      await waitFor(() => {
        expect(onStatusChange).toHaveBeenCalled()
      })
    })
  })
})
