import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { ApiProvider, useApi, usePolling } from '../ApiProvider'
import { server } from '../../test/mocks/server'
import { fixtures } from '../../test/mocks/handlers'

describe('ApiProvider', () => {
  describe('useApi hook', () => {
    it('provides api client when used within provider', () => {
      const { result } = renderHook(() => useApi(), {
        wrapper: ApiProvider
      })

      expect(result.current).toBeDefined()
      expect(typeof result.current.getTrainingStatus).toBe('function')
    })

    it('throws error when used outside provider', () => {
      // Suppress console error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      expect(() => {
        renderHook(() => useApi())
      }).toThrow('useApi must be used within an ApiProvider')

      consoleSpy.mockRestore()
    })

    it('makes API calls successfully', async () => {
      const { result } = renderHook(() => useApi(), {
        wrapper: ApiProvider
      })

      const response = await result.current.getTrainingStatus()
      
      expect(response).toEqual(fixtures.trainingStatus)
    })

    it('handles API errors with retry and error handling', async () => {
      // Mock error response
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          return HttpResponse.json(
            { success: false, error: 'Server error' },
            { status: 500 }
          )
        })
      )

      const { result } = renderHook(() => useApi(), {
        wrapper: ApiProvider
      })

      await expect(result.current.getTrainingStatus()).rejects.toThrow()
    })
  })

  describe('usePolling hook', () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('polls API endpoint at specified interval', async () => {
      let callCount = 0
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          callCount++
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      const { result } = renderHook(() => 
        usePolling('getTrainingStatus', 1000, { enabled: true }), {
          wrapper: ApiProvider
        }
      )

      // Initial call should be made
      await waitFor(() => {
        expect(result.current.data).toBeDefined()
      })

      expect(callCount).toBe(1)
      expect(result.current.isPolling).toBe(true)

      // Advance timer to trigger polling
      vi.advanceTimersByTime(1000)

      await waitFor(() => {
        expect(callCount).toBe(2)
      })
    })

    it('does not poll when disabled', async () => {
      let callCount = 0
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          callCount++
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      renderHook(() => 
        usePolling('getTrainingStatus', 1000, { enabled: false }), {
          wrapper: ApiProvider
        }
      )

      // Wait and advance timer
      vi.advanceTimersByTime(2000)

      // Should not have made any calls
      expect(callCount).toBe(0)
    })

    it('handles polling errors', async () => {
      const onError = vi.fn()
      
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          return HttpResponse.error()
        })
      )

      const { result } = renderHook(() => 
        usePolling('getTrainingStatus', 1000, { 
          enabled: true,
          onError 
        }), {
          wrapper: ApiProvider
        }
      )

      await waitFor(() => {
        expect(result.current.error).toBeTruthy()
      })

      expect(onError).toHaveBeenCalled()
    })

    it('calls success callback on successful polling', async () => {
      const onSuccess = vi.fn()
      
      renderHook(() => 
        usePolling('getTrainingStatus', 1000, { 
          enabled: true,
          onSuccess 
        }), {
          wrapper: ApiProvider
        }
      )

      await waitFor(() => {
        expect(onSuccess).toHaveBeenCalledWith(fixtures.trainingStatus)
      })
    })

    it('can be manually refreshed', async () => {
      let callCount = 0
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          callCount++
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      const { result } = renderHook(() => 
        usePolling('getTrainingStatus', 10000, { enabled: true }), {
          wrapper: ApiProvider
        }
      )

      // Wait for initial call
      await waitFor(() => {
        expect(callCount).toBe(1)
      })

      // Manual refresh
      result.current.refresh()

      await waitFor(() => {
        expect(callCount).toBe(2)
      })
    })

    it('can be started and stopped manually', async () => {
      let callCount = 0
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          callCount++
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      const { result } = renderHook(() => 
        usePolling('getTrainingStatus', 1000, { enabled: false }), {
          wrapper: ApiProvider
        }
      )

      // Should not be polling initially
      expect(result.current.isPolling).toBe(false)
      expect(callCount).toBe(0)

      // Start polling manually
      result.current.startPolling()

      await waitFor(() => {
        expect(result.current.isPolling).toBe(true)
        expect(callCount).toBe(1)
      })

      // Stop polling
      result.current.stopPolling()

      expect(result.current.isPolling).toBe(false)

      // Advance timer - should not poll anymore
      vi.advanceTimersByTime(2000)
      expect(callCount).toBe(1)
    })

    it('handles method arguments correctly', async () => {
      let receivedArgs = []
      
      server.use(
        http.post('http://127.0.0.1:5000/training/reference-point', async ({ request }) => {
          const body = await request.json()
          receivedArgs.push(body)
          return HttpResponse.json(fixtures.referencePoint)
        })
      )

      const { result } = renderHook(() => 
        usePolling('getTrainingReferencePoint', 1000, { 
          enabled: true,
          methodArgs: [{ mesh: 'test.obj' }]
        }), {
          wrapper: ApiProvider
        }
      )

      await waitFor(() => {
        expect(result.current.data).toBeDefined()
      })

      expect(receivedArgs[0]).toEqual({ mesh: 'test.obj' })
    })

    it('restarts polling when dependencies change', async () => {
      let callCount = 0
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          callCount++
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      const { result, rerender } = renderHook(
        ({ dependency }) => usePolling('getTrainingStatus', 1000, { 
          enabled: true,
          dependencies: [dependency]
        }), 
        {
          wrapper: ApiProvider,
          initialProps: { dependency: 'value1' }
        }
      )

      // Wait for initial call
      await waitFor(() => {
        expect(callCount).toBe(1)
      })

      // Change dependency
      rerender({ dependency: 'value2' })

      // Should restart and make another call
      await waitFor(() => {
        expect(callCount).toBe(2)
      })
    })
  })

  describe('API methods', () => {
    it('provides access to all API client methods', () => {
      const { result } = renderHook(() => useApi(), {
        wrapper: ApiProvider
      })

      const expectedMethods = [
        'getTrainingStatus',
        'startTraining',
        'stopTraining',
        'getMeshBoundary',
        'getMeshData',
        'getTrainingReferencePoint'
      ]

      expectedMethods.forEach(method => {
        expect(typeof result.current[method]).toBe('function')
      })
    })

    it('enhances API methods with error handling and retry', async () => {
      let attemptCount = 0
      
      server.use(
        http.get('http://127.0.0.1:5000/training/status', () => {
          attemptCount++
          if (attemptCount < 2) {
            return HttpResponse.error()
          }
          return HttpResponse.json(fixtures.trainingStatus)
        })
      )

      const { result } = renderHook(() => useApi(), {
        wrapper: ApiProvider
      })

      const response = await result.current.getTrainingStatus()
      
      // Should have retried and eventually succeeded
      expect(attemptCount).toBeGreaterThan(1)
      expect(response).toEqual(fixtures.trainingStatus)
    })
  })
})
