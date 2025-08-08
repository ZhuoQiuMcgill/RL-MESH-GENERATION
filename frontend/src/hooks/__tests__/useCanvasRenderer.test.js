import { renderHook, act } from '@testing-library/react';
import { useRef } from 'react';
import useCanvasRenderer from '../useCanvasRenderer';

// Mock CanvasRenderer
jest.mock('../../../tools/js/canvas-renderer.js', () => ({
  CanvasRenderer: jest.fn().mockImplementation(() => ({
    renderScene: jest.fn(),
    renderBoundaryPreview: jest.fn(),
    clearCanvas: jest.fn(),
    getCurrentTransform: jest.fn(() => ({ scale: 1, offsetX: 0, offsetY: 0 })),
    screenToWorld: jest.fn((x, y, transform) => [x, y]),
    worldToScreen: jest.fn((coords, transform) => coords),
    onResize: jest.fn(),
    setSizeOverrides: jest.fn(),
    getCurrentSizes: jest.fn(() => ({})),
    destroy: jest.fn(),
  }))
}));

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  disconnect: jest.fn(),
  unobserve: jest.fn(),
}));

describe('useCanvasRenderer Hook', () => {
  let mockCanvasElement;

  beforeEach(() => {
    // Mock canvas element
    mockCanvasElement = {
      getContext: jest.fn(() => ({})),
      parentElement: document.createElement('div'),
      getBoundingClientRect: jest.fn(() => ({
        left: 0,
        top: 0,
        width: 800,
        height: 600,
      })),
    };

    jest.clearAllMocks();
  });

  it('should initialize properly when canvasRef is available', () => {
    const { result } = renderHook(() => {
      const canvasRef = { current: mockCanvasElement };
      return useCanvasRenderer(canvasRef);
    });

    expect(result.current).toHaveProperty('drawScene');
    expect(result.current).toHaveProperty('renderBoundaryPreview');
    expect(result.current).toHaveProperty('clearCanvas');
    expect(result.current).toHaveProperty('screenToWorld');
    expect(result.current).toHaveProperty('worldToScreen');
    expect(result.current).toHaveProperty('getCurrentTransform');
  });

  it('should handle null canvasRef gracefully', () => {
    const { result } = renderHook(() => {
      const canvasRef = { current: null };
      return useCanvasRenderer(canvasRef);
    });

    // Should not throw and should return methods that handle null state
    expect(() => result.current.clearCanvas()).not.toThrow();
    expect(() => result.current.drawScene({}, [])).not.toThrow();
  });

  it('should provide working coordinate transformation methods', () => {
    const { result } = renderHook(() => {
      const canvasRef = { current: mockCanvasElement };
      return useCanvasRenderer(canvasRef);
    });

    act(() => {
      const worldCoords = result.current.screenToWorld(100, 100);
      expect(worldCoords).toEqual([100, 100]);

      const screenCoords = result.current.worldToScreen([50, 50]);
      expect(screenCoords).toEqual([50, 50]);

      const transform = result.current.getCurrentTransform();
      expect(transform).toEqual({ scale: 1, offsetX: 0, offsetY: 0 });
    });
  });

  it('should handle errors gracefully', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    const { result } = renderHook(() => {
      const canvasRef = { current: null };
      return useCanvasRenderer(canvasRef);
    });

    // These should log warnings but not throw
    act(() => {
      result.current.clearCanvas();
      result.current.drawScene({}, []);
      result.current.renderBoundaryPreview([]);
    });

    consoleSpy.mockRestore();
  });

  it('should return stable references', () => {
    const { result, rerender } = renderHook(() => {
      const canvasRef = { current: mockCanvasElement };
      return useCanvasRenderer(canvasRef);
    });

    const firstResult = result.current;
    
    rerender();
    
    const secondResult = result.current;

    // Methods should be stable (same references)
    expect(firstResult.drawScene).toBe(secondResult.drawScene);
    expect(firstResult.clearCanvas).toBe(secondResult.clearCanvas);
    expect(firstResult.screenToWorld).toBe(secondResult.screenToWorld);
  });

  it('should setup ResizeObserver when available', () => {
    const { result } = renderHook(() => {
      const canvasRef = { current: mockCanvasElement };
      return useCanvasRenderer(canvasRef);
    });

    expect(global.ResizeObserver).toHaveBeenCalled();
  });

  it('should cleanup resources on unmount', () => {
    const { result, unmount } = renderHook(() => {
      const canvasRef = { current: mockCanvasElement };
      return useCanvasRenderer(canvasRef);
    });

    const { CanvasRenderer } = require('../../../tools/js/canvas-renderer.js');
    const mockInstance = CanvasRenderer.mock.results[0].value;

    unmount();

    expect(mockInstance.destroy).toHaveBeenCalled();
  });
});
