import { useRef, useEffect, useCallback, useMemo } from 'react';
import { CanvasRenderer } from '../utils/CanvasRenderer.js';

/**
 * useCanvasRenderer Hook - 2D Canvas Renderer
 * Wraps the CanvasRenderer class as a React Hook, providing responsive 2D mesh rendering functionality
 * 
 * @param {React.RefObject} canvasRef - ref to the canvas element
 * @returns {Object} object containing rendering methods and state
 */
export const useCanvasRenderer = (canvasRef) => {
  const rendererRef = useRef(null);
  const cleanupFnRef = useRef(null);
  const resizeObserverRef = useRef(null);

  // Initialize renderer
  const initRenderer = useCallback(() => {
    if (!canvasRef?.current || rendererRef.current) {
      return;
    }

    try {
      // Create CanvasRenderer instance
      const renderer = new CanvasRenderer(canvasRef.current);
      rendererRef.current = renderer;

      // Set up ResizeObserver to monitor container size changes
      if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver((entries) => {
          // Delay processing to ensure DOM updates are complete
          setTimeout(() => {
            if (rendererRef.current) {
              rendererRef.current.onResize();
            }
          }, 50);
        });

        // Observe canvas parent container
        const container = canvasRef.current.parentElement;
        if (container) {
          resizeObserver.observe(container);
          resizeObserverRef.current = resizeObserver;
        }
      }

    } catch (error) {
      console.error('Failed to initialize CanvasRenderer:', error);
    }
  }, [canvasRef]);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect();
      resizeObserverRef.current = null;
    }

    if (rendererRef.current) {
      rendererRef.current.destroy();
      rendererRef.current = null;
    }

    if (cleanupFnRef.current) {
      cleanupFnRef.current();
      cleanupFnRef.current = null;
    }
  }, []);

  // Initialize renderer when canvasRef is available
  useEffect(() => {
    if (canvasRef?.current) {
      initRenderer();
    }
    
    return cleanup;
  }, [canvasRef, initRenderer, cleanup]);

  // Handle resize event cleanup
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Core rendering methods
  const drawScene = useCallback((meshData, boundaryVertices, refPointInfo = null) => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return;
    }

    try {
      rendererRef.current.renderScene(meshData, boundaryVertices, refPointInfo);
    } catch (error) {
      console.error('Failed to render scene:', error);
    }
  }, []);

  const renderBoundaryPreview = useCallback((boundaryVertices, meshName = '') => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return;
    }

    try {
      rendererRef.current.renderBoundaryPreview(boundaryVertices, meshName);
    } catch (error) {
      console.error('Failed to render boundary preview:', error);
    }
  }, []);

  const clearCanvas = useCallback(() => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return;
    }

    try {
      rendererRef.current.clearCanvas();
    } catch (error) {
      console.error('Failed to clear canvas:', error);
    }
  }, []);

  // Coordinate conversion methods
  const screenToWorld = useCallback((screenX, screenY) => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return [0, 0];
    }

    try {
      const transform = rendererRef.current.getCurrentTransform();
      if (!transform) {
        return [0, 0];
      }
      return rendererRef.current.screenToWorld(screenX, screenY, transform);
    } catch (error) {
      console.error('Failed to convert screen to world coordinates:', error);
      return [0, 0];
    }
  }, []);

  const worldToScreen = useCallback((worldCoords) => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return [0, 0];
    }

    try {
      const transform = rendererRef.current.getCurrentTransform();
      if (!transform) {
        return [0, 0];
      }
      return rendererRef.current.worldToScreen(worldCoords, transform);
    } catch (error) {
      console.error('Failed to convert world to screen coordinates:', error);
      return [0, 0];
    }
  }, []);

  const getCurrentTransform = useCallback(() => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return null;
    }

    try {
      return rendererRef.current.getCurrentTransform();
    } catch (error) {
      console.error('Failed to get current transform:', error);
      return null;
    }
  }, []);

  // Utility methods
  const onResize = useCallback(() => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return;
    }

    try {
      rendererRef.current.onResize();
    } catch (error) {
      console.error('Failed to handle resize:', error);
    }
  }, []);

  const getRenderer = useCallback(() => {
    return rendererRef.current;
  }, []);

  const getCanvas = useCallback(() => {
    return canvasRef?.current;
  }, [canvasRef]);

  // Set custom size (for testing and debugging)
  const setSizeOverrides = useCallback((sizeOverrides) => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return;
    }

    try {
      rendererRef.current.setSizeOverrides(sizeOverrides);
    } catch (error) {
      console.error('Failed to set size overrides:', error);
    }
  }, []);

  // Get current adaptive size (for debugging)
  const getCurrentSizes = useCallback(() => {
    if (!rendererRef.current) {
      console.warn('CanvasRenderer not initialized');
      return {};
    }

    try {
      return rendererRef.current.getCurrentSizes();
    } catch (error) {
      console.error('Failed to get current sizes:', error);
      return {};
    }
  }, []);

  // Return memoized object to avoid unnecessary re-renders
  return useMemo(() => ({
    // Core rendering methods
    drawScene,
    renderBoundaryPreview,
    clearCanvas,
    
    // Coordinate conversion
    screenToWorld,
    worldToScreen,
    getCurrentTransform,
    
    // Utility methods
    onResize,
    getRenderer,
    getCanvas,
    
    // Debugging and testing methods
    setSizeOverrides,
    getCurrentSizes,
    
    // Cleanup method (usually not needed to call manually)
    cleanup
  }), [
    drawScene,
    renderBoundaryPreview,
    clearCanvas,
    screenToWorld,
    worldToScreen,
    getCurrentTransform,
    onResize,
    getRenderer,
    getCanvas,
    setSizeOverrides,
    getCurrentSizes,
    cleanup
  ]);
};

export default useCanvasRenderer;
