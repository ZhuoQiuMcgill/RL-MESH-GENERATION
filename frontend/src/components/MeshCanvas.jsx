import React, { useRef, useEffect, forwardRef, useImperativeHandle, useState, useCallback, useMemo, memo } from 'react';
import { CanvasRenderer } from '../utils/CanvasRenderer.js';

/**
 * MeshCanvas - Enhanced React wrapper component for CanvasRenderer
 * 
 * Features:
 * - High-DPI scaling with devicePixelRatio handling
 * - Proper ResizeObserver integration
 * - Optional props: backgroundColor, showGrid, zoom/pan controls, etc.
 * - Overlay layer for UI annotations
 * - Comprehensive interaction pattern support
 * 
 * @param {Object} props - Component props
 * @param {string} props.className - Additional CSS classes
 * @param {Object} props.style - Inline styles
 * @param {Function} props.onCanvasClick - Click handler callback
 * @param {string} props.backgroundColor - Canvas background color (default: transparent)
 * @param {boolean} props.showGrid - Show/hide grid overlay (default: true)
 * @param {boolean} props.enableZoom - Enable zoom functionality (default: true)
 * @param {boolean} props.enablePan - Enable pan functionality (default: true)
 * @param {number} props.minZoom - Minimum zoom level (default: 0.1)
 * @param {number} props.maxZoom - Maximum zoom level (default: 5.0)
 * @param {number} props.devicePixelRatio - Override device pixel ratio (optional)
 * @param {boolean} props.showOverlay - Show annotation overlay layer (default: false)
 * @param {Array} props.annotations - Array of annotation objects for overlay
 * @param {Function} props.onZoomChange - Callback for zoom level changes
 * @param {Function} props.onPanChange - Callback for pan position changes
 */
const MeshCanvas = forwardRef((props, ref) => {
  const {
    className = '',
    style = {},
    onCanvasClick = null,
    backgroundColor = 'transparent',
    showGrid = true,
    enableZoom = true,
    enablePan = true,
    minZoom = 0.1,
    maxZoom = 5.0,
    devicePixelRatio = null,
    showOverlay = false,
    annotations = [],
    onZoomChange = null,
    onPanChange = null,
    ...canvasProps
  } = props;

  // Refs
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const resizeObserverRef = useRef(null);
  const resizeCleanupRef = useRef(null);
  
  // State for enhanced features
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isInteracting, setIsInteracting] = useState(false);

  // Memoize renderer configuration to prevent unnecessary re-renders
  const _rendererConfig = useMemo(() => ({
    backgroundColor,
    showGrid,
    enableZoom,
    enablePan,
    devicePixelRatio,
    minZoom,
    maxZoom
  }), [backgroundColor, showGrid, enableZoom, enablePan, devicePixelRatio, minZoom, maxZoom]);

  // Memoize container styles to prevent unnecessary re-renders
  const containerStyles = useMemo(() => ({
    position: 'relative',
    display: 'block',
    width: '100%',
    height: '100%',
    backgroundColor,
    overflow: 'hidden',
    ...style
  }), [backgroundColor, style]);

  // Memoize canvas styles to prevent unnecessary re-renders
  const canvasStyles = useMemo(() => ({
    display: 'block',
    width: '100%',
    height: '100%',
    cursor: isInteracting ? 'grabbing' : 
           (enablePan ? 'grab' : 
           (onCanvasClick ? 'crosshair' : 'default')),
    touchAction: enablePan || enableZoom ? 'none' : 'auto'
  }), [isInteracting, enablePan, enableZoom, onCanvasClick]);

  // Enhanced zoom/pan interaction handlers
  const handleZoomChange = useCallback((newZoom) => {
    const clampedZoom = Math.max(minZoom, Math.min(maxZoom, newZoom));
    setZoomLevel(clampedZoom);
    if (onZoomChange) {
      onZoomChange(clampedZoom);
    }
  }, [minZoom, maxZoom, onZoomChange]);

  const handlePanChange = useCallback((newPan) => {
    setPanOffset(newPan);
    if (onPanChange) {
      onPanChange(newPan);
    }
  }, [onPanChange]);

  // Initialize canvas renderer with enhanced options
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    try {
      // Create renderer instance with enhanced options
      const renderer = new CanvasRenderer(canvasRef.current, {
        backgroundColor,
        showGrid,
        enableZoom,
        enablePan,
        devicePixelRatio,
        minZoom,
        maxZoom
      });
      rendererRef.current = renderer;

      // Enhanced click handler with zoom/pan support
      if (onCanvasClick) {
        const handleCanvasClick = (event) => {
          if (!renderer || isInteracting) return;

          const transform = renderer.getCurrentTransform();
          if (!transform) {
            onCanvasClick(null, event);
            return;
          }

          // Get mouse position relative to canvas with high-DPI support
          const rect = event.target.getBoundingClientRect();
          const dpr = devicePixelRatio || window.devicePixelRatio || 1;
          const screenX = (event.clientX - rect.left) * dpr;
          const screenY = (event.clientY - rect.top) * dpr;

          // Convert to world coordinates with zoom/pan adjustments
          const worldCoords = renderer.screenToWorld(
            screenX / dpr, 
            screenY / dpr, 
            transform,
            { zoom: zoomLevel, pan: panOffset }
          );
          
          onCanvasClick(worldCoords, event);
        };

        canvasRef.current.addEventListener('click', handleCanvasClick);

        // Enhanced interaction handlers for zoom/pan
        if (enableZoom || enablePan) {
          const handleWheel = (event) => {
            if (!enableZoom) return;
            event.preventDefault();
            
            const delta = event.deltaY > 0 ? 0.9 : 1.1;
            handleZoomChange(zoomLevel * delta);
          };

          let isPanning = false;
          let lastPanPoint = { x: 0, y: 0 };

          const handleMouseDown = (event) => {
            if (!enablePan) return;
            if (event.button === 0) { // Left mouse button
              isPanning = true;
              setIsInteracting(true);
              lastPanPoint = { x: event.clientX, y: event.clientY };
              event.preventDefault();
            }
          };

          const handleMouseMove = (event) => {
            if (!enablePan || !isPanning) return;
            
            const deltaX = event.clientX - lastPanPoint.x;
            const deltaY = event.clientY - lastPanPoint.y;
            
            handlePanChange({
              x: panOffset.x + deltaX,
              y: panOffset.y + deltaY
            });
            
            lastPanPoint = { x: event.clientX, y: event.clientY };
          };

          const handleMouseUp = () => {
            if (isPanning) {
              isPanning = false;
              setIsInteracting(false);
            }
          };

          if (enableZoom) {
            canvasRef.current.addEventListener('wheel', handleWheel, { passive: false });
          }
          
          if (enablePan) {
            canvasRef.current.addEventListener('mousedown', handleMouseDown);
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
          }

          // Cleanup enhanced interaction handlers
          return () => {
            if (canvasRef.current) {
              canvasRef.current.removeEventListener('click', handleCanvasClick);
              if (enableZoom) {
                canvasRef.current.removeEventListener('wheel', handleWheel);
              }
              if (enablePan) {
                canvasRef.current.removeEventListener('mousedown', handleMouseDown);
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              }
            }
          };
        } else {
          return () => {
            if (canvasRef.current) {
              canvasRef.current.removeEventListener('click', handleCanvasClick);
            }
          };
        }
      }
    } catch (error) {
      console.error('Failed to initialize CanvasRenderer:', error);
    }

    // Cleanup function
    return () => {
      if (rendererRef.current) {
        rendererRef.current.destroy();
        rendererRef.current = null;
      }
      if (resizeCleanupRef.current) {
        resizeCleanupRef.current();
        resizeCleanupRef.current = null;
      }
    };
  }, [onCanvasClick, backgroundColor, showGrid, enableZoom, enablePan, devicePixelRatio, minZoom, maxZoom, zoomLevel, panOffset, isInteracting, handleZoomChange, handlePanChange]);

  // Enhanced ResizeObserver with proper high-DPI handling
  useEffect(() => {
    if (!rendererRef.current || !containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      if (!rendererRef.current) return;
      
      // Handle multiple resize entries properly
      for (const entry of entries) {
        // Use contentBoxSize for more accurate measurements
        const { inlineSize: width, blockSize: height } = entry.contentBoxSize?.[0] || 
          { inlineSize: entry.contentRect.width, blockSize: entry.contentRect.height };
        
        // Pass dimensions to renderer for high-DPI calculations
        rendererRef.current.onResize({
          width,
          height,
          devicePixelRatio: devicePixelRatio || window.devicePixelRatio || 1
        });
      }
    });

    // Observe the container element for more accurate resize detection
    resizeObserverRef.current = resizeObserver;
    resizeObserver.observe(containerRef.current);

    // Also observe device pixel ratio changes
    let lastDpr = window.devicePixelRatio || 1;
    const checkDprChange = () => {
      const currentDpr = window.devicePixelRatio || 1;
      if (Math.abs(currentDpr - lastDpr) > 0.001) {
        lastDpr = currentDpr;
        if (rendererRef.current) {
          rendererRef.current.onResize({ devicePixelRatio: currentDpr });
        }
      }
      requestAnimationFrame(checkDprChange);
    };
    
    const rafId = requestAnimationFrame(checkDprChange);

    return () => {
      resizeObserver.disconnect();
      resizeObserverRef.current = null;
      cancelAnimationFrame(rafId);
    };
  }, [devicePixelRatio]);

  // Update overlay annotations when they change
  useEffect(() => {
    if (showOverlay && overlayRef.current) {
      updateOverlay();
    }
  }, [annotations, showOverlay, zoomLevel, panOffset, updateOverlay]);

  // Update overlay content
  const updateOverlay = useCallback(() => {
    if (!overlayRef.current || !rendererRef.current) return;
    
    const overlay = overlayRef.current;
    const transform = rendererRef.current.getCurrentTransform();
    
    if (!transform) {
      overlay.innerHTML = '';
      return;
    }

    // Clear existing annotations
    overlay.innerHTML = '';

    // Render each annotation
    annotations.forEach((annotation, _index) => {
      if (!annotation.position) return;

      const screenPos = rendererRef.current.worldToScreen(
        annotation.position, 
        transform,
        { zoom: zoomLevel, pan: panOffset }
      );

      const element = document.createElement('div');
      element.className = `mesh-canvas-annotation ${annotation.type || 'default'}`;
      element.style.cssText = `
        position: absolute;
        left: ${screenPos[0]}px;
        top: ${screenPos[1]}px;
        transform: translate(-50%, -50%);
        pointer-events: ${annotation.interactive ? 'auto' : 'none'};
        z-index: ${annotation.zIndex || 10};
        ${annotation.style || ''}
      `;
      
      if (annotation.content) {
        if (typeof annotation.content === 'string') {
          element.innerHTML = annotation.content;
        } else {
          element.appendChild(annotation.content);
        }
      }

      if (annotation.onClick) {
        element.addEventListener('click', (e) => {
          e.stopPropagation();
          annotation.onClick(annotation, e);
        });
      }

      overlay.appendChild(element);
    });
  }, [annotations, zoomLevel, panOffset]);

  // Enhanced imperative API with new features
  useImperativeHandle(ref, () => ({
    // Core rendering methods
    clearCanvas: () => {
      if (rendererRef.current) {
        rendererRef.current.clearCanvas();
      }
    },

    renderBoundaryPreview: (boundaryVertices, meshName = '') => {
      if (rendererRef.current) {
        rendererRef.current.renderBoundaryPreview(boundaryVertices, meshName);
      }
    },

    renderScene: (meshData, boundaryVertices, refPointInfo = null) => {
      if (rendererRef.current) {
        rendererRef.current.renderScene(meshData, boundaryVertices, refPointInfo);
      }
    },

    // Enhanced coordinate transformation with zoom/pan support
    getCurrentTransform: () => {
      return rendererRef.current?.getCurrentTransform() || null;
    },

    screenToWorld: (screenX, screenY) => {
      const transform = rendererRef.current?.getCurrentTransform();
      if (!rendererRef.current || !transform) {
        return [0, 0];
      }
      return rendererRef.current.screenToWorld(
        screenX, 
        screenY, 
        transform,
        { zoom: zoomLevel, pan: panOffset }
      );
    },

    worldToScreen: (worldCoords) => {
      const transform = rendererRef.current?.getCurrentTransform();
      if (!rendererRef.current || !transform) {
        return [0, 0];
      }
      return rendererRef.current.worldToScreen(
        worldCoords, 
        transform,
        { zoom: zoomLevel, pan: panOffset }
      );
    },

    // Enhanced canvas control methods
    onResize: (dimensions) => {
      if (rendererRef.current) {
        rendererRef.current.onResize(dimensions);
      }
    },

    // Zoom and pan controls
    setZoom: (zoom) => {
      handleZoomChange(zoom);
    },

    getZoom: () => zoomLevel,

    setPan: (pan) => {
      handlePanChange(pan);
    },

    getPan: () => panOffset,

    resetView: () => {
      setZoomLevel(1.0);
      setPanOffset({ x: 0, y: 0 });
    },

    // Overlay management
    updateOverlay: () => {
      updateOverlay();
    },

    addAnnotation: (annotation) => {
      // This would typically be managed by parent component
      // but provided for convenience
      console.warn('addAnnotation should be managed via props.annotations');
    },

    // Access to underlying elements
    getRenderer: () => {
      return rendererRef.current;
    },

    getCanvas: () => {
      return canvasRef.current;
    },

    getOverlay: () => {
      return overlayRef.current;
    },

    getContainer: () => {
      return containerRef.current;
    },

    // State information
    getState: () => ({
      zoom: zoomLevel,
      pan: panOffset,
      isInteracting,
      showGrid,
      showOverlay
    })
  }), [zoomLevel, panOffset, isInteracting, showGrid, showOverlay, handleZoomChange, handlePanChange, updateOverlay]);

  return (
    <div 
      ref={containerRef}
      className={`mesh-canvas-container ${className}`}
      style={containerStyles}
    >
      <canvas
        ref={canvasRef}
        className="mesh-canvas"
        style={canvasStyles}
        {...canvasProps}
      />
      
      {/* Overlay layer for annotations */}
      {showOverlay && (
        <div
          ref={overlayRef}
          className="mesh-canvas-overlay"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            pointerEvents: 'none',
            zIndex: 10
          }}
        />
      )}
      
      {/* Optional zoom/pan controls UI */}
      {(enableZoom || enablePan) && (
        <div 
          className="mesh-canvas-controls"
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            zIndex: 20
          }}
        >
          {enableZoom && (
            <div 
              className="zoom-indicator"
              style={{
                background: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontFamily: 'monospace',
                userSelect: 'none'
              }}
            >
              {(zoomLevel * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}
    </div>
  );
});

MeshCanvas.displayName = 'MeshCanvas';

// Memoize the component to prevent unnecessary re-renders
// Only re-render if props actually change
const MemoizedMeshCanvas = memo(MeshCanvas, (prevProps, nextProps) => {
  // Custom comparison function for better performance
  // Return true if props are equal (should not re-render)
  
  // Check if handlers are the same reference
  if (prevProps.onCanvasClick !== nextProps.onCanvasClick ||
      prevProps.onZoomChange !== nextProps.onZoomChange ||
      prevProps.onPanChange !== nextProps.onPanChange) {
    return false;
  }
  
  // Check configuration props
  const configProps = [
    'className', 'backgroundColor', 'showGrid', 'enableZoom', 'enablePan',
    'minZoom', 'maxZoom', 'devicePixelRatio', 'showOverlay'
  ];
  
  for (const prop of configProps) {
    if (prevProps[prop] !== nextProps[prop]) {
      return false;
    }
  }
  
  // Deep compare style objects
  const prevStyle = prevProps.style || {};
  const nextStyle = nextProps.style || {};
  const styleKeys = [...new Set([...Object.keys(prevStyle), ...Object.keys(nextStyle)])];
  
  for (const key of styleKeys) {
    if (prevStyle[key] !== nextStyle[key]) {
      return false;
    }
  }
  
  // Compare annotations array length and content
  const prevAnnotations = prevProps.annotations || [];
  const nextAnnotations = nextProps.annotations || [];
  
  if (prevAnnotations.length !== nextAnnotations.length) {
    return false;
  }
  
  // Shallow compare annotations (assuming they're immutable)
  for (let i = 0; i < prevAnnotations.length; i++) {
    if (prevAnnotations[i] !== nextAnnotations[i]) {
      return false;
    }
  }
  
  // If we get here, props are effectively equal
  return true;
});

export default MemoizedMeshCanvas;
