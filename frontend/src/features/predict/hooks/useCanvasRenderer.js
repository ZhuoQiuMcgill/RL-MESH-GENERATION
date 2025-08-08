import { useState, useCallback, useRef, useEffect } from 'react';
import { usePredictSession } from '../contexts/PredictSessionContext';

// 渲染模式常量
export const RenderMode = {
  WIREFRAME: 'wireframe',
  SOLID: 'solid',
  POINTS: 'points',
  HYBRID: 'hybrid'
};

// 视图模式常量
export const ViewMode = {
  PERSPECTIVE: 'perspective',
  ORTHOGRAPHIC: 'orthographic'
};

// 相机控制类型
export const CameraControl = {
  ORBIT: 'orbit',
  FLY: 'fly',
  FIRST_PERSON: 'first_person'
};

/**
 * 画布渲染相关的自定义Hook
 * 提供3D网格渲染、视图控制、交互等功能
 */
export const useCanvasRenderer = () => {
  const { meshData, refPoint, actions } = usePredictSession();
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameRef = useRef(null);

  // 渲染状态
  const [renderState, setRenderState] = useState({
    isInitialized: false,
    isRendering: false,
    renderMode: RenderMode.SOLID,
    viewMode: ViewMode.PERSPECTIVE,
    cameraControl: CameraControl.ORBIT,
    showWireframe: true,
    showNodes: false,
    showAxes: true,
    showGrid: true,
    backgroundColor: '#f0f0f0',
    wireframeColor: '#333333',
    meshColor: '#4A90E2',
    nodeColor: '#ff6b35',
    refPointColor: '#ff0000'
  });

  // 性能监控状态
  const [performanceState, setPerformanceState] = useState({
    fps: 0,
    frameTime: 0,
    vertexCount: 0,
    triangleCount: 0,
    drawCalls: 0
  });

  // 相机状态
  const [cameraState, setCameraState] = useState({
    position: { x: 10, y: 10, z: 10 },
    target: { x: 0, y: 0, z: 0 },
    zoom: 1,
    fov: 75,
    near: 0.1,
    far: 1000
  });

  // 错误状态
  const [renderError, setRenderError] = useState(null);

  // 初始化渲染器
  const initializeRenderer = useCallback(async () => {
    if (!canvasRef.current || renderState.isInitialized) return;

    try {
      setRenderError(null);
      
      // 这里应该集成实际的3D渲染库（如Three.js, Babylon.js等）
      // 以下是伪代码示例
      
      // 创建渲染器
      // rendererRef.current = new THREE.WebGLRenderer({ 
      //   canvas: canvasRef.current,
      //   antialias: true,
      //   alpha: true
      // });
      
      // 创建场景
      // sceneRef.current = new THREE.Scene();
      
      // 创建相机
      // cameraRef.current = new THREE.PerspectiveCamera(
      //   cameraState.fov,
      //   canvasRef.current.offsetWidth / canvasRef.current.offsetHeight,
      //   cameraState.near,
      //   cameraState.far
      // );
      
      // 创建控制器
      // controlsRef.current = new OrbitControls(
      //   cameraRef.current,
      //   canvasRef.current
      // );

      setRenderState(prev => ({
        ...prev,
        isInitialized: true
      }));

      // 开始渲染循环
      startRenderLoop();
      
    } catch (error) {
      setRenderError(`渲染器初始化失败: ${error.message}`);
      actions.addLog({
        level: 'error',
        message: '3D渲染器初始化失败',
        data: { error: error.message }
      });
    }
  }, [renderState.isInitialized, cameraState, actions]);

  // 开始渲染循环
  const startRenderLoop = useCallback(() => {
    const lastFrameTime = performance.now();
    let frameCount = 0;
    let lastFpsUpdate = performance.now();

    const render = (currentTime) => {
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

      // 计算帧时间
      const frameTime = currentTime - lastFrameTime;
      frameCount++;

      // 每秒更新一次FPS
      if (currentTime - lastFpsUpdate >= 1000) {
        setPerformanceState(prev => ({
          ...prev,
          fps: Math.round((frameCount * 1000) / (currentTime - lastFpsUpdate)),
          frameTime: frameTime
        }));
        frameCount = 0;
        lastFpsUpdate = currentTime;
      }

      // 更新控制器
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      // 渲染场景
      try {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      } catch (error) {
        setRenderError(`渲染失败: ${error.message}`);
        return;
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);
    
    setRenderState(prev => ({
      ...prev,
      isRendering: true
    }));
  }, []);

  // 停止渲染循环
  const stopRenderLoop = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    setRenderState(prev => ({
      ...prev,
      isRendering: false
    }));
  }, []);

  // 更新网格数据
  const updateMeshData = useCallback((newMeshData) => {
    if (!sceneRef.current || !newMeshData) return;

    try {
      // 清除现有网格
      // const meshObjects = sceneRef.current.children.filter(child => 
      //   child.userData.type === 'mesh'
      // );
      // meshObjects.forEach(mesh => sceneRef.current.remove(mesh));

      // 添加新网格
      // const geometry = createGeometryFromMeshData(newMeshData);
      // const material = createMeshMaterial();
      // const mesh = new THREE.Mesh(geometry, material);
      // mesh.userData.type = 'mesh';
      // sceneRef.current.add(mesh);

      // 更新性能统计
      setPerformanceState(prev => ({
        ...prev,
        vertexCount: newMeshData.vertices?.length || 0,
        triangleCount: newMeshData.triangles?.length || 0
      }));

    } catch (error) {
      setRenderError(`网格更新失败: ${error.message}`);
    }
  }, []);

  // 更新参考点
  const updateRefPoint = useCallback((point) => {
    if (!sceneRef.current || !point) return;

    try {
      // 移除现有参考点
      // const existingRefPoint = sceneRef.current.getObjectByName('refPoint');
      // if (existingRefPoint) sceneRef.current.remove(existingRefPoint);

      // 添加新参考点
      // const geometry = new THREE.SphereGeometry(0.1, 16, 16);
      // const material = new THREE.MeshBasicMaterial({ 
      //   color: renderState.refPointColor 
      // });
      // const refPointMesh = new THREE.Mesh(geometry, material);
      // refPointMesh.position.set(point.x, point.y, point.z);
      // refPointMesh.name = 'refPoint';
      // sceneRef.current.add(refPointMesh);

    } catch (error) {
      setRenderError(`参考点更新失败: ${error.message}`);
    }
  }, [renderState.refPointColor]);

  // 更新渲染设置
  const updateRenderSettings = useCallback((newSettings) => {
    setRenderState(prev => ({
      ...prev,
      ...newSettings
    }));

    // 应用新设置到渲染器
    if (rendererRef.current && sceneRef.current) {
      // 更新背景色
      if (newSettings.backgroundColor) {
        sceneRef.current.background = new THREE.Color(newSettings.backgroundColor);
      }

      // 更新材质设置
      if (newSettings.renderMode || newSettings.wireframeColor || newSettings.meshColor) {
        // updateMaterials(newSettings);
      }
    }
  }, []);

  // 相机控制方法
  const setCameraPosition = useCallback((position) => {
    if (!cameraRef.current) return;

    // cameraRef.current.position.set(position.x, position.y, position.z);
    setCameraState(prev => ({
      ...prev,
      position
    }));
  }, []);

  const setCameraTarget = useCallback((target) => {
    if (!controlsRef.current) return;

    // controlsRef.current.target.set(target.x, target.y, target.z);
    setCameraState(prev => ({
      ...prev,
      target
    }));
  }, []);

  const resetCamera = useCallback(() => {
    const defaultPosition = { x: 10, y: 10, z: 10 };
    const defaultTarget = { x: 0, y: 0, z: 0 };
    
    setCameraPosition(defaultPosition);
    setCameraTarget(defaultTarget);
  }, [setCameraPosition, setCameraTarget]);

  const fitCameraToMesh = useCallback(() => {
    if (!cameraRef.current || !meshData) return;

    // 计算网格包围盒
    // const box = calculateBoundingBox(meshData);
    // const center = box.getCenter();
    // const size = box.getSize();
    // const maxDim = Math.max(size.x, size.y, size.z);
    // const distance = maxDim * 2;
    
    // 设置相机位置
    // setCameraPosition({
    //   x: center.x + distance,
    //   y: center.y + distance,
    //   z: center.z + distance
    // });
    // setCameraTarget(center);
  }, [meshData, setCameraPosition, setCameraTarget]);

  // 截图功能
  const captureScreenshot = useCallback((format = 'png', quality = 0.92) => {
    if (!canvasRef.current) return null;

    return new Promise((resolve) => {
      canvasRef.current.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        resolve(url);
      }, `image/${format}`, quality);
    });
  }, []);

  // 导出网格数据
  const exportMeshData = useCallback((format = 'obj') => {
    if (!meshData) return null;

    // 根据格式导出网格数据
    let content = '';
    
    switch (format) {
      case 'obj':
        // content = exportToOBJ(meshData);
        break;
      case 'ply':
        // content = exportToPLY(meshData);
        break;
      case 'stl':
        // content = exportToSTL(meshData);
        break;
      default:
        content = JSON.stringify(meshData, null, 2);
        break;
    }

    return content;
  }, [meshData]);

  // 处理画布大小变化
  const handleResize = useCallback(() => {
    if (!canvasRef.current || !rendererRef.current || !cameraRef.current) return;

    const canvas = canvasRef.current;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // 更新渲染器大小
    rendererRef.current.setSize(width, height, false);
    
    // 更新相机纵横比
    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
  }, []);

  // 清理资源
  const cleanup = useCallback(() => {
    stopRenderLoop();

    if (rendererRef.current) {
      rendererRef.current.dispose();
      rendererRef.current = null;
    }

    if (sceneRef.current) {
      // 清理场景中的所有对象
      // sceneRef.current.clear();
      sceneRef.current = null;
    }

    if (controlsRef.current) {
      controlsRef.current.dispose();
      controlsRef.current = null;
    }

    cameraRef.current = null;

    setRenderState(prev => ({
      ...prev,
      isInitialized: false,
      isRendering: false
    }));
  }, [stopRenderLoop]);

  // 监听网格数据变化
  useEffect(() => {
    if (meshData && renderState.isInitialized) {
      updateMeshData(meshData);
    }
  }, [meshData, renderState.isInitialized, updateMeshData]);

  // 监听参考点变化
  useEffect(() => {
    if (refPoint && renderState.isInitialized) {
      updateRefPoint(refPoint);
    }
  }, [refPoint, renderState.isInitialized, updateRefPoint]);

  // 监听窗口大小变化
  useEffect(() => {
    const handleWindowResize = () => {
      setTimeout(handleResize, 100); // 延迟处理以确保DOM更新完成
    };

    window.addEventListener('resize', handleWindowResize);
    return () => window.removeEventListener('resize', handleWindowResize);
  }, [handleResize]);

  // 组件卸载时清理
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    // Refs
    canvasRef,
    
    // 状态
    renderState,
    performanceState,
    cameraState,
    renderError,
    
    // 初始化和控制
    initializeRenderer,
    startRenderLoop,
    stopRenderLoop,
    cleanup,
    
    // 渲染设置
    updateRenderSettings,
    updateMeshData,
    updateRefPoint,
    
    // 相机控制
    setCameraPosition,
    setCameraTarget,
    resetCamera,
    fitCameraToMesh,
    handleResize,
    
    // 工具功能
    captureScreenshot,
    exportMeshData,
    
    // 常量
    RenderMode,
    ViewMode,
    CameraControl
  };
};

export default useCanvasRenderer;
