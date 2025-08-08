import React, { useRef, useCallback, useState } from 'react';
import MeshCanvas from './MeshCanvas';
import { Button } from './index';

/**
 * MeshCanvasExample - MeshCanvas 组件使用示例
 * 演示如何使用 MeshCanvas 组件进行网格渲染和交互
 */
const MeshCanvasExample = () => {
  const canvasRef = useRef(null);
  const [clickedPoint, setClickedPoint] = useState(null);
  
  // 示例边界数据
  const sampleBoundary = [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
  ];

  // 示例网格数据
  const sampleMeshData = {
    "[25, 25]": [[50, 25], [25, 50]],
    "[50, 25]": [[25, 25], [75, 25], [50, 50]],
    "[75, 25]": [[50, 25], [75, 50]],
    "[25, 50]": [[25, 25], [50, 50], [25, 75]],
    "[50, 50]": [[25, 50], [50, 25], [75, 50], [50, 75]],
    "[75, 50]": [[75, 25], [50, 50], [75, 75]],
    "[25, 75]": [[25, 50], [50, 75]],
    "[50, 75]": [[25, 75], [50, 50], [75, 75]],
    "[75, 75]": [[75, 50], [50, 75]]
  };

  // 示例参考点信息
  const sampleRefPointInfo = {
    ref_vertex: [50, 50],
    local_env_vertices: [[25, 50], [50, 25], [75, 50], [50, 75]]
  };

  // 处理画布点击
  const handleCanvasClick = useCallback((worldCoords, event) => {
    if (worldCoords) {
      setClickedPoint(worldCoords);
      console.log('Clicked at world coordinates:', worldCoords);
      console.log('Screen coordinates:', event.clientX, event.clientY);
    } else {
      console.log('Click occurred but no valid coordinate transform available');
    }
  }, []);

  // 清空画布
  const clearCanvas = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
      setClickedPoint(null);
    }
  }, []);

  // 渲染边界预览
  const renderBoundaryPreview = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.renderBoundaryPreview(sampleBoundary, 'Sample Square Mesh');
    }
  }, []);

  // 渲染完整场景
  const renderFullScene = useCallback(() => {
    if (canvasRef.current) {
      const refPointWithClick = clickedPoint ? {
        ...sampleRefPointInfo,
        clicked_point: clickedPoint
      } : sampleRefPointInfo;

      canvasRef.current.renderScene(sampleMeshData, sampleBoundary, refPointWithClick);
    }
  }, [clickedPoint]);

  // 手动触发重绘
  const handleResize = useCallback(() => {
    if (canvasRef.current) {
      canvasRef.current.onResize();
    }
  }, []);

  // 获取坐标转换信息
  const getTransformInfo = useCallback(() => {
    if (canvasRef.current) {
      const transform = canvasRef.current.getCurrentTransform();
      console.log('Current transform:', transform);
      
      if (transform) {
        // 测试坐标转换
        const testWorld = [50, 50];
        const screenCoords = canvasRef.current.worldToScreen(testWorld);
        const backToWorld = canvasRef.current.screenToWorld(screenCoords[0], screenCoords[1]);
        
        console.log('Coordinate transformation test:');
        console.log('Original world coords:', testWorld);
        console.log('Converted to screen:', screenCoords);
        console.log('Converted back to world:', backToWorld);
      }
    }
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h2>MeshCanvas 示例</h2>
      
      {/* 控制按钮 */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <Button onClick={clearCanvas}>清空画布</Button>
        <Button onClick={renderBoundaryPreview}>渲染边界预览</Button>
        <Button onClick={renderFullScene}>渲染完整场景</Button>
        <Button onClick={handleResize}>手动重绘</Button>
        <Button onClick={getTransformInfo}>获取坐标信息</Button>
      </div>

      {/* 点击信息显示 */}
      {clickedPoint && (
        <div style={{ 
          marginBottom: '20px', 
          padding: '10px', 
          backgroundColor: '#f0f8ff',
          border: '1px solid #ccc',
          borderRadius: '4px'
        }}>
          <strong>最后点击的坐标:</strong> ({clickedPoint[0].toFixed(2)}, {clickedPoint[1].toFixed(2)})
        </div>
      )}

      {/* 画布容器 */}
      <div style={{ 
        width: '100%', 
        height: '500px', 
        border: '2px solid #333',
        borderRadius: '8px',
        backgroundColor: '#f5f5f5'
      }}>
        <MeshCanvas
          ref={canvasRef}
          onCanvasClick={handleCanvasClick}
          className="example-mesh-canvas"
          style={{ 
            width: '100%', 
            height: '100%'
          }}
        />
      </div>

      {/* 使用说明 */}
      <div style={{ marginTop: '20px' }}>
        <h3>使用说明:</h3>
        <ul>
          <li>点击"渲染边界预览"显示简单的边界线</li>
          <li>点击"渲染完整场景"显示网格、边界和参考点</li>
          <li>在画布上点击可以获取世界坐标</li>
          <li>点击的坐标会作为新的点显示在场景中</li>
          <li>画布会自动响应窗口大小变化</li>
        </ul>
      </div>
    </div>
  );
};

export default MeshCanvasExample;
