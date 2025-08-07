# Mesh Management API

> **Status**: `Official`  
> **Version**: v2.0.0  
> **Maintainer**: @ZhuoQiuMcgill  
> **Last Updated**: 2025-01-07  
> **Blueprint**: `mesh`  
> **URL Prefix**: `/mesh`

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [List Meshes](#list-meshes)
  - [Get Mesh Info](#get-mesh-info)
  - [Get Mesh Boundary](#get-mesh-boundary)
  - [Health Check](#health-check)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Frontend Integration Guide](#frontend-integration-guide)

## Overview

The Mesh Management API provides access to mesh files and boundary data for training and visualization. It handles mesh file discovery, metadata retrieval, and boundary vertex extraction.

**Key Features:**
- **Mesh Discovery**: List all available mesh files
- **Metadata Access**: Get mesh file information and statistics
- **Boundary Extraction**: Retrieve boundary vertices for visualization
- **File Validation**: Check mesh file existence and validity

## Base URL

```
http://127.0.0.1:5000/mesh
```

---

## Endpoints

### List Meshes

Retrieve a list of all available mesh files.

#### Request

```http
GET /mesh/list?subfolder=mesh
```

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subfolder | string | No | "mesh" | Subfolder to search for mesh files |

#### Response

**Success (200 OK):**
```json
{
  "meshes": [
    "1",
    "2", 
    "3",
    "simple_square",
    "triangle",
    "rectangle",
    "pentagon",
    "hexagon"
  ],
  "count": 8
}
```

**Error (500 Internal Server Error):**
```json
{
  "error": "获取mesh列表失败: Directory not found",
  "meshes": [],
  "count": 0
}
```

---

### Get Mesh Info

Retrieve detailed information about a specific mesh file.

#### Request

```http
GET /mesh/info/{mesh_name}?subfolder=mesh
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| **mesh_name** | string | Mesh file name (without .txt extension) |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subfolder | string | No | "mesh" | Subfolder containing the mesh file |

#### Response

**Success (200 OK):**
```json
{
  "name": "simple_square",
  "subfolder": "mesh",
  "configured_path": "/project/data/mesh",
  "file_path": "/project/data/mesh/simple_square.txt",
  "exists": true,
  "vertex_count": 4,
  "file_size": 128,
  "error": null
}
```

**File Not Found (200 OK):**
```json
{
  "name": "nonexistent_mesh",
  "subfolder": "mesh", 
  "exists": false,
  "vertex_count": 0,
  "file_size": 0,
  "error": "File does not exist"
}
```

---

### Get Mesh Boundary

Retrieve boundary vertices for a specific mesh.

#### Request

```http
GET /mesh/boundary/{mesh_name}?subfolder=mesh
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| **mesh_name** | string | Mesh file name (without .txt extension) |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subfolder | string | No | "mesh" | Subfolder containing the mesh file |

#### Response

**Success (200 OK):**
```json
{
  "mesh_name": "simple_square",
  "subfolder": "mesh",
  "boundary_vertices": [
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 2.0],
    [0.0, 2.0]
  ],
  "vertex_count": 4,
  "success": true
}
```

**File Not Found (404 Not Found):**
```json
{
  "error": "Mesh文件不存在: nonexistent_mesh",
  "success": false
}
```

---

### Health Check

Check the health status of the mesh management service.

#### Request

```http
GET /mesh/health
```

#### Response

**Success (200 OK):**
```json
{
  "status": "healthy",
  "service": "mesh-api",
  "timestamp": 1641888000.123
}
```

---

## Data Models

### Mesh Info Model

```typescript
interface MeshInfo {
  name: string;
  subfolder: string;
  configured_path?: string;
  file_path?: string;
  exists: boolean;
  vertex_count: number;
  file_size: number;
  error?: string | null;
}
```

### Mesh Boundary Model

```typescript
interface MeshBoundary {
  mesh_name: string;
  subfolder: string;
  boundary_vertices: [number, number][];
  vertex_count: number;
  success: boolean;
}
```

### Vertex Coordinate

```typescript
type Vertex = [number, number]; // [x, y]
```

---

## Error Handling

| HTTP Code | Description | Cause |
|-----------|-------------|-------|
| 200 | Success | Request processed successfully |
| 404 | Not Found | Mesh file does not exist |
| 500 | Internal Server Error | File system error or processing failure |

### Common Error Response Format

```json
{
  "error": "Error description",
  "success": false
}
```

---

## Frontend Integration Guide

### Mesh Manager Service

```javascript
class MeshManager {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.meshCache = new Map();
  }

  async listMeshes(subfolder = 'mesh') {
    const response = await fetch(
      `${this.apiBaseUrl}/mesh/list?subfolder=${subfolder}`
    );
    const data = await response.json();
    return data;
  }

  async getMeshInfo(meshName, subfolder = 'mesh') {
    const cacheKey = `${meshName}_${subfolder}`;
    if (this.meshCache.has(cacheKey)) {
      return this.meshCache.get(cacheKey);
    }

    const response = await fetch(
      `${this.apiBaseUrl}/mesh/info/${meshName}?subfolder=${subfolder}`
    );
    const data = await response.json();
    
    if (data.exists) {
      this.meshCache.set(cacheKey, data);
    }
    
    return data;
  }

  async getBoundaryVertices(meshName, subfolder = 'mesh') {
    const response = await fetch(
      `${this.apiBaseUrl}/mesh/boundary/${meshName}?subfolder=${subfolder}`
    );
    const data = await response.json();
    
    if (!response.ok || !data.success) {
      throw new Error(data.error || `Failed to load boundary for ${meshName}`);
    }
    
    return data.boundary_vertices;
  }

  clearCache() {
    this.meshCache.clear();
  }
}
```

### React Component Example

```jsx
import { useState, useEffect } from 'react';

export function MeshSelector({ onMeshSelect }) {
  const [meshes, setMeshes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedMesh, setSelectedMesh] = useState(null);
  const [meshManager] = useState(() => new MeshManager());

  useEffect(() => {
    loadMeshes();
  }, []);

  const loadMeshes = async () => {
    try {
      setLoading(true);
      const data = await meshManager.listMeshes();
      setMeshes(data.meshes || []);
    } catch (error) {
      console.error('Failed to load meshes:', error);
    } finally {
      setLoading(false);
    }
  };

  const selectMesh = async (meshName) => {
    try {
      const [info, boundary] = await Promise.all([
        meshManager.getMeshInfo(meshName),
        meshManager.getBoundaryVertices(meshName)
      ]);
      
      const meshData = {
        name: meshName,
        info,
        boundary,
        vertices: boundary
      };
      
      setSelectedMesh(meshData);
      onMeshSelect?.(meshData);
    } catch (error) {
      console.error('Failed to load mesh data:', error);
    }
  };

  if (loading) {
    return <div>Loading meshes...</div>;
  }

  return (
    <div className="mesh-selector">
      <h3>Select Mesh</h3>
      <div className="mesh-list">
        {meshes.map(mesh => (
          <button
            key={mesh}
            onClick={() => selectMesh(mesh)}
            className={selectedMesh?.name === mesh ? 'selected' : ''}
          >
            {mesh}
          </button>
        ))}
      </div>
      
      {selectedMesh && (
        <div className="mesh-info">
          <h4>Mesh Info</h4>
          <p>Name: {selectedMesh.name}</p>
          <p>Vertices: {selectedMesh.info.vertex_count}</p>
          <p>File Size: {selectedMesh.info.file_size} bytes</p>
        </div>
      )}
    </div>
  );
}
```

### Mesh Visualization Component

```jsx
import { useRef, useEffect } from 'react';

export function MeshVisualizer({ boundary, width = 400, height = 400 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!boundary || boundary.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate bounds
    const xs = boundary.map(v => v[0]);
    const ys = boundary.map(v => v[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    
    // Add padding
    const padding = 20;
    const scaleX = (width - 2 * padding) / (maxX - minX);
    const scaleY = (height - 2 * padding) / (maxY - minY);
    const scale = Math.min(scaleX, scaleY);
    
    // Center the mesh
    const offsetX = (width - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (height - (maxY - minY) * scale) / 2 - minY * scale;
    
    // Draw boundary
    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    boundary.forEach((vertex, index) => {
      const x = vertex[0] * scale + offsetX;
      const y = height - (vertex[1] * scale + offsetY); // Flip Y axis
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.closePath();
    ctx.stroke();
    
    // Draw vertices
    ctx.fillStyle = '#dc3545';
    boundary.forEach(vertex => {
      const x = vertex[0] * scale + offsetX;
      const y = height - (vertex[1] * scale + offsetY);
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
    
  }, [boundary, width, height]);

  return (
    <div className="mesh-visualizer">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ border: '1px solid #ccc' }}
      />
    </div>
  );
}
```

### Best Practices

1. **Caching**: Cache mesh info and boundary data to avoid repeated API calls
2. **Error Handling**: Handle file not found and network errors gracefully
3. **Loading States**: Show loading indicators for async operations
4. **Validation**: Validate mesh data before visualization
5. **Performance**: Use efficient rendering for complex meshes

### Usage Examples

**Load and display mesh list:**
```javascript
const meshManager = new MeshManager();
const meshes = await meshManager.listMeshes();
console.log('Available meshes:', meshes.meshes);
```

**Load mesh boundary for visualization:**
```javascript
const boundary = await meshManager.getBoundaryVertices('simple_square');
// Use boundary data for Canvas or SVG rendering
```

**Check mesh existence before use:**
```javascript
const info = await meshManager.getMeshInfo('my_mesh');
if (info.exists) {
  // Proceed with mesh operations
} else {
  console.error('Mesh not found:', info.error);
}
```
