# Quality & Action APIs

> **Status**: `Official`  
> **Version**: v2.0.0  
> **Maintainer**: @ZhuoQiuMcgill  
> **Last Updated**: 2025-01-07  
> **Blueprints**: `quality`, `action`, `geometry`  

## Table of Contents

- [Overview](#overview)
- [Quality API](#quality-api)
- [Action API](#action-api)  
- [Geometry API](#geometry-api)
- [Frontend Integration Guide](#frontend-integration-guide)

## Overview

This document covers three related APIs that are essential for mesh quality analysis, action testing, and coordinate processing in the mesh generation system.

---

## Quality API

**Base URL:** `http://127.0.0.1:5000/quality`

### Get Quality Methods

List all available quality calculation methods.

#### Request

```http
GET /quality/methods
```

#### Response

**Success (200 OK):**
```json
{
  "methods": ["robust", "area", "aspect_ratio", "skewness", "hybrid"],
  "method_info": {
    "robust": {
      "description": "Combined edge and angle quality metric",
      "parameters": [],
      "range": "[0, 1]"
    },
    "hybrid": {
      "description": "Weighted combination of quality metrics",
      "parameters": ["gamma"],
      "range": "[0, 1]"
    }
  },
  "count": 5,
  "success": true
}
```

### Calculate Quality

Calculate quality score for a given quadrilateral.

#### Request

```http
POST /quality/calculate
Content-Type: application/json
```

**Body:**
```json
{
  "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
  "method": "robust",
  "gamma": 1.0
}
```

#### Response

**Success (200 OK):**
```json
{
  "quality_score": 1.0,
  "method": "robust",
  "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
  "success": true
}
```

---

## Action API

**Base URL:** `http://127.0.0.1:5000/action`

### Find Reference Point

Find the reference point for a given mesh.

#### Request

```http
GET /action/find-ref-point/{mesh_name}
```

#### Response

**Success (200 OK):**
```json
{
  "success": true,
  "reference_point": {
    "index": 10,
    "coordinates": [600.0, 300.0],
    "interior_angle": 89.99999999999994,
    "neighbor_vertices": [
      [593.338, 206.662],
      [693.338, 393.338],
      [600.0, 300.0],
      [506.662, 206.662],
      [400.0, 400.0]
    ]
  }
}
```

### Execute Action

Execute and validate a specific action.

#### Request

```http
POST /action/execute
Content-Type: application/json
```

**Body:**
```json
{
  "mesh_name": "simple_square",
  "action_type": "type1",
  "reference_point_index": 10,
  "clicked_point": [0.5, 0.5]
}
```

#### Response

**Success (200 OK):**
```json
{
  "success": true,
  "result": {
    "valid": true,
    "action_name": "type1",
    "decoded_coords": [[0.5, 0.5]],
    "generated_element": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    "polar_coordinates": {
      "r": 0.707,
      "theta": 0.785
    }
  }
}
```

---

## Geometry API

**Base URL:** `http://127.0.0.1:5000/geometry`

### Normalize Coordinates

Convert coordinates to normalized polar coordinates.

#### Request

```http
POST /geometry/normalize
Content-Type: application/json
```

**Body:**
```json
{
  "coordinates": [
    [0.0, 0.0],
    [1.0, 0.0], 
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5]
  ]
}
```

#### Response

**Success (200 OK):**
```json
{
  "status": "success",
  "original_coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
  "normalized_coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.571], [0.0, 3.141], [0.707, 0.785]],
  "ref_vertex_index": 2,
  "right_neighbor_index": 1,
  "scale_factor": 1.0,
  "average_edge_length": 1.0,
  "edges_used_for_scale": 4
}
```

---

## Frontend Integration Guide

### Quality Manager

```javascript
class QualityManager {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.methodsCache = null;
  }

  async getQualityMethods() {
    if (this.methodsCache) {
      return this.methodsCache;
    }

    const response = await fetch(`${this.apiBaseUrl}/quality/methods`);
    const data = await response.json();
    
    if (data.success) {
      this.methodsCache = data;
    }
    return data;
  }

  async calculateQuality(vertices, method = 'robust', gamma = 1.0) {
    const response = await fetch(`${this.apiBaseUrl}/quality/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vertices, method, gamma })
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error);
    }
    
    return data.quality_score;
  }

  async batchCalculateQuality(elements, method = 'robust') {
    const promises = elements.map(element => 
      this.calculateQuality(element, method)
        .catch(error => ({ error: error.message }))
    );
    
    return Promise.all(promises);
  }
}
```

### Action Tester

```javascript
class ActionTester {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
  }

  async findReferencePoint(meshName) {
    const response = await fetch(
      `${this.apiBaseUrl}/action/find-ref-point/${meshName}`
    );
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error);
    }
    
    return data.reference_point;
  }

  async executeAction(meshName, actionType, refPointIndex, clickedPoint = null) {
    const body = {
      mesh_name: meshName,
      action_type: actionType,
      reference_point_index: refPointIndex
    };
    
    if (clickedPoint) {
      body.clicked_point = clickedPoint;
    }

    const response = await fetch(`${this.apiBaseUrl}/action/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error);
    }
    
    return data.result;
  }

  async testActionSequence(meshName, actions) {
    const results = [];
    
    for (const action of actions) {
      try {
        const result = await this.executeAction(
          meshName,
          action.type,
          action.refIndex,
          action.clickPoint
        );
        results.push({ ...action, result, success: true });
      } catch (error) {
        results.push({ ...action, error: error.message, success: false });
      }
    }
    
    return results;
  }
}
```

### Geometry Processor

```javascript
class GeometryProcessor {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
  }

  async normalizeCoordinates(coordinates) {
    const response = await fetch(`${this.apiBaseUrl}/geometry/normalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coordinates })
    });
    
    const data = await response.json();
    
    if (data.status !== 'success') {
      throw new Error(data.message);
    }
    
    return {
      original: data.original_coordinates,
      normalized: data.normalized_coordinates,
      refIndex: data.ref_vertex_index,
      scaleFactor: data.scale_factor
    };
  }

  async batchNormalize(coordinateSets) {
    const promises = coordinateSets.map(coords => 
      this.normalizeCoordinates(coords)
        .catch(error => ({ error: error.message }))
    );
    
    return Promise.all(promises);
  }
}
```

### React Integration Example

```jsx
import { useState, useEffect } from 'react';

export function QualityActionTester() {
  const [qualityManager] = useState(() => new QualityManager());
  const [actionTester] = useState(() => new ActionTester());
  const [selectedMesh, setSelectedMesh] = useState('simple_square');
  const [refPoint, setRefPoint] = useState(null);
  const [qualityMethods, setQualityMethods] = useState([]);
  const [selectedMethod, setSelectedMethod] = useState('robust');

  useEffect(() => {
    loadQualityMethods();
    if (selectedMesh) {
      loadReferencePoint();
    }
  }, [selectedMesh]);

  const loadQualityMethods = async () => {
    try {
      const data = await qualityManager.getQualityMethods();
      setQualityMethods(data.methods);
    } catch (error) {
      console.error('Failed to load quality methods:', error);
    }
  };

  const loadReferencePoint = async () => {
    try {
      const refPoint = await actionTester.findReferencePoint(selectedMesh);
      setRefPoint(refPoint);
    } catch (error) {
      console.error('Failed to find reference point:', error);
    }
  };

  const testAction = async (actionType, clickPoint = null) => {
    if (!refPoint) return;

    try {
      const result = await actionTester.executeAction(
        selectedMesh,
        actionType,
        refPoint.index,
        clickPoint
      );

      if (result.valid && result.generated_element) {
        // Calculate quality of generated element
        const quality = await qualityManager.calculateQuality(
          result.generated_element,
          selectedMethod
        );
        
        console.log(`Action ${actionType} - Valid: ${result.valid}, Quality: ${quality.toFixed(3)}`);
        
        return { ...result, quality };
      }
      
      return result;
    } catch (error) {
      console.error('Action test failed:', error);
      return { valid: false, error: error.message };
    }
  };

  const testAllActions = async () => {
    const actions = ['type0_left', 'type0_right'];
    const results = [];
    
    for (const actionType of actions) {
      const result = await testAction(actionType);
      results.push({ actionType, ...result });
    }
    
    // Test type1 action with a sample point
    const type1Result = await testAction('type1', [0.5, 0.5]);
    results.push({ actionType: 'type1', ...type1Result });
    
    return results;
  };

  return (
    <div className="quality-action-tester">
      <h1>Quality & Action Tester</h1>
      
      <div className="controls">
        <div>
          <label>Mesh:</label>
          <select 
            value={selectedMesh}
            onChange={(e) => setSelectedMesh(e.target.value)}
          >
            <option value="simple_square">Simple Square</option>
            <option value="triangle">Triangle</option>
            <option value="pentagon">Pentagon</option>
          </select>
        </div>
        
        <div>
          <label>Quality Method:</label>
          <select 
            value={selectedMethod}
            onChange={(e) => setSelectedMethod(e.target.value)}
          >
            {qualityMethods.map(method => (
              <option key={method} value={method}>{method}</option>
            ))}
          </select>
        </div>
      </div>
      
      {refPoint && (
        <div className="reference-point">
          <h2>Reference Point</h2>
          <p>Index: {refPoint.index}</p>
          <p>Coordinates: [{refPoint.coordinates.join(', ')}]</p>
          <p>Interior Angle: {refPoint.interior_angle.toFixed(2)}°</p>
        </div>
      )}
      
      <div className="action-buttons">
        <button onClick={() => testAction('type0_left')}>
          Test Type0 Left
        </button>
        <button onClick={() => testAction('type0_right')}>
          Test Type0 Right
        </button>
        <button onClick={() => testAction('type1', [0.5, 0.5])}>
          Test Type1
        </button>
        <button onClick={testAllActions}>
          Test All Actions
        </button>
      </div>
    </div>
  );
}
```

### Best Practices

1. **Validation**: Always validate action results before applying them
2. **Quality Monitoring**: Calculate quality scores for generated elements
3. **Error Handling**: Handle invalid actions and coordinate processing errors
4. **Performance**: Cache method information and reference points
5. **User Feedback**: Provide clear feedback for action validity and quality scores

### Usage Examples

**Calculate element quality:**
```javascript
const quality = await qualityManager.calculateQuality(
  [[0, 0], [1, 0], [1, 1], [0, 1]], 
  'robust'
);
```

**Test action validity:**
```javascript
const result = await actionTester.executeAction(
  'simple_square', 
  'type1', 
  10, 
  [0.5, 0.5]
);
if (result.valid) {
  console.log('Action is valid!');
}
```

**Process coordinates:**
```javascript
const normalized = await geometryProcessor.normalizeCoordinates([
  [0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]
]);
```
