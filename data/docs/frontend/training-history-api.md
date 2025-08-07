# Training History API

> **Status**: `Official`  
> **Version**: v2.0.0  
> **Maintainer**: @ZhuoQiuMcgill  
> **Last Updated**: 2025-01-07  
> **Blueprint**: `training_history`  
> **URL Prefix**: `/training/history`

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [List Training History](#list-training-history)
  - [Get Training Info](#get-training-info)
  - [Get Episode Data](#get-episode-data)
  - [Health Check](#health-check)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Frontend Integration Guide](#frontend-integration-guide)

## Overview

The Training History API provides access to historical training session data. It allows querying training session metadata, episode details, and performance statistics for analysis and visualization.

**Key Features:**
- **Session Discovery**: List all available training sessions
- **Episode Retrieval**: Access detailed episode data
- **Performance Analysis**: Get training statistics and metrics
- **Historical Insights**: Compare training runs and track progress

## Base URL

```
http://127.0.0.1:5000/training/history
```

---

## Endpoints

### List Training History

Retrieve a list of all available training session IDs.

#### Request

```http
GET /training/history/list
```

#### Response

**Success (200 OK):**
```json
{
  "training_ids": [
    "sac_20250107_143022_simple_square",
    "continue_checkpoint1_20250107_150030_triangle",
    "sac_20250106_091500_pentagon"
  ],
  "count": 3,
  "success": true
}
```

**Error (500 Internal Server Error):**
```json
{
  "error": "获取训练历史列表失败: Directory access denied",
  "training_ids": [],
  "count": 0,
  "success": false
}
```

---

### Get Training Info

Retrieve basic information about a specific training session.

#### Request

```http
POST /training/history/info/{training_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| **training_id** | string | Unique training session identifier |

#### Response

**Success (200 OK):**
```json
{
  "training_id": "sac_20250107_143022_simple_square",
  "detail_length": 1000,
  "best_episode": 847,
  "success": true
}
```

**Training Not Found (404 Not Found):**
```json
{
  "error": "训练历史不存在: invalid_training_id",
  "training_id": "invalid_training_id",
  "success": false
}
```

---

### Get Episode Data

Retrieve detailed data for a specific episode within a training session.

#### Request

```http
POST /training/history/episode/{training_id}/{episode_index}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| **training_id** | string | Unique training session identifier |
| **episode_index** | integer | Episode index (0-based) |

#### Response

**Success (200 OK):**
```json
{
  "training_id": "sac_20250107_143022_simple_square",
  "episode_index": 150,
  "episode_data": {
    "r": 0.842,
    "l": 98,
    "mesh_data": {
      "[0.0,0.0]": [
        [1.0, 0.0],
        [0.0, 1.0]
      ],
      "[1.0,0.0]": [
        [2.0, 0.0],
        [1.0, 1.0]
      ]
    },
    "boundary_vertices_data": [
      [0.0, 0.0],
      [2.0, 0.0],
      [2.0, 2.0],
      [0.0, 2.0]
    ],
    "last_ref_point": {
      "ref_vertex": [1.0, 1.0],
      "local_env_vertices": [
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
      ]
    },
    "is_completed": false
  },
  "success": true
}
```

**Episode Index Out of Range (400 Bad Request):**
```json
{
  "error": "Episode索引超出范围: 1500",
  "training_id": "sac_20250107_143022_simple_square",
  "episode_index": 1500,
  "success": false
}
```

---

### Health Check

Check the health status of the training history service.

#### Request

```http
GET /training/history/health
```

#### Response

**Success (200 OK):**
```json
{
  "status": "healthy",
  "service": "training-history-api",
  "available_trainings": 5,
  "current_focus": "sac_20250107_143022_simple_square",
  "timestamp": 1641888000.123
}
```

---

## Data Models

### Episode Data Model

```typescript
interface EpisodeData {
  r: number;                    // Episode reward
  l: number;                    // Episode length (number of steps)
  mesh_data: MeshData;          // Mesh adjacency data
  boundary_vertices_data: [number, number][]; // Boundary vertices coordinates
  last_ref_point: RefPointInfo; // Reference point information
  is_completed: boolean;        // Whether episode completed successfully
}
```

### Mesh Data Model

```typescript
interface MeshData {
  [vertexKey: string]: [number, number][]; // "[x,y]" -> [[neighbor1_x, neighbor1_y], ...]
}
```

### Reference Point Info Model

```typescript
interface RefPointInfo {
  ref_vertex: [number, number];
  local_env_vertices: [number, number][];
}
```

### Training Info Model

```typescript
interface TrainingInfo {
  training_id: string;
  detail_length: number;        // Total number of episodes
  best_episode: number;         // Index of best performing episode
  success: boolean;
}
```

---

## Error Handling

| HTTP Code | Description | Cause |
|-----------|-------------|-------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Episode index out of range |
| 404 | Not Found | Training session not found |
| 500 | Internal Server Error | File corruption, access denied, or other system error |

---

## Frontend Integration Guide

### Training History Manager

```javascript
class TrainingHistoryManager {
  constructor(apiBaseUrl = 'http://127.0.0.1:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.historyCache = new Map();
  }

  async listTrainingHistory() {
    const response = await fetch(`${this.apiBaseUrl}/training/history/list`);
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error);
    }
    
    return data.training_ids;
  }

  async getTrainingInfo(trainingId) {
    const cacheKey = `info_${trainingId}`;
    if (this.historyCache.has(cacheKey)) {
      return this.historyCache.get(cacheKey);
    }

    const response = await fetch(
      `${this.apiBaseUrl}/training/history/info/${trainingId}`,
      { method: 'POST' }
    );
    const data = await response.json();
    
    if (!response.ok || !data.success) {
      throw new Error(data.error || `Failed to get info for ${trainingId}`);
    }
    
    this.historyCache.set(cacheKey, data);
    return data;
  }

  async getEpisodeData(trainingId, episodeIndex) {
    const response = await fetch(
      `${this.apiBaseUrl}/training/history/episode/${trainingId}/${episodeIndex}`,
      { method: 'POST' }
    );
    const data = await response.json();
    
    if (!response.ok || !data.success) {
      throw new Error(data.error || `Failed to get episode ${episodeIndex} for ${trainingId}`);
    }
    
    return data.episode_data;
  }

  async getBestEpisode(trainingId) {
    const info = await this.getTrainingInfo(trainingId);
    return this.getEpisodeData(trainingId, info.best_episode);
  }

  async getEpisodeRange(trainingId, startIndex, count) {
    const episodes = [];
    const promises = [];
    
    for (let i = 0; i < count; i++) {
      const episodeIndex = startIndex + i;
      promises.push(
        this.getEpisodeData(trainingId, episodeIndex)
          .then(data => ({ index: episodeIndex, data }))
          .catch(error => ({ index: episodeIndex, error: error.message }))
      );
    }
    
    const results = await Promise.all(promises);
    return results;
  }

  clearCache() {
    this.historyCache.clear();
  }
}
```

### React Training History Component

```jsx
import { useState, useEffect } from 'react';

export function TrainingHistoryViewer() {
  const [historyManager] = useState(() => new TrainingHistoryManager());
  const [trainingIds, setTrainingIds] = useState([]);
  const [selectedTraining, setSelectedTraining] = useState(null);
  const [trainingInfo, setTrainingInfo] = useState(null);
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadTrainingHistory();
  }, []);

  const loadTrainingHistory = async () => {
    try {
      setLoading(true);
      const ids = await historyManager.listTrainingHistory();
      setTrainingIds(ids);
    } catch (error) {
      console.error('Failed to load training history:', error);
    } finally {
      setLoading(false);
    }
  };

  const selectTraining = async (trainingId) => {
    try {
      setLoading(true);
      const info = await historyManager.getTrainingInfo(trainingId);
      setSelectedTraining(trainingId);
      setTrainingInfo(info);
      setSelectedEpisode(null);
    } catch (error) {
      console.error('Failed to load training info:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadEpisode = async (episodeIndex) => {
    if (!selectedTraining) return;

    try {
      setLoading(true);
      const episodeData = await historyManager.getEpisodeData(
        selectedTraining,
        episodeIndex
      );
      setSelectedEpisode({ index: episodeIndex, data: episodeData });
    } catch (error) {
      console.error('Failed to load episode data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadBestEpisode = async () => {
    if (!trainingInfo) return;
    await loadEpisode(trainingInfo.best_episode);
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="training-history-viewer">
      <h1>Training History</h1>
      
      <div className="training-list">
        <h2>Available Training Sessions</h2>
        {trainingIds.map(id => (
          <button
            key={id}
            onClick={() => selectTraining(id)}
            className={selectedTraining === id ? 'selected' : ''}
          >
            {id}
          </button>
        ))}
      </div>

      {trainingInfo && (
        <div className="training-info">
          <h2>Training Info</h2>
          <p>Training ID: {selectedTraining}</p>
          <p>Total Episodes: {trainingInfo.detail_length}</p>
          <p>Best Episode: #{trainingInfo.best_episode}</p>
          
          <div className="episode-controls">
            <button onClick={loadBestEpisode}>
              Load Best Episode
            </button>
            <input
              type="number"
              placeholder="Episode index"
              min="0"
              max={trainingInfo.detail_length - 1}
              onChange={(e) => {
                const index = parseInt(e.target.value);
                if (!isNaN(index)) loadEpisode(index);
              }}
            />
          </div>
        </div>
      )}

      {selectedEpisode && (
        <div className="episode-viewer">
          <h2>Episode #{selectedEpisode.index}</h2>
          <div className="episode-stats">
            <p>Reward: {selectedEpisode.data.r.toFixed(3)}</p>
            <p>Length: {selectedEpisode.data.l} steps</p>
            <p>Completed: {selectedEpisode.data.is_completed ? 'Yes' : 'No'}</p>
          </div>
          
          <EpisodeMeshVisualization 
            meshData={selectedEpisode.data.mesh_data}
            boundaryVertices={selectedEpisode.data.boundary_vertices_data}
            refPointInfo={selectedEpisode.data.last_ref_point}
          />
        </div>
      )}
    </div>
  );
}
```

### Episode Analysis Component

```jsx
export function EpisodeAnalyzer({ trainingId, onAnalysisComplete }) {
  const [historyManager] = useState(() => new TrainingHistoryManager());
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const analyzeTraining = async () => {
    if (!trainingId) return;

    setAnalyzing(true);
    try {
      // Get training info
      const info = await historyManager.getTrainingInfo(trainingId);
      
      // Sample episodes for analysis (every 10th episode)
      const sampleSize = Math.min(100, Math.floor(info.detail_length / 10));
      const sampleIndices = Array.from(
        { length: sampleSize }, 
        (_, i) => Math.floor(i * info.detail_length / sampleSize)
      );
      
      // Get episode data
      const episodePromises = sampleIndices.map(index => 
        historyManager.getEpisodeData(trainingId, index)
          .then(data => ({ index, ...data }))
          .catch(() => null)
      );
      
      const episodes = (await Promise.all(episodePromises)).filter(Boolean);
      
      // Analyze performance
      const rewards = episodes.map(e => e.r);
      const lengths = episodes.map(e => e.l);
      
      const analysisResult = {
        trainingId,
        totalEpisodes: info.detail_length,
        bestEpisode: info.best_episode,
        sampleSize: episodes.length,
        rewardStats: {
          mean: rewards.reduce((a, b) => a + b, 0) / rewards.length,
          min: Math.min(...rewards),
          max: Math.max(...rewards),
          std: Math.sqrt(
            rewards.reduce((acc, val) => acc + Math.pow(val - rewards.reduce((a, b) => a + b, 0) / rewards.length, 2), 0) / rewards.length
          )
        },
        lengthStats: {
          mean: lengths.reduce((a, b) => a + b, 0) / lengths.length,
          min: Math.min(...lengths),
          max: Math.max(...lengths)
        },
        episodeData: episodes
      };
      
      setAnalysis(analysisResult);
      onAnalysisComplete?.(analysisResult);
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="episode-analyzer">
      <button 
        onClick={analyzeTraining}
        disabled={analyzing || !trainingId}
      >
        {analyzing ? 'Analyzing...' : 'Analyze Training'}
      </button>
      
      {analysis && (
        <div className="analysis-results">
          <h3>Training Analysis</h3>
          <div className="stats">
            <h4>Reward Statistics</h4>
            <p>Mean: {analysis.rewardStats.mean.toFixed(3)}</p>
            <p>Min: {analysis.rewardStats.min.toFixed(3)}</p>
            <p>Max: {analysis.rewardStats.max.toFixed(3)}</p>
            <p>Std Dev: {analysis.rewardStats.std.toFixed(3)}</p>
          </div>
          
          <div className="stats">
            <h4>Episode Length Statistics</h4>
            <p>Mean: {analysis.lengthStats.mean.toFixed(1)}</p>
            <p>Min: {analysis.lengthStats.min}</p>
            <p>Max: {analysis.lengthStats.max}</p>
          </div>
        </div>
      )}
    </div>
  );
}
```

### Best Practices

1. **Caching**: Cache training info to avoid repeated API calls
2. **Error Handling**: Handle missing episodes and training sessions gracefully
3. **Performance**: Sample episodes for analysis rather than loading all data
4. **Visualization**: Use appropriate charts for reward trends and statistics
5. **Memory Management**: Clear caches when switching between training sessions

### Usage Examples

**Load training history:**
```javascript
const historyManager = new TrainingHistoryManager();
const trainingIds = await historyManager.listTrainingHistory();
```

**Get best episode from a training session:**
```javascript
const bestEpisode = await historyManager.getBestEpisode('sac_20250107_143022_simple_square');
```

**Analyze reward progression:**
```javascript
const episodes = await historyManager.getEpisodeRange('training_id', 0, 100);
const rewards = episodes.map(e => e.data?.r).filter(r => r !== undefined);
// Use rewards array for visualization
```
