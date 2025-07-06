# Flask ML Training API Documentation

> **Status**: `Official`  
> **Version**: v1.0.0  
> **Maintainer**: @ML-Team  
> **Last Updated**: 2025-07-06  
> **Base URL**: `http://localhost:5000`

## Table of Contents
- [Overview](#overview)
- [Training API](#training-api)
  - [启动训练](#启动训练)
  - [停止训练](#停止训练)
  - [获取训练状态](#获取训练状态)
  - [训练健康检查](#训练健康检查)
- [Mesh API](#mesh-api)
  - [获取Mesh列表](#获取mesh列表)
  - [获取Mesh信息](#获取mesh信息)
  - [Mesh健康检查](#mesh健康检查)
- [Error Codes](#error-codes)
- [Appendix](#appendix)

## Overview
本API系统提供机器学习训练管理和网格文件管理功能，支持启动/停止训练任务、实时监控训练状态以及管理训练用的mesh几何文件。系统基于Flask框架构建，支持CORS跨域访问。

### 主要功能模块
- **Training API**: 训练过程的启动、停止和状态监控
- **Mesh API**: 网格文件的查询和信息获取

---

## Training API

### 启动训练

#### Request Endpoint
```http
POST /training/start
Content-Type: application/json
```

#### Request Parameters

##### Body Parameters
```json
{
  "mesh_name": "simple_square",
  "subfolder": "mesh",
  "max_episodes": 1000,
  "max_steps": 500
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| mesh_name | string | No | - | 要使用的网格文件名称 |
| subfolder | string | No | "mesh" | 网格文件所在的子文件夹 |
| max_episodes | integer | No | - | 最大训练轮数 |
| max_steps | integer | No | - | 每轮最大步数 |

#### Response Examples

##### Success Response (200 OK)
```json
{
  "message": "training_started",
  "success": true,
  "config": {
    "mesh_name": "simple_square",
    "subfolder": "mesh",
    "max_episodes": 1000,
    "max_steps": 500
  }
}
```

##### Failure Response (400 Bad Request)
```json
{
  "error": "训练已在运行中",
  "success": false
}
```

##### Failure Response (500 Internal Server Error)
```json
{
  "error": "启动训练时发生未知错误: 详细错误信息",
  "success": false
}
```

---

### 停止训练

#### Request Endpoint
```http
POST /training/stop
Content-Type: application/json
```

#### Request Parameters
无需请求参数

#### Response Examples

##### Success Response (200 OK)
```json
{
  "message": "stop_requested",
  "success": true
}
```

##### Failure Response (500 Internal Server Error)
```json
{
  "error": "停止训练时发生错误: 详细错误信息",
  "success": false
}
```

---

### 获取训练状态

#### Request Endpoint
```http
GET /training/status
```

#### Request Parameters
无需请求参数

#### Response Examples

##### Success Response (200 OK)
```json
{
  "running": true,
  "status": "training",
  "stats": {
    "episode": 150,
    "total_steps": 75000,
    "episode_reward": 125.5,
    "average_reward": 98.2,
    "buffer_size": 10000
  },
  "progress": {
    "current_episode": 150,
    "total_steps": 75000,
    "latest_reward": 125.5,
    "average_reward": 98.2,
    "buffer_utilization": 10000
  },
  "timestamp": 1720285200.123
}
```

##### Failure Response (500 Internal Server Error)
```json
{
  "running": false,
  "status": "error",
  "stats": null,
  "error": "获取状态时发生错误: 详细错误信息",
  "timestamp": 1720285200.123
}
```

---

### 训练健康检查

#### Request Endpoint
```http
GET /training/health
```

#### Request Parameters
无需请求参数

#### Response Examples

##### Success Response (200 OK)
```json
{
  "status": "healthy",
  "service": "training-api",
  "manager_running": false,
  "timestamp": 1720285200.123
}
```

---

## Mesh API

### 获取Mesh列表

#### Request Endpoint
```http
GET /mesh/list
```

#### Request Parameters

##### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subfolder | string | No | "mesh" | 子文件夹名称 |

#### Response Examples

##### Success Response (200 OK)
```json
{
  "meshes": [
    "1",
    "simple_square", 
    "triangle",
    "rectangle",
    "pentagon",
    "hexagon"
  ],
  "count": 6
}
```

##### Failure Response (500 Internal Server Error)
```json
{
  "error": "获取mesh列表失败: 详细错误信息",
  "meshes": [],
  "count": 0
}
```

---

### 获取Mesh信息

#### Request Endpoint
```http
GET /mesh/info/<mesh_name>
```

#### Request Parameters

##### Path Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| **mesh_name** | string | 网格文件名称 |

##### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subfolder | string | No | "mesh" | 子文件夹名称 |

#### Response Examples

##### Success Response (200 OK)
```json
{
  "name": "simple_square",
  "subfolder": "mesh",
  "exists": true,
  "vertex_count": 4,
  "file_size": 128,
  "error": null
}
```

##### File Not Found Response (500 Internal Server Error)
```json
{
  "name": "nonexistent_mesh",
  "subfolder": "mesh", 
  "exists": false,
  "vertex_count": 0,
  "file_size": 0,
  "error": "文件不存在"
}
```

##### System Error Response (500 Internal Server Error)
```json
{
  "name": "simple_square",
  "subfolder": "mesh",
  "exists": false,
  "vertex_count": 0,
  "file_size": 0,
  "error": "获取mesh信息失败: 详细错误信息"
}
```

---

### Mesh健康检查

#### Request Endpoint
```http
GET /mesh/health
```

#### Request Parameters
无需请求参数

#### Response Examples

##### Success Response (200 OK)
```json
{
  "status": "healthy",
  "service": "mesh-api",
  "timestamp": 1720285200.123
}
```

---

## Error Codes

| Error Code | HTTP Status Code | Description | Example |
|------------|------------------|-------------|---------|
| - | 400 | 请求参数错误或训练已在运行 | 训练已在运行中 |
| - | 500 | 服务器内部错误 | 启动训练时发生未知错误 |
| - | 500 | 无法获取训练状态 | 获取状态时发生错误 |
| - | 500 | 无法获取Mesh信息 | 获取mesh列表失败 |

## Appendix

### 注意事项
1. **CORS配置**: API已配置CORS支持，允许前端跨域访问`/training/*`和`/mesh/*`路径
2. **健康检查**: 每个API模块都提供`/health`端点用于服务状态监控
3. **错误处理**: 所有接口都包含详细的错误信息和状态码
4. **实时状态**: 训练状态接口提供实时的训练进度和统计信息

### 环境配置
- **开发环境**: `http://localhost:5000`
- **调试模式**: 默认启用
- **CORS**: 允许所有来源访问训练和mesh相关API

### 版本历史
- v1.0.0 (2025-07-06): 初始版本，包含训练管理和mesh文件管理功能

### 使用示例

#### JavaScript调用示例
```javascript
// 启动训练
const response = await fetch('http://localhost:5000/training/start', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    mesh_name: 'simple_square',
    max_episodes: 1000
  })
});

// 获取训练状态
const status = await fetch('http://localhost:5000/training/status');
const data = await status.json();
console.log('训练状态:', data);
```

#### Python调用示例
```python
import requests

# 启动训练
response = requests.post('http://localhost:5000/training/start', 
                        json={
                          'mesh_name': 'simple_square',
                          'max_episodes': 1000
                        })

# 获取mesh列表
meshes = requests.get('http://localhost:5000/mesh/list')
print('可用的mesh:', meshes.json())
```