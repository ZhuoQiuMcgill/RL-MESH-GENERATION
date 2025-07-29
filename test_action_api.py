#!/usr/bin/env python3
"""
Simple test script for Action API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_action_health():
    """Test action API health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/action/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_mesh_list():
    """Test mesh list endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/mesh/list")
        print(f"Mesh list: {response.status_code}")
        data = response.json()
        print(f"Available meshes data: {data}")
        meshes = data.get('meshes', [])
        print(f"Available meshes: {len(meshes)}")
        return meshes
    except Exception as e:
        print(f"Mesh list failed: {e}")
        return []

def test_find_reference_point(mesh_name):
    """Test find reference point endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/action/find-ref-point/{mesh_name}")
        print(f"Find reference point: {response.status_code}")
        data = response.json()
        print(f"Reference point result: {data}")
        return data
    except Exception as e:
        print(f"Find reference point failed: {e}")
        return None

def test_execute_action(mesh_name, action_type, ref_index):
    """Test execute action endpoint"""
    try:
        payload = {
            "mesh_name": mesh_name,
            "action_type": action_type,
            "reference_point_index": ref_index
        }
        
        if action_type == "type1":
            payload["clicked_point"] = [0.5, 0.5]  # Test point
        
        response = requests.post(
            f"{BASE_URL}/action/execute", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Execute action {action_type}: {response.status_code}")
        data = response.json()
        print(f"Execution result: {data}")
        return data
    except Exception as e:
        print(f"Execute action failed: {e}")
        return None

def main():
    print("=== Testing Action API ===")
    
    # Test health
    if not test_action_health():
        print("Health check failed, stopping tests")
        return
    
    # Get available meshes
    meshes = test_mesh_list()
    if not meshes:
        print("No meshes available, stopping tests")
        return
    
    # Use first available mesh
    if isinstance(meshes[0], dict):
        test_mesh = meshes[0]['name']
    else:
        test_mesh = meshes[0]
    print(f"\nUsing test mesh: {test_mesh}")
    
    # Test reference point finding
    ref_result = test_find_reference_point(test_mesh)
    if not ref_result or not ref_result.get('success'):
        print("Reference point test failed, stopping")
        return
    
    ref_index = ref_result['reference_point']['index']
    print(f"Found reference point at index: {ref_index}")
    
    # Test action execution
    print("\n=== Testing Actions ===")
    for action_type in ['type0_left', 'type0_right', 'type1']:
        print(f"\nTesting {action_type}:")
        result = test_execute_action(test_mesh, action_type, ref_index)
        if result and result.get('success'):
            print(f"  ✓ Action {action_type} executed successfully")
            print(f"  Valid: {result['result']['valid']}")
        else:
            print(f"  ✗ Action {action_type} failed")

if __name__ == "__main__":
    main()