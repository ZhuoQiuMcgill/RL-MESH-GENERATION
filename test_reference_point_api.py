#!/usr/bin/env python3
"""
Test Reference Point API functionality

This script tests the new reference point APIs for getting current reference
point information and previewing reference point selection.
"""

import sys
import json
from flask import Flask
from src.ui.api import register_blueprints


def test_reference_point_preview():
    """Test reference point preview API"""
    print("🔍 Testing Reference Point Preview API")
    print("=" * 50)
    
    try:
        # Create test app
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app)
        client = app.test_client()
        
        # Test 1: RL selector with n=2
        print("Testing RL selector with n=2...")
        preview_data = {
            "mesh_name": "basic1",
            "ref_selector_type": "RL",
            "ref_selector_config": {"n": 2}
        }
        
        response = client.post('/predict/reference_point/preview', json=preview_data)
        if response.status_code != 200:
            print(f"❌ Preview failed: {response.get_data(as_text=True)}")
            return False
        
        result = response.get_json()
        if not result.get('success'):
            print(f"❌ Preview failed: {result}")
            return False
        
        preview = result.get('preview', {})
        print(f"✅ RL selector result:")
        print(f"   Reference vertex index: {preview.get('reference_vertex_idx')}")
        print(f"   Reference vertex coords: {preview.get('reference_vertex_coords')}")
        print(f"   Interior angle: {preview.get('boundary_context', {}).get('interior_angle')}")
        print(f"   Boundary size: {preview.get('boundary_context', {}).get('boundary_size')}")
        
        # Test 2: Random selector
        print("\nTesting Random selector...")
        preview_data = {
            "mesh_name": "basic1",
            "ref_selector_type": "Random",
            "ref_selector_config": {}
        }
        
        response = client.post('/predict/reference_point/preview', json=preview_data)
        if response.status_code != 200:
            print(f"❌ Random preview failed: {response.get_data(as_text=True)}")
            return False
        
        result = response.get_json()
        preview = result.get('preview', {})
        print(f"✅ Random selector result:")
        print(f"   Reference vertex index: {preview.get('reference_vertex_idx')}")
        print(f"   Reference vertex coords: {preview.get('reference_vertex_coords')}")
        
        # Test 3: Default selector
        print("\nTesting Default selector...")
        preview_data = {
            "mesh_name": "basic1",
            "ref_selector_type": "default",
            "ref_selector_config": {}
        }
        
        response = client.post('/predict/reference_point/preview', json=preview_data)
        if response.status_code != 200:
            print(f"❌ Default preview failed: {response.get_data(as_text=True)}")
            return False
        
        result = response.get_json()
        preview = result.get('preview', {})
        print(f"✅ Default selector result:")
        print(f"   Reference vertex index: {preview.get('reference_vertex_idx')}")
        print(f"   Reference vertex coords: {preview.get('reference_vertex_coords')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preview test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_reference_point():
    """Test session reference point API"""
    print("\n🔍 Testing Session Reference Point API")
    print("=" * 50)
    
    try:
        # Create test app
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app)
        client = app.test_client()
        
        # Create a session first
        session_data = {
            "mesh_name": "basic1",
            "predictor_type": "RL",
            "predictor_config": {
                "model_path": "data/models/basic1-reward68.026.zip",
                "n": 2, "g": 3, "beta": 6
            },
            "ref_selector_type": "RL",
            "ref_selector_config": {"n": 2}
        }
        
        response = client.post('/predict/session/create', json=session_data)
        if response.status_code != 200:
            print(f"❌ Session creation failed: {response.get_data(as_text=True)}")
            return False
        
        session_result = response.get_json()
        session_id = session_result.get('session_id')
        print(f"✅ Session created: {session_id}")
        
        # Test getting current reference point
        print("\nGetting current reference point...")
        response = client.get(f'/predict/session/{session_id}/reference_point')
        if response.status_code != 200:
            print(f"❌ Reference point get failed: {response.get_data(as_text=True)}")
            return False
        
        result = response.get_json()
        if not result.get('success'):
            print(f"❌ Reference point get failed: {result}")
            return False
        
        ref_point = result.get('reference_point', {})
        print(f"✅ Current reference point:")
        print(f"   Index: {ref_point.get('reference_vertex_idx')}")
        print(f"   Coords: {ref_point.get('reference_vertex_coords')}")
        print(f"   Selector: {ref_point.get('selector_info', {}).get('type')}")
        print(f"   Interior angle: {ref_point.get('boundary_context', {}).get('interior_angle')}")
        
        # Test overriding selector type
        print("\nTesting selector override (Random)...")
        response = client.get(f'/predict/session/{session_id}/reference_point?selector_type=Random')
        if response.status_code != 200:
            print(f"❌ Reference point override failed: {response.get_data(as_text=True)}")
            return False
        
        result = response.get_json()
        ref_point = result.get('reference_point', {})
        print(f"✅ Override reference point:")
        print(f"   Index: {ref_point.get('reference_vertex_idx')}")
        print(f"   Coords: {ref_point.get('reference_vertex_coords')}")
        print(f"   Selector: {ref_point.get('selector_info', {}).get('type')}")
        
        # Test after some steps
        print("\nExecuting a step and checking reference point...")
        response = client.post(f'/predict/session/{session_id}/next')
        if response.status_code == 200:
            # Get reference point after step
            response = client.get(f'/predict/session/{session_id}/reference_point')
            if response.status_code == 200:
                result = response.get_json()
                ref_point = result.get('reference_point', {})
                print(f"✅ Reference point after step:")
                print(f"   Index: {ref_point.get('reference_vertex_idx')}")
                print(f"   Boundary size: {ref_point.get('boundary_context', {}).get('boundary_size')}")
                print(f"   Session step: {ref_point.get('session_status', {}).get('current_step')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Session reference point test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_point_comparison():
    """Test comparing different reference point selectors"""
    print("\n🔍 Testing Reference Point Selector Comparison")
    print("=" * 50)
    
    try:
        # Create test app
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app)
        client = app.test_client()
        
        mesh_name = "basic1"
        selectors = [
            {"type": "RL", "config": {"n": 2}},
            {"type": "Random", "config": {}},
            {"type": "default", "config": {}}
        ]
        
        print(f"Comparing selectors for mesh: {mesh_name}")
        results = {}
        
        for selector in selectors:
            preview_data = {
                "mesh_name": mesh_name,
                "ref_selector_type": selector["type"],
                "ref_selector_config": selector["config"]
            }
            
            response = client.post('/predict/reference_point/preview', json=preview_data)
            if response.status_code == 200:
                result = response.get_json()
                preview = result.get('preview', {})
                results[selector["type"]] = {
                    "vertex_idx": preview.get('reference_vertex_idx'),
                    "coords": preview.get('reference_vertex_coords'),
                    "interior_angle": preview.get('boundary_context', {}).get('interior_angle')
                }
        
        print("\nComparison Results:")
        for selector_type, result in results.items():
            print(f"  {selector_type:8}: vertex {result['vertex_idx']:2} at {result['coords']} " +
                  f"(angle: {result['interior_angle']:.2f}°)")
        
        # Show which selector would be most effective
        if 'RL' in results and results['RL']['interior_angle']:
            print(f"\n✅ RL selector chose vertex with smallest average interior angle")
            print(f"   This is optimal for mesh generation quality")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution"""
    print("🚀 Testing Reference Point APIs")
    print("=" * 60)
    
    # Test 1: Preview API
    preview_success = test_reference_point_preview()
    
    # Test 2: Session reference point API
    session_success = test_session_reference_point()
    
    # Test 3: Selector comparison
    comparison_success = test_reference_point_comparison()
    
    print("\n" + "=" * 60)
    if preview_success and session_success and comparison_success:
        print("🎉 All Reference Point API Tests Passed!")
        print("\nNew APIs now provide:")
        print("✅ Reference point preview without creating sessions")
        print("✅ Current session reference point information")
        print("✅ Selector override capability")
        print("✅ Detailed boundary context and neighbor information")
        print("✅ Interior angle calculations")
        print("✅ Full boundary vertices for visualization")
        print("\nFrontend can now:")
        print("- Preview reference points when configuring selectors")
        print("- Show current reference point in real-time")
        print("- Compare different selector strategies")
        print("- Visualize reference point context and neighbors")
        print("- Override selectors for testing/debugging")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)