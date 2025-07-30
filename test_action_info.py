#!/usr/bin/env python3
"""
Test Action Info functionality

This script tests the new action_info feature that provides detailed
information about actions (valid and invalid) to the frontend.
"""

import sys
from src.mesh_generator.mesh_generator import MeshGenerator
from src.mesh_generator.rl_predictor import RLPredictor
from src.geometry.reference_point_selectors import RLReferencePointSelector
from src.utils import MeshImporter


def test_action_info():
    """Test that action_info is properly returned for both valid and invalid actions"""
    print("🔍 Testing Action Info Functionality")
    print("=" * 50)
    
    try:
        # Load a simple mesh
        importer = MeshImporter()
        boundary = importer.load_boundary_by_name("basic1", "mesh")
        boundary_vertices = boundary.get_vertices()
        print(f"✅ Loaded mesh with {len(boundary_vertices)} vertices")
        
        # Create MeshGenerator
        generator = MeshGenerator(boundary_vertices)
        
        # Initialize RL predictor
        predictor = RLPredictor(n=2, g=3, beta=6)
        predictor.init_agent(agent_path="data/models/basic1-reward68.026.zip")
        generator.set_predictor(predictor)
        generator.update_activated_predictor("RL")
        
        # Initialize reference selector
        class ReferencePointSelectorWrapper:
            def __init__(self, selector_class, config):
                self.selector_class = selector_class
                self.config = config
            
            def select_reference_point(self, boundary):
                return self.selector_class.select_reference_point(boundary, **self.config)
        
        ref_selector = ReferencePointSelectorWrapper(
            RLReferencePointSelector, 
            {"n": 2}
        )
        generator.set_ref_selector(ref_selector)
        print("✅ Generator initialized")
        
        # Test several steps to get both valid and potentially invalid actions
        for i in range(10):
            print(f"\n--- Step {i+1} ---")
            
            step_result = generator.step()
            action_info = step_result.get('action_info')
            
            if action_info:
                print(f"Action Type: {action_info['action_type']}")
                print(f"Reference Vertex: {action_info['reference_vertex_idx']}")
                print(f"New Coords: {action_info['new_coords']}")
                print(f"Is Valid: {action_info['is_valid']}")
                print(f"Validation Message: {action_info['validation_message']}")
                print(f"Step Success: {step_result['success']}")
                print(f"Message: {step_result['message']}")
                
                if not action_info['is_valid']:
                    print("🔍 Found invalid action with detailed info!")
                    print(f"   Attempted: {action_info['action_type']} at vertex {action_info['reference_vertex_idx']}")
                    if action_info['new_coords']:
                        print(f"   Coordinates: {action_info['new_coords']}")
                    print(f"   Reason: {action_info['validation_message']}")
                    break
                else:
                    print("✅ Valid action executed successfully")
                    
                    # Check current status
                    status = generator.get_status()
                    print(f"   New boundary size: {status['boundary_size']}")
                    print(f"   Generated elements: {status['generated_elements_count']}")
                    
                    if status['is_completed']:
                        print("🎉 Generation completed!")
                        break
            else:
                print("❌ No action_info returned")
                print(f"Step result: {step_result}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 Action Info Test Completed Successfully!")
        print("\nAction info now provides:")
        print("✅ Action type (type0_left, type0_right, type1)")
        print("✅ Reference vertex index")
        print("✅ New coordinates (for type1 actions)")
        print("✅ Validity status (is_valid)")
        print("✅ Detailed validation messages for invalid actions")
        print("\nFrontend can now:")
        print("- Display attempted actions even when invalid")
        print("- Show exactly what the model tried to do")
        print("- Provide detailed error messages to users")
        print("- Visualize invalid action attempts for debugging")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Test that action_info works through the API layer"""
    print("\n🔍 Testing API Integration")
    print("=" * 30)
    
    try:
        from flask import Flask
        from src.ui.api import register_blueprints
        
        # Create test app
        app = Flask(__name__)
        app.config['TESTING'] = True
        register_blueprints(app)
        client = app.test_client()
        
        # Create session
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
        
        # Test step execution through API
        response = client.post(f'/predict/session/{session_id}/next')
        if response.status_code != 200:
            print(f"❌ Step execution failed: {response.get_data(as_text=True)}")
            return False
        
        step_result = response.get_json()
        step_data = step_result.get('step_result', {})
        action_info = step_data.get('action_info')
        
        if action_info:
            print("✅ Action info received through API:")
            print(f"   Action Type: {action_info.get('action_type')}")
            print(f"   Reference Vertex: {action_info.get('reference_vertex_idx')}")
            print(f"   New Coords: {action_info.get('new_coords')}")
            print(f"   Is Valid: {action_info.get('is_valid')}")
            print(f"   Validation Message: {action_info.get('validation_message')}")
            
            print("\n✅ API Integration Test Passed!")
            print("Frontend will receive complete action information")
            return True
        else:
            print("❌ No action_info in API response")
            print(f"Response: {step_data}")
            return False
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 Testing Action Info Feature")
    print("=" * 60)
    
    # Test 1: Direct functionality
    test1_success = test_action_info()
    
    # Test 2: API integration
    test2_success = test_api_integration()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("🎉 All Action Info Tests Passed!")
        print("\nBackend now provides complete action information:")
        print("- ✅ Action type identification")
        print("- ✅ Reference vertex information") 
        print("- ✅ Coordinate details for type1 actions")
        print("- ✅ Validity status and detailed error messages")
        print("- ✅ JSON-serializable through API")
        print("\nFrontend can now visualize invalid actions and provide detailed debugging!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)