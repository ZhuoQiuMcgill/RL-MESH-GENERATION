import os
import traceback
import copy
from flask import Blueprint, jsonify, request, current_app
from src.mesh_generator.mesh_generator import MeshGenerator
from src.mesh_generator.rl_predictor import RLPredictor
from src.geometry import AVALIABLE_REFERENCE_POINT_SELECTORS as AVAILABLE_REF_SELECTORS
from src.utils import MeshImporter

predict_bp = Blueprint("predict", __name__, url_prefix="/predict")

# Global state for prediction session
prediction_sessions = {}

# Available components
AVAILABLE_PREDICTORS = {
    "RL": {
        "name": "RL",
        "class": RLPredictor,
        "description": "Reinforcement Learning predictor using trained SAC model",
        "parameters": ["n", "g", "beta"]
    }
}




def get_available_models():
    """Get list of available trained models from data/models directory"""
    models_dir = "data/models"
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.zip'):
                model_path = os.path.join(models_dir, file)
                model_info = {
                    "name": file,
                    "path": model_path,
                    "size": os.path.getsize(model_path),
                    "description": f"Trained SAC model: {file}"
                }
                models.append(model_info)
    
    return models


@predict_bp.route("/components", methods=["GET"])
def list_components():
    """
    List all available components for prediction
    
    Returns:
        JSON response containing available predictors, reference selectors, 
        initial meshes, and trained models
    """
    try:
        # Get available meshes
        importer = MeshImporter()
        meshes = importer.list_available_meshes("mesh")
        
        # Get available models
        models = get_available_models()
        
        # Create JSON-serializable versions of the component data
        serializable_predictors = {}
        for name, info in AVAILABLE_PREDICTORS.items():
            serializable_predictors[name] = {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            }
        
        serializable_ref_selectors = {}
        for name, selector in AVAILABLE_REF_SELECTORS.items():
            serializable_ref_selectors[name] = {
                "name": name,
                "description": selector.__doc__ or "No description available",
                "parameters": getattr(selector, 'parameters', [])
            }
        
        return jsonify({
            "predictors": serializable_predictors,
            "reference_selectors": serializable_ref_selectors,
            "initial_meshes": meshes,
            "trained_models": models,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in list_components: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to list components: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/create", methods=["POST"])
def create_session():
    """
    Create a new prediction session
    
    Request JSON:
        {
            "mesh_name": "basic1.txt",
            "predictor_type": "RL",
            "predictor_config": {
                "model_path": "data/models/basic1-reward68.026.zip",
                "n": 2, "g": 3, "beta": 6
            },
            "ref_selector_type": "RL",
            "ref_selector_config": {"n": 2}
        }
    
    Returns:
        JSON response with session_id
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["mesh_name", "predictor_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "success": False
                }), 400
        
        mesh_name = data["mesh_name"]
        predictor_type = data["predictor_type"]
        predictor_config = data.get("predictor_config", {})
        ref_selector_type = data.get("ref_selector_type", "default")
        ref_selector_config = data.get("ref_selector_config", {})
        
        # Load initial mesh
        importer = MeshImporter()
        boundary = importer.load_boundary_by_name(mesh_name, "mesh")
        boundary_vertices = boundary.get_vertices()
        
        # Create MeshGenerator
        generator = MeshGenerator(boundary_vertices)
        
        # Initialize predictor
        if predictor_type not in AVAILABLE_PREDICTORS:
            return jsonify({
                "error": f"Unknown predictor type: {predictor_type}",
                "success": False
            }), 400
        
        predictor_class = AVAILABLE_PREDICTORS[predictor_type]["class"]
        if predictor_type == "RL":
            n = predictor_config.get("n", 2)
            g = predictor_config.get("g", 3)
            beta = predictor_config.get("beta", 6)
            predictor = predictor_class(n=n, g=g, beta=beta)
            
            # Load model if specified
            model_path = predictor_config.get("model_path")
            if model_path:
                predictor.init_agent(agent_path=model_path)
            else:
                return jsonify({
                    "error": "RL predictor requires model_path in predictor_config",
                    "success": False
                }), 400
        else:
            predictor = predictor_class()
        
        # Set predictor
        generator.set_predictor(predictor)
        generator.update_activated_predictor(predictor_type)
        
        # Initialize reference selector
        if ref_selector_type != "default":
            if ref_selector_type not in AVAILABLE_REF_SELECTORS:
                return jsonify({
                    "error": f"Unknown reference selector type: {ref_selector_type}",
                    "success": False
                }), 400
            
            # Clone the base selector to prevent modifying the shared instance
            base_selector = AVAILABLE_REF_SELECTORS[ref_selector_type]
            ref_selector = copy.deepcopy(base_selector)

            # Update the cloned selector with new config
            if hasattr(ref_selector, 'parameters'):
                for param in ref_selector.parameters:
                    if param in ref_selector_config:
                        setattr(ref_selector, param, ref_selector_config[param])
            
            generator.set_ref_selector(ref_selector)
        
        # Generate session ID
        session_id = f"session_{len(prediction_sessions)}_{hash(str(data))}"
        
        # Store session
        prediction_sessions[session_id] = {
            "generator": generator,
            "config": {
                "mesh_name": mesh_name,
                "predictor_type": predictor_type,
                "predictor_config": predictor_config,
                "ref_selector_type": ref_selector_type,
                "ref_selector_config": ref_selector_config
            },
            "history": [],
            "current_ref_point_idx": None  # Initialize with no selected reference point
        }
        
        # Get initial status
        status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "initial_status": status,
            "config": prediction_sessions[session_id]["config"],
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in create_session: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to create session: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/status", methods=["GET"])
def get_session_status(session_id):
    """
    Get current status of prediction session
    
    Returns:
        JSON response with current generator status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        generator = session["generator"]
        status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "status": status,
            "config": session["config"],
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in get_session_status: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get session status: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/config", methods=["PUT"])
def update_session_config(session_id):
    """
    Update session configuration (predictor or reference selector)
    
    Request JSON:
        {
            "predictor_type": "RL",
            "predictor_config": {...},
            "ref_selector_type": "RL",
            "ref_selector_config": {...}
        }
    
    Returns:
        JSON response with updated status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        data = request.get_json()
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        # Update predictor if specified
        if "predictor_type" in data:
            predictor_type = data["predictor_type"]
            predictor_config = data.get("predictor_config", {})
            
            if predictor_type not in AVAILABLE_PREDICTORS:
                return jsonify({
                    "error": f"Unknown predictor type: {predictor_type}",
                    "success": False
                }), 400
            
            predictor_class = AVAILABLE_PREDICTORS[predictor_type]["class"]
            if predictor_type == "RL":
                n = predictor_config.get("n", 2)
                g = predictor_config.get("g", 3)
                beta = predictor_config.get("beta", 6)
                predictor = predictor_class(n=n, g=g, beta=beta)
                
                model_path = predictor_config.get("model_path")
                if model_path:
                    predictor.init_agent(agent_path=model_path)
                else:
                    return jsonify({
                        "error": "RL predictor requires model_path in predictor_config",
                        "success": False
                    }), 400
            else:
                predictor = predictor_class()
            
            generator.set_predictor(predictor)
            generator.update_activated_predictor(predictor_type)
            
            # Update session config
            session["config"]["predictor_type"] = predictor_type
            session["config"]["predictor_config"] = predictor_config
        
        # Update reference selector if specified
        if "ref_selector_type" in data:
            ref_selector_type = data["ref_selector_type"]
            ref_selector_config = data.get("ref_selector_config", {})
            
            if ref_selector_type == "default":
                generator.set_ref_selector(None)
            else:
                if ref_selector_type not in AVAILABLE_REF_SELECTORS:
                    return jsonify({
                        "error": f"Unknown reference selector type: {ref_selector_type}",
                        "success": False
                    }), 400
                
                # Clone the base selector to prevent modifying the shared instance
                base_selector = AVAILABLE_REF_SELECTORS[ref_selector_type]
                ref_selector = copy.deepcopy(base_selector)
                
                # Update the cloned selector with new config
                if hasattr(ref_selector, 'parameters'):
                    for param in ref_selector.parameters:
                        if param in ref_selector_config:
                            setattr(ref_selector, param, ref_selector_config[param])
                
                generator.set_ref_selector(ref_selector)
            
            # Update session config
            session["config"]["ref_selector_type"] = ref_selector_type
            session["config"]["ref_selector_config"] = ref_selector_config
        
        # Get updated status and new reference point in one go
        status = generator.get_status()
        
        # Manually get the new reference point to include in the response
        try:
            ref_point_response = get_reference_point(session_id)
            if ref_point_response.status_code == 200:
                status['reference_point'] = ref_point_response.get_json().get('reference_point')
        except Exception as e:
            current_app.logger.warning(f"Could not fetch reference point during config update: {e}")
            status['reference_point'] = None

        return jsonify({
            "session_id": session_id,
            "status": status,
            "config": session["config"],
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in update_session_config: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to update session config: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/next", methods=["POST"])
def next_step(session_id):
    """
    Execute next prediction step
    
    Returns:
        JSON response with step result and updated status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        # Use the stored reference point for this step
        ref_idx_to_use = session.get("current_ref_point_idx")

        # Execute step with the specific reference point
        step_result = generator.step(ref_idx=ref_idx_to_use)

        # Clear the used reference point to ensure a new one is selected for the next step
        session["current_ref_point_idx"] = None
        
        # Create JSON-serializable version of step result (remove command object but keep action info)
        serializable_step_result = {
            "success": step_result.get("success"),
            "element": step_result.get("element"),
            "message": step_result.get("message"),
            "action_info": step_result.get("action_info")  # This is already JSON-serializable
        }
        
        # Record step in history (also remove command from history)
        session["history"].append({
            "action": "next",
            "result": serializable_step_result,
            "timestamp": __import__("time").time()
        })
        
        # Get updated status
        status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "step_result": serializable_step_result,
            "status": status,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in next_step: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to execute next step: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/prev", methods=["POST"])
def prev_step(session_id):
    """
    Undo previous step (go back to previous state)
    
    Returns:
        JSON response with undo result and updated status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        # Execute undo
        undo_result = generator.undo()
        
        # Clear the locked reference point, as the state has changed
        session["current_ref_point_idx"] = None

        # Record undo in history
        session["history"].append({
            "action": "prev",
            "result": undo_result,
            "timestamp": __import__("time").time()
        })
        
        # Get updated status
        status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "undo_result": undo_result,
            "status": status,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in prev_step: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to undo step: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/process_all", methods=["POST"])
def process_all(session_id):
    """
    Execute all steps until invalid action or completion
    
    Query parameters:
        max_steps: Maximum number of steps to execute (default: 100)
    
    Returns:
        JSON response with process results and final status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        max_steps = int(request.args.get("max_steps", 100))
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        results = []
        step_count = 0
        
        while step_count < max_steps:
            # Check if completed
            if generator.get_status()["is_completed"]:
                break
            
            # Execute step
            step_result = generator.step()
            
            # Create JSON-serializable version (remove command object but keep action info)
            serializable_result = {
                "success": step_result.get("success"),
                "element": step_result.get("element"),
                "message": step_result.get("message"),
                "action_info": step_result.get("action_info")  # This is already JSON-serializable
            }
            results.append(serializable_result)
            step_count += 1
            
            # Break if step failed (invalid action)
            if not step_result["success"]:
                break
        
        # Record process_all in history
        session["history"].append({
            "action": "process_all",
            "result": {
                "steps_executed": step_count,
                "total_results": len(results),
                "final_result": results[-1] if results else None
            },
            "timestamp": __import__("time").time()
        })
        
        # Get final status
        final_status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "steps_executed": step_count,
            "results": results,
            "final_status": final_status,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in process_all: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to process all steps: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/reset", methods=["POST"])
def reset_session(session_id):
    """
    Reset session to initial state
    
    Returns:
        JSON response with reset result and initial status
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        # Execute reset
        reset_result = generator.reset()
        
        # Clear history and locked reference point
        session["history"] = []
        session["current_ref_point_idx"] = None
        
        # Get reset status
        status = generator.get_status()
        
        return jsonify({
            "session_id": session_id,
            "reset_result": reset_result,
            "status": status,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in reset_session: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to reset session: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/history", methods=["GET"])
def get_session_history(session_id):
    """
    Get session history
    
    Returns:
        JSON response with session history
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        
        return jsonify({
            "session_id": session_id,
            "history": session["history"],
            "total_actions": len(session["history"]),
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in get_session_history: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get session history: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """
    Delete prediction session
    
    Returns:
        JSON response confirming deletion
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        del prediction_sessions[session_id]
        
        return jsonify({
            "message": f"Session {session_id} deleted successfully",
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in delete_session: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to delete session: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """
    List all active prediction sessions
    
    Returns:
        JSON response with session list
    """
    try:
        sessions_info = {}
        for session_id, session in prediction_sessions.items():
            generator = session["generator"]
            status = generator.get_status()
            sessions_info[session_id] = {
                "config": session["config"],
                "status": status,
                "history_length": len(session["history"])
            }
        
        return jsonify({
            "sessions": sessions_info,
            "total_sessions": len(sessions_info),
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in list_sessions: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to list sessions: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/session/<session_id>/reference_point", methods=["GET"])
def get_reference_point(session_id):
    """
    Get current reference point information based on session's reference selector
    
    Query Parameters:
        selector_type: Override selector type (optional)
        selector_config: Override selector config as JSON string (optional)
    
    Returns:
        JSON response with reference point details
    """
    try:
        if session_id not in prediction_sessions:
            return jsonify({
                "error": f"Session not found: {session_id}",
                "success": False
            }), 404
        
        session = prediction_sessions[session_id]
        generator = session["generator"]
        
        # Get current boundary
        current_boundary = generator.boundary
        
        # Get selector configuration from query params or session config
        selector_type = request.args.get('selector_type', session["config"]["ref_selector_type"])
        selector_config_str = request.args.get('selector_config')
        
        if selector_config_str:
            import json
            selector_config = json.loads(selector_config_str)
        else:
            selector_config = session["config"]["ref_selector_config"]
        
        # Get reference point using specified selector
        if selector_type == "default":
            # Default selector uses RL method with n=1
            ref_vertex_idx = current_boundary.get_ref_vertex(1)
            selector_info = {
                "type": "default",
                "method": "boundary default (RL with n=1)",
                "config": {"n": 1}
            }
        else:
            if selector_type not in AVAILABLE_REF_SELECTORS:
                return jsonify({
                    "error": f"Unknown reference selector type: {selector_type}",
                    "success": False
                }), 400
            
            import copy

# ... (other imports)

# ... (inside get_reference_point function)
            
            # Clone the base selector to avoid modifying the shared instance
            base_selector = AVAILABLE_REF_SELECTORS[selector_type]
            ref_selector = copy.deepcopy(base_selector)
            
            # Update the cloned selector with new config
            if hasattr(ref_selector, 'parameters'):
                for param in ref_selector.parameters:
                    if param in selector_config:
                        setattr(ref_selector, param, selector_config[param])
            
            ref_vertex_idx = ref_selector.select_reference_point(current_boundary, **selector_config)
            
            selector_info = {
                "type": selector_type,
                "method": ref_selector.__doc__ or "No description available",
                "config": {param: getattr(ref_selector, param) for param in getattr(ref_selector, 'parameters', [])}
            }
        
        # Store the selected reference point index in the session state
        session["current_ref_point_idx"] = ref_vertex_idx

        # Get vertex coordinates and additional info
        ref_vertex = current_boundary.get_vertex_by_index(ref_vertex_idx)
        
        # Calculate additional reference point details
        boundary_size = current_boundary.size()
        
        # Get neighbor information
        left_neighbor_idx = (ref_vertex_idx - 1) % boundary_size
        right_neighbor_idx = (ref_vertex_idx + 1) % boundary_size
        left_neighbor = current_boundary.get_vertex_by_index(left_neighbor_idx)
        right_neighbor = current_boundary.get_vertex_by_index(right_neighbor_idx)
        
        # Determine the value of 'n' from the request's configuration
        n_val = selector_config.get("n", 1)

        # Calculate interior angle if possible
        interior_angle = None
        try:
            from src.utils import get_avg_interior_angle
            interior_angle = get_avg_interior_angle(current_boundary, ref_vertex_idx, n_val)
        except Exception:
            pass  # Interior angle calculation failed, leave as None

        # Get local environment for rendering
        local_env_vertices = current_boundary.get_neighbors(ref_vertex_idx, n_val)
        
        reference_point_info = {
            "reference_vertex_idx": ref_vertex_idx,
            "reference_vertex_coords": list(ref_vertex),
            "selector_info": selector_info,
            "boundary_context": {
                "boundary_size": boundary_size,
                "left_neighbor_idx": left_neighbor_idx,
                "left_neighbor_coords": list(left_neighbor),
                "right_neighbor_idx": right_neighbor_idx,
                "right_neighbor_coords": list(right_neighbor),
                "interior_angle": interior_angle,
                "local_env": local_env_vertices,
                "n": n_val
            },
            "session_status": generator.get_status()
        }
        
        return jsonify({
            "session_id": session_id,
            "reference_point": reference_point_info,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in get_reference_point: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get reference point: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/reference_point/preview", methods=["POST"])
def preview_reference_point():
    """
    Preview reference point selection for a given mesh and selector configuration
    without creating a session
    
    Request JSON:
        {
            "mesh_name": "basic1.txt",
            "ref_selector_type": "RL",
            "ref_selector_config": {"n": 2}
        }
    
    Returns:
        JSON response with reference point preview
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["mesh_name", "ref_selector_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "success": False
                }), 400
        
        mesh_name = data["mesh_name"]
        selector_type = data["ref_selector_type"]
        selector_config = data.get("ref_selector_config", {})
        
        # Load mesh
        importer = MeshImporter()
        boundary = importer.load_boundary_by_name(mesh_name, "mesh")
        
        # Get reference point using specified selector
        if selector_type == "default":
            # Default selector uses RL method with n=1
            ref_vertex_idx = boundary.get_ref_vertex(n=1)
            selector_info = {
                "type": "default",
                "method": "boundary default (RL with n=1)",
                "config": {"n": 1}
            }
        else:
            if selector_type not in AVAILABLE_REF_SELECTORS:
                return jsonify({
                    "error": f"Unknown reference selector type: {selector_type}",
                    "success": False
                }), 400
            
            # Clone the base selector to avoid modifying the shared instance
            base_selector = AVAILABLE_REF_SELECTORS[selector_type]
            ref_selector = copy.deepcopy(base_selector)
            
            # Update the cloned selector with new config
            if hasattr(ref_selector, 'parameters'):
                for param in ref_selector.parameters:
                    if param in selector_config:
                        setattr(ref_selector, param, selector_config[param])
            
            ref_vertex_idx = ref_selector.select_reference_point(boundary, **selector_config)
            
            selector_info = {
                "type": selector_type,
                "method": ref_selector.__doc__ or "No description available",
                "config": {param: getattr(ref_selector, param) for param in getattr(ref_selector, 'parameters', [])}
            }
        
        # Get vertex coordinates and mesh info
        ref_vertex = boundary.get_vertex_by_index(ref_vertex_idx)
        boundary_vertices = boundary.get_vertices()
        boundary_size = boundary.size()
        
        # Get neighbor information
        left_neighbor_idx = (ref_vertex_idx - 1) % boundary_size
        right_neighbor_idx = (ref_vertex_idx + 1) % boundary_size
        left_neighbor = boundary.get_vertex_by_index(left_neighbor_idx)
        right_neighbor = boundary.get_vertex_by_index(right_neighbor_idx)
        
        # Determine the value of 'n' from the request's configuration
        n_val = selector_config.get("n", 1)

        # Calculate interior angle if possible
        interior_angle = None
        try:
            from src.utils import get_avg_interior_angle
            interior_angle = get_avg_interior_angle(boundary, ref_vertex_idx, n_val)
        except Exception:
            pass

        # Get local environment for rendering
        local_env_vertices = boundary.get_neighbors(ref_vertex_idx, n_val)
        
        preview_info = {
            "mesh_name": mesh_name,
            "reference_vertex_idx": ref_vertex_idx,
            "reference_vertex_coords": list(ref_vertex),
            "selector_info": selector_info,
            "boundary_context": {
                "boundary_size": boundary_size,
                "total_vertices": len(boundary_vertices),
                "left_neighbor_idx": left_neighbor_idx,
                "left_neighbor_coords": list(left_neighbor),
                "right_neighbor_idx": right_neighbor_idx,
                "right_neighbor_coords": list(right_neighbor),
                "interior_angle": interior_angle,
                "local_env": local_env_vertices,
                "n": n_val
            },
            "boundary_vertices": boundary_vertices  # Full boundary for visualization
        }
        
        return jsonify({
            "preview": preview_info,
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Exception in preview_reference_point: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to preview reference point: {str(e)}",
            "success": False
        }), 500


@predict_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    
    Returns:
        JSON response indicating service status
    """
    try:
        return jsonify({
            "status": "healthy",
            "service": "predict-api",
            "active_sessions": len(prediction_sessions),
            "timestamp": __import__("time").time()
        })
    except Exception as e:
        current_app.logger.error(f"Exception in health_check: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "unhealthy",  
            "service": "predict-api",
            "error": str(e),
            "timestamp": __import__("time").time()
        }), 500