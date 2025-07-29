"""
Action API Blueprint
Handles action testing and validation for RL mesh generation
"""

from flask import Blueprint, request, jsonify
import os
import traceback
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.geometry import Boundary
from src.rl.action.action_manager import ActionManager
from src.rl.environment import MeshEnv
from src.utils import normalize_coordinates, euclidean_distance

action_bp = Blueprint('action', __name__, url_prefix='/action')


class ActionTesterService:
    """Service class for handling action testing operations"""

    def __init__(self):
        self.mesh_cache = {}  # Cache for loaded meshes
        self.boundary_cache = {}  # Cache for boundaries
        self.action_manager = None
        self.environment = None

    def _get_mesh_file_path(self, mesh_name: str, subfolder: str = 'mesh') -> str:
        """Get the full path to a mesh file"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        mesh_dir = os.path.join(base_dir, 'data', subfolder)
        # Add .txt extension if not present
        if not mesh_name.endswith('.txt'):
            mesh_name = mesh_name + '.txt'
        return os.path.join(mesh_dir, mesh_name)

    def _load_boundary_from_mesh(self, mesh_name: str) -> Optional[Boundary]:
        """Load boundary from mesh file"""
        if mesh_name in self.boundary_cache:
            return self.boundary_cache[mesh_name]

        try:
            mesh_path = self._get_mesh_file_path(mesh_name)
            if not os.path.exists(mesh_path):
                return None

            # Load mesh data
            vertices = []
            # Try to load as simple text format (x y coordinates)
            with open(mesh_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                vertices.append((x, y))
                            except ValueError:
                                continue

            if len(vertices) < 3:
                return None

            # Create boundary
            boundary = Boundary(vertices)
            self.boundary_cache[mesh_name] = boundary
            return boundary

        except Exception as e:
            print(f"Error loading boundary from mesh {mesh_name}: {e}")
            return None

    def find_reference_point(self, mesh_name: str) -> Dict[str, Any]:
        """Find the reference point for a given mesh"""
        try:
            boundary = self._load_boundary_from_mesh(mesh_name)
            if boundary is None:
                return {
                    'success': False,
                    'error': f'Could not load boundary from mesh: {mesh_name}'
                }

            # Get reference vertex index using boundary's method
            ref_index = boundary.get_ref_vertex()
            ref_vertex = boundary.get_vertex_by_index(ref_index)
            ref_angle = boundary.get_avg_interior_angle(ref_index)

            # Get neighbor vertices for visualization
            neighbor_vertices = boundary.get_neighbors(ref_index, n=2, get_type="exclude ref")
            neighbor_vertices.insert(2, ref_vertex)  # Insert ref point in the middle

            return {
                'success': True,
                'reference_point': {
                    'index': ref_index,
                    'coordinates': list(ref_vertex),
                    'interior_angle': ref_angle,
                    'neighbor_vertices': neighbor_vertices
                }
            }

        except Exception as e:
            print(f"Error finding reference point: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def execute_action(self, mesh_name: str, action_type: str, reference_point_index: int,
                       clicked_point: Optional[List[float]] = None) -> Dict[str, Any]:
        """Execute and validate an action"""
        try:
            boundary = self._load_boundary_from_mesh(mesh_name)
            if boundary is None:
                return {
                    'success': False,
                    'error': f'Could not load boundary from mesh: {mesh_name}'
                }

            # Initialize action manager if not already done
            if self.action_manager is None:
                action_config = {
                    "enabled": ["type0_left", "type0_right", "type1"],
                    "auto_remap": True
                }
                self.action_manager = ActionManager(
                    alpha=2, n=2, max_steps=1000, action_config=action_config
                )

            # Create appropriate action vector based on type
            if action_type == "type0_left":
                # type0_left corresponds to type_logit >= 0.5
                action_vector = np.array([0.7, 0.0, 0.0], dtype=np.float32)
            elif action_type == "type0_right":
                # type0_right corresponds to type_logit <= -0.5
                action_vector = np.array([-0.7, 0.0, 0.0], dtype=np.float32)
            elif action_type == "type1":
                if clicked_point is None:
                    return {
                        'success': False,
                        'error': 'Clicked point required for type1 action'
                    }

                # Convert clicked point to normalized polar coordinates
                ref_vertex = boundary.get_vertex_by_index(reference_point_index)
                right_neighbor = boundary.get_vertex_by_index(reference_point_index - 1)
                base_len = boundary.get_avg_neighbor_length(reference_point_index, 2)
                scale_factor = 1.0 / base_len if base_len > 0 else 1.0

                # Normalize the clicked point
                normalized_coords = normalize_coordinates(
                    [clicked_point], ref_vertex, right_neighbor, scale_factor
                )[0]

                r, theta = normalized_coords
                # type1 corresponds to -0.5 < type_logit < 0.5
                action_vector = np.array([0.0, r, theta], dtype=np.float32)
            else:
                return {
                    'success': False,
                    'error': f'Unknown action type: {action_type}'
                }

            # Decode and validate action
            action_name, action_instance, new_coords, ref_idx = self.action_manager.decode_action(
                action_vector, boundary, reference_point_index
            )

            # Check if action is valid
            is_valid = self.action_manager.is_valid(
                boundary, ref_idx, action_instance, action_name, new_coords
            )

            # Get the actual element that would be generated (for visualization, even if invalid)
            generated_element = None
            try:
                if action_name == "type1":
                    # For type1, we need the clicked coordinates
                    generated_element = action_instance.get_element(boundary, ref_idx, new_coords[0])
                else:
                    # For type0 actions, no additional coordinates needed
                    generated_element = action_instance.get_element(boundary, ref_idx)
            except Exception as e:
                print(f"Error getting element: {e}")
                generated_element = None

            result = {
                'valid': is_valid,
                'action_name': action_name,
                'decoded_coords': new_coords,
                'generated_element': generated_element
            }

            # For type1 actions, include the normalized polar coordinates
            if action_type == "type1" and clicked_point is not None:
                result['polar_coordinates'] = {
                    'r': float(r),
                    'theta': float(theta)
                }

            return {
                'success': True,
                'result': result
            }

        except Exception as e:
            print(f"Error executing action: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# Create service instance
action_service = ActionTesterService()


@action_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'action_tester'
    })


@action_bp.route('/find-ref-point/<mesh_name>', methods=['GET'])
def find_reference_point(mesh_name: str):
    """Find reference point for a mesh"""
    try:
        result = action_service.find_reference_point(mesh_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@action_bp.route('/execute', methods=['POST'])
def execute_action():
    """Execute and validate an action"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Validate required fields
        required_fields = ['mesh_name', 'action_type', 'reference_point_index']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        result = action_service.execute_action(
            mesh_name=data['mesh_name'],
            action_type=data['action_type'],
            reference_point_index=data['reference_point_index'],
            clicked_point=data.get('clicked_point')
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@action_bp.route('/validate/<action_type>', methods=['POST'])
def validate_action(action_type: str):
    """Validate a specific action type without executing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # This could be extended to validate actions without full execution
        # For now, redirect to execute_action
        data['action_type'] = action_type
        return execute_action()

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@action_bp.route('/info', methods=['GET'])
def get_action_info():
    """Get information about available actions"""
    try:
        if action_service.action_manager is None:
            action_config = {
                "enabled": ["type0_left", "type0_right", "type1"],
                "auto_remap": True
            }
            action_service.action_manager = ActionManager(
                alpha=2, n=2, max_steps=1000, action_config=action_config
            )

        action_info = action_service.action_manager.get_action_info()

        return jsonify({
            'success': True,
            'action_info': action_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
