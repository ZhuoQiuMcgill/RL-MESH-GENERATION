import math
import numpy as np
from gymnasium import spaces
from .type0_left import ActionType0Left
from .type0_right import ActionType0Right
from .type1 import ActionType1
from src.utils import euclidean_distance, decode_coordinate


class ActionManager:
    _EPS = 0.0001

    def __init__(self, alpha=2, n=2, max_steps=1000, action_config=None):
        """
        Initialize ActionManager with configurable action types
        
        Args:
            alpha: Action space radius factor
            n: Reference vertex neighbors count
            max_steps: Maximum steps for environment
            action_config: Configuration dictionary for actions
        """
        # Initialize all action type instances
        self._all_action_types = {
            "type0_left": ActionType0Left(),
            "type0_right": ActionType0Right(),
            "type1": ActionType1(),
        }

        # Parse action configuration
        if action_config is None:
            action_config = {
                "enabled": ["type0_left", "type0_right", "type1"],
                "auto_remap": True
            }

        self.action_config = action_config
        self.enabled_actions = action_config.get("enabled", ["type0_left", "type0_right", "type1"])

        # Validate enabled actions
        for action_name in self.enabled_actions:
            if action_name not in self._all_action_types:
                raise ValueError(f"Unknown action type: {action_name}")

        # Create enabled action type instances mapping
        self.action_types = {name: self._all_action_types[name] for name in self.enabled_actions}

        # Create action type mapping for decoding
        self._setup_action_mapping()

        # Environment parameters
        self.alpha = alpha
        self.n = n
        self.max_steps = max_steps

        # Action: [type_logit, x_coord, y_coord]
        self.action_dim = 3

    def _setup_action_mapping(self):
        """Setup mapping between action logits and enabled action types"""
        num_enabled = len(self.enabled_actions)

        # Map from original action names to indices
        self._original_action_indices = {
            "type0_left": 0,
            "type0_right": 1,
            "type1": 2,
        }

        self.action_logit_mapping = {}
        for action_name in self.enabled_actions:
            original_idx = self._original_action_indices[action_name]
            self.action_logit_mapping[original_idx] = {
                'name': action_name,
                'instance': self.action_types[action_name]
            }

    def get_enabled_actions(self):
        """Get list of enabled action names"""
        return self.enabled_actions.copy()

    def get_action_descriptions(self):
        """Get descriptions of enabled actions"""
        descriptions = self.action_config.get("descriptions", {})
        return {name: descriptions.get(name, f"Action: {name}") for name in self.enabled_actions}

    def get_action_space(self):
        """
        Return the action space for the environment
        
        Returns:
            spaces.Box: Action space with shape (3,) and range [-1, 1]
        """
        # return spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        return spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)

    def decode_action(self, action, boundary, reference_vertex_idx):
        """
        Decode action from SAC returned action data and compute new coordinates
        
        Args:
            action: Raw action vector [type_logit, x_coord, y_coord]
            boundary: Current boundary object
            reference_vertex_idx: Reference vertex index
            
        Returns:
            tuple: (action_name, action_instance, new_coords, reference_vertex_idx)
        """
        type_logit = action[0]

        # Find the appropriate action based on configuration
        action_name = None
        action_instance = None

        if type_logit <= -0.5:
            target_idx = 0  # type0_right
        elif type_logit >= 0.5:
            target_idx = 1  # type0_left
        else:
            target_idx = 2  # type1

        if target_idx in self.action_logit_mapping:
            action_info = self.action_logit_mapping[target_idx]
            action_name = action_info['name']
            action_instance = action_info['instance']

        # Fallback to first enabled action if no match found
        if action_name is None:
            action_name = self.enabled_actions[0]
            action_instance = self.action_types[action_name]

        # Decode new vertex coordinates (if needed)
        new_coords = []
        if target_idx == 2:
            # Calculate action space radius
            ref_v = boundary.get_vertex_by_index(reference_vertex_idx)
            right_neighbor_v = boundary.get_vertex_by_index(reference_vertex_idx - 1)
            base_len = boundary.get_avg_neighbor_length(reference_vertex_idx, self.n)

            scale_factor = 1.0 / base_len if base_len > 0 else 1.0
            new_r, new_theta = action[1], action[2]

            x, y = decode_coordinate(ref_v, right_neighbor_v, scale_factor, new_r, new_theta)

            new_coords.append((x, y))
            eps_dist = boundary.get_avg_vertex_distance() * self._EPS

            if euclidean_distance((x, y), boundary.get_vertex_by_index(reference_vertex_idx + 2)) <= eps_dist:
                # type1 -> type0_right
                action_name = self.enabled_actions[1]
                action_instance = self.action_types[action_name]
            elif euclidean_distance((x, y), boundary.get_vertex_by_index(reference_vertex_idx - 2)) <= eps_dist:
                # type1 -> type0_left
                action_name = self.enabled_actions[0]
                action_instance = self.action_types[action_name]

        return action_name, action_instance, new_coords, reference_vertex_idx

    def is_valid(self, boundary, reference_vertex_idx, action_instance, action_name, new_coords):
        """
        Check if the decoded action is valid
        
        Args:
            boundary: Current boundary object
            reference_vertex_idx: Reference vertex index
            action_instance: Action type instance
            action_name: Name of the action type
            new_coords: List of new coordinates
            
        Returns:
            bool: Whether the action is valid
        """
        try:
            if action_name in ["type0_left", "type0_right"]:
                return action_instance.is_valid(boundary, reference_vertex_idx, alpha=self.alpha, n=self.n)
            elif action_name == "type1":
                return action_instance.is_valid(boundary, reference_vertex_idx, new_coords[0], alpha=self, n=self.n)
            else:
                return False
        except Exception:
            return False

    def get_element_quality(self, boundary, reference_vertex_idx, action_instance, action_name, new_coords):
        """
        Get element quality for the given action
        
        Args:
            boundary: Current boundary object
            reference_vertex_idx: Reference vertex index
            action_instance: Action type instance
            action_name: Name of the action type
            new_coords: List of new coordinates
            
        Returns:
            float: Element quality reward
        """
        if action_name in ["type0_left", "type0_right"]:
            return action_instance.get_element_quality(boundary, reference_vertex_idx)
        elif action_name == "type1":
            return action_instance.get_element_quality(boundary, reference_vertex_idx, new_coords[0])
        else:
            return 0.0

    def get_boundary_quality(self, boundary, reference_vertex_idx, action_instance, action_name, new_coords, M_angle):
        """
        Get boundary quality for the given action
        
        Args:
            boundary: Current boundary object
            reference_vertex_idx: Reference vertex index
            action_instance: Action type instance
            action_name: Name of the action type
            new_coords: List of new coordinates
            M_angle: Angle threshold parameter
            
        Returns:
            float: Boundary quality reward
        """
        if action_name in ["type0_left", "type0_right"]:
            return action_instance.get_boundary_quality(boundary, reference_vertex_idx, M_angle=M_angle)
        elif action_name == "type1":
            return action_instance.get_boundary_quality(boundary, reference_vertex_idx, new_coords[0], M_angle=M_angle)
        else:
            return -1

    def execute_action(self, mesh, boundary, reference_vertex_idx, action_instance, action_name, new_coords):
        """
        Execute the given action and return the generated element
        
        Args:
            mesh: Mesh object to modify
            boundary: Boundary object to modify
            reference_vertex_idx: Reference vertex index
            action_instance: Action type instance
            action_name: Name of the action type
            new_coords: List of new coordinates
            
        Returns:
            list or None: Generated element (quadrilateral vertices) or None if failed
        """
        try:
            if action_name in ["type0_left", "type0_right"]:
                return action_instance.execute(mesh, boundary, reference_vertex_idx)
            elif action_name == "type1":
                return action_instance.execute(mesh, boundary, reference_vertex_idx, new_coords[0])
            else:
                return None
        except Exception:
            return None

    def process_action(self, action, mesh, boundary, reference_vertex_idx, M_angle):
        """
        Process a complete action: decode, validate, and execute
        
        Args:
            action: Raw action vector from SAC
            mesh: Mesh object to modify
            boundary: Boundary object to modify  
            reference_vertex_idx: Reference vertex index
            M_angle: Angle threshold parameter
            
        Returns:
            dict: Dictionary containing action results with keys:
                - action_valid: bool
                - action_name: str
                - element_quality_reward: float
                - boundary_quality_reward: float  
                - generated_element: list or None
        """
        # Decode action
        action_name, action_instance, new_coords, ref_idx = self.decode_action(action, boundary, reference_vertex_idx)

        # Initialize default values
        result = {
            'action_valid': False,
            'action_name': action_name,
            'element_quality_reward': 0.0,
            'boundary_quality_reward': -0.1,
            'generated_element': None
        }

        # Check validity
        action_valid = self.is_valid(boundary, ref_idx, action_instance, action_name, new_coords)
        result['action_valid'] = action_valid

        if action_valid:
            # Get quality rewards
            result['element_quality_reward'] = self.get_element_quality(boundary,
                                                                        ref_idx,
                                                                        action_instance,
                                                                        action_name,
                                                                        new_coords)

            result['boundary_quality_reward'] = self.get_boundary_quality(boundary,
                                                                          ref_idx,
                                                                          action_instance,
                                                                          action_name,
                                                                          new_coords,
                                                                          M_angle)

            # Execute action
            result['generated_element'] = self.execute_action(mesh,
                                                              boundary,
                                                              ref_idx,
                                                              action_instance,
                                                              action_name,
                                                              new_coords)

        return result

    def get_action_info(self):
        """
        Get detailed information about the current action configuration
        
        Returns:
            dict: Information about action configuration
        """
        return {
            "enabled_actions": self.enabled_actions,
            "total_enabled": len(self.enabled_actions),
            "auto_remap": self.auto_remap,
            "action_mapping": self.action_logit_mapping,
            "descriptions": self.get_action_descriptions()
        }

    def print_action_config(self):
        """Print current action configuration for debugging"""
        info = self.get_action_info()
        print("=== Action Manager Configuration ===")
        print(f"Enabled Actions: {info['enabled_actions']}")
        print(f"Total Enabled: {info['total_enabled']}")
        print(f"Auto Remap: {info['auto_remap']}")
        print("\nAction Descriptions:")
        for name, desc in info['descriptions'].items():
            print(f"  {name}: {desc}")
        print("\nAction Mapping:")
        for idx, mapping in info['action_mapping'].items():
            if 'range' in mapping:
                print(f"  Index {idx}: {mapping['name']} (range: {mapping['range']})")
            else:
                print(f"  Index {idx}: {mapping['name']}")
