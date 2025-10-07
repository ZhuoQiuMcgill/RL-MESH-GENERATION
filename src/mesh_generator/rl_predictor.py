from src.interfaces import Predictor
from src.utils import normalize_coordinates_cartesian
import numpy as np
import os
from stable_baselines3 import SAC


class RLPredictor(Predictor):
    """
    Reinforcement Learning predictor for mesh generation.
    
    This predictor converts the general state information from MeshGenerator
    into the specific state format expected by RL agents, consistent with
    the MeshEnv implementation.
    """
    
    def __init__(self, n=2, g=3, beta=6):
        """
        Initialize the RL predictor.
        
        Args:
            n (int): Number of neighbor vertices to consider (default: 2)
            g (int): Number of fan points to generate (default: 3) 
            beta (float): Fan angle parameter (default: 6)
        """
        super().__init__()
        self.agent = None
        self.n = n
        self.g = g
        self.beta = beta
        self.is_loaded = False

    def init_agent(self, agent_path=None, agent_instance=None):
        """
        Initialize the RL agent.
        
        Args:
            agent_path (str, optional): Path to saved Stable-Baselines3 SAC model (.zip file)
            agent_instance (optional): Pre-initialized agent instance
            
        Raises:
            ValueError: If neither agent_path nor agent_instance is provided
            RuntimeError: If agent loading fails
        """
        if agent_instance is not None:
            self.agent = agent_instance
            self.is_loaded = True
        elif agent_path is not None:
            try:
                # Validate path exists
                if not os.path.exists(agent_path):
                    raise FileNotFoundError(f"Agent model file not found: {agent_path}")
                
                # Validate file extension
                if not agent_path.endswith('.zip'):
                    raise ValueError(f"Agent model must be a .zip file from Stable-Baselines3, got: {agent_path}")
                
                # Load Stable-Baselines3 SAC model
                print(f"Loading SAC model from: {agent_path}")
                self.agent = SAC.load(agent_path)
                print(f"Successfully loaded SAC model")
                
                self.is_loaded = True
                
            except FileNotFoundError as e:
                raise RuntimeError(f"Model file not found: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load SAC model from {agent_path}: {e}")
        else:
            raise ValueError("Either agent_path or agent_instance must be provided")

    def predict(self, state_info):
        """
        Make a prediction using the RL agent.
        
        This method converts the general state_info from MeshGenerator into
        the specific RL state format and gets a prediction from the agent.
        
        Args:
            state_info (dict): State information from MeshGenerator containing:
                - boundary: Current boundary object
                - mesh: Current mesh object  
                - reference_vertex_idx: Current reference vertex index
                - step: Current generation step
                - completed: Whether generation is completed
                - etc.
                
        Returns:
            dict: Prediction result containing:
                - action_vector: Raw action vector from agent for ActionManager.decode_action()
                - confidence: Prediction confidence (optional)
                
        Raises:
            RuntimeError: If agent is not initialized, prediction fails, or generation is completed
        """
        if not self.is_loaded or self.agent is None:
            raise RuntimeError("Agent not initialized. Call init_agent() first.")
        
        # Check if generation is completed - predictor should only be used when NOT completed
        if state_info.get('completed', False):
            raise RuntimeError("Cannot predict on completed generation. Generation is already finished.")
        
        try:
            # Extract required information from state_info
            boundary = state_info['boundary']
            reference_vertex_idx = state_info['reference_vertex_idx']
            
            # Build RL state vector following MeshEnv._get_obs() pattern
            rl_state = self._build_rl_state(boundary, reference_vertex_idx)
            
            # Get prediction from Stable-Baselines3 SAC agent
            # SAC.predict() returns (action, _states) tuple
            action, _states = self.agent.predict(rl_state, deterministic=True)
            
            # Return raw action vector for ActionManager to decode
            return {
                'action_vector': action
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _build_rl_state(self, boundary, reference_vertex_idx):
        """
        Build RL state vector consistent with MeshEnv._get_obs().
        
        Args:
            boundary: Boundary object
            reference_vertex_idx (int): Reference vertex index
            
        Returns:
            np.ndarray: State vector for RL agent
        """
        if boundary.size() < 3:
            # Return zero state for minimal boundary
            state_dim = (self.n * 2 + self.g) * 2
            return np.zeros(state_dim, dtype=np.float32)
        
        # Get reference vertex and neighbors (following MeshEnv pattern)
        get_type = "exclude ref"
        ref_v = boundary.get_vertex_by_index(reference_vertex_idx)
        right_neighbor_v = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        neighbor_coords = boundary.get_neighbors(reference_vertex_idx, self.n, get_type=get_type)
        
        # Get fan-sector vertices
        try:
            fan_coords = list(boundary.get_fan_points(reference_vertex_idx, self.n, self.beta, self.g))
        except Exception:
            fan_coords = [None] * self.g
        
        # Calculate scale factor
        base_len = boundary.get_avg_neighbor_length(reference_vertex_idx, self.n)
        scale_factor = 1.0 / base_len if base_len > 0 else 1.0
        
        # Normalize coordinates
        normalized_vertex = normalize_coordinates_cartesian(
            neighbor_coords + fan_coords, ref_v, right_neighbor_v, scale_factor
        )
        
        # Build state vector
        state_components = []
        for r, theta in normalized_vertex:
            state_components.extend([r, theta])
        
        return np.array(state_components, dtype=np.float32)
    
    

    def name(self):
        """
        Get the name of this predictor.
        
        Returns:
            str: Predictor name
        """
        return "RL"
    
    def is_ready(self):
        """
        Check if the predictor is ready for making predictions.
        
        Returns:
            bool: True if agent is loaded and ready
        """
        return self.is_loaded and self.agent is not None