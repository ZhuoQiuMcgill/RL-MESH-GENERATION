from src.geometry import *
from src.rl.action import ACTION_COMMAND_MAPPING
from src.rl.action.action_manager import ActionManager
from src.interfaces import Command
from copy import deepcopy


class MeshGenerator:
    """
    Modular mesh generation pipeline that supports reversible operations.
    
    This class provides a high-level interface for mesh generation using
    various predictor algorithms and reference point selectors. It maintains
    a command history for reversibility and supports step-by-step generation.
    
    The class is designed to be predictor-agnostic and can work with any
    predictor that implements the Predictor interface.
    """

    def __init__(self, boundary_vertices):
        """
        Initialize the mesh generator with boundary vertices.
        
        Args:
            boundary_vertices (list): List of boundary vertex coordinates
        """
        # Predictor management
        self.predictors = {}
        self.current_activated_predictor = None
        self.current_activated_ref_selector = None
        self.ref_selector_config = {}

        # Geometry objects - these are the master copies
        self.initial_boundary = Boundary(boundary_vertices)
        self.boundary = deepcopy(self.initial_boundary)
        self.mesh = Mesh(self.boundary)

        # Command history for reversibility
        self.command_history = []
        self.current_step = 0

        # Generation statistics
        self.generated_elements = []
        self.is_completed = False

        # Action manager for proper action decoding
        self.action_manager = ActionManager()

    def set_predictor(self, predictor):
        self.predictors[predictor.name()] = predictor

    def set_ref_selector(self, ref_selector, config=None):
        self.current_activated_ref_selector = ref_selector
        self.ref_selector_config = config or {}

    def update_activated_predictor(self, name):
        self.current_activated_predictor = self.predictors.get(name)
        if not self.current_activated_predictor:
            raise KeyError(f"[ERROR] Predictor {name} not found.")

    def check_complete(self):
        return self.boundary.size() <= 4

    def get_current_state_info(self):
        """
        Get general state information for predictors.
        
        This method returns generic information that can be used by any predictor.
        Specific state processing (like RL state vectors) should be handled
        by the individual predictor implementations.
        
        Returns:
            dict: Dictionary containing:
                - boundary: Current boundary object
                - mesh: Current mesh object
                - reference_vertex_idx: Current reference vertex index
                - step: Current generation step
                - completed: Whether generation is completed
                - generated_elements: List of generated elements
        """
        # Update completion status based on current boundary size
        if self.check_complete():
            self.is_completed = True
            
        if self.current_activated_ref_selector is None:
            # Fallback to boundary's default reference selection
            reference_vertex_idx = self.boundary.get_ref_vertex(2)  # Default n=2
        else:
            # Pass the stored configuration to the selector
            reference_vertex_idx = self.current_activated_ref_selector.select_reference_point(
                self.boundary, **self.ref_selector_config
            )

        return {
            'boundary': self.boundary,
            'mesh': self.mesh,
            'reference_vertex_idx': reference_vertex_idx,
            'step': self.current_step,
            'completed': self.is_completed,
            'generated_elements': self.generated_elements.copy(),
            'boundary_size': self.boundary.size(),
            'initial_boundary': self.initial_boundary
        }

    def step(self, ref_idx=None):
        """
        Execute one generation step using the current activated predictor.
        
        This method gets the current state, uses the predictor to make a decision,
        creates and executes an action command, and updates the generation state.
        
        Args:
            ref_idx (int, optional): If provided, force the step to use this reference vertex.
                                     If None, a reference vertex will be selected automatically.
                                     Defaults to None.
        
        Returns:
            dict: Step result containing:
                - success: Whether the step was successful
                - element: Generated element (if any)
                - command: Executed command for potential undo
                - message: Status message
                
        Raises:
            RuntimeError: If no predictor is activated or step fails
        """
        if self.current_activated_predictor is None:
            raise RuntimeError("No predictor activated. Call update_activated_predictor() first.")

        # Check completion status BEFORE starting any step processing
        if self.check_complete():
            self.is_completed = True
            return {
                'success': False,
                'element': None,
                'command': None,
                'message': 'Generation completed - boundary size <= 4'
            }

        if self.is_completed:
            return {
                'success': False,
                'element': None,
                'command': None,
                'message': 'Generation already marked as completed'
            }

        try:
            # Get current state information
            state_info = self.get_current_state_info()
            
            # If a specific reference index is provided, use it. Otherwise, use the one from the state.
            if ref_idx is not None:
                reference_vertex_idx = ref_idx
                state_info['reference_vertex_idx'] = ref_idx  # Ensure state is consistent for predictor
            else:
                reference_vertex_idx = state_info['reference_vertex_idx']

            # Use predictor to make decision
            prediction = self.current_activated_predictor.predict(state_info)
            action_vector = prediction['action_vector']

            # Use ActionManager to decode action properly
            action_name, new_coords, action_attempted = self.action_manager.decode_action(
                action_vector, self.boundary, reference_vertex_idx, command=True
            )

            # Map action names to command classes
            action_name_to_command = {
                'type0_left': ACTION_COMMAND_MAPPING[0],  # ActionType0LeftCommand
                'type0_right': ACTION_COMMAND_MAPPING[1],  # ActionType0RightCommand
                'type1': ACTION_COMMAND_MAPPING[2]  # ActionType1Command
            }

            command_class = action_name_to_command.get(action_name)
            if command_class is None:
                raise ValueError(f"Unknown action name: {action_name}")

            # Create command with appropriate parameters
            if action_name in ['type0_left', 'type0_right']:
                command = command_class(
                    self.boundary,
                    self.mesh,
                    reference_vertex_idx
                )
            elif action_name == 'type1':
                if not new_coords:
                    raise ValueError("ActionType1 requires new_coords from ActionManager")
                command = command_class(
                    self.boundary,
                    self.mesh,
                    reference_vertex_idx,
                    new_coords[0]  # new_coords is a list
                )
            else:
                raise ValueError(f"Unsupported action name: {action_name}")

            # Get validation details before attempting execution
            is_valid = command.is_valid()
            validation_message = None
            if not is_valid:
                # Try to get specific validation failure reason
                try:
                    # Attempt to get detailed error by trying to execute and catching the error
                    command.execute()
                except Exception as e:
                    validation_message = str(e)
            
            # Prepare action information for frontend
            action_info = {
                'action_type': action_name,
                'reference_vertex_idx': reference_vertex_idx,
                'new_coords': new_coords if new_coords else None,
                'is_valid': is_valid,
                'validation_message': validation_message,
                'action_attempted': action_attempted
            }
            
            if not is_valid:
                return {
                    'success': False,
                    'element': None,
                    'command': command,
                    'action_info': action_info,
                    'message': f'Invalid action {action_name} at reference {reference_vertex_idx}' + 
                              (f': {validation_message}' if validation_message else '')
                }

            # Execute command - this returns new boundary and mesh
            new_boundary, new_mesh, element = command.execute()

            # Update current state with new objects
            self.boundary = new_boundary
            self.mesh = new_mesh
            self.generated_elements.append(element)

            # Store command for potential undo
            self.command_history.append(command)
            self.current_step += 1

            # Prepare action information for successful execution
            action_info = {
                'action_type': action_name,
                'reference_vertex_idx': reference_vertex_idx,
                'new_coords': new_coords if new_coords else None,
                'is_valid': True,
                'validation_message': None
            }
            
            return {
                'success': True,
                'element': element,
                'command': command,
                'action_info': action_info,
                'message': f'Successfully executed action {action_name}'
            }

        except Exception as e:
            return {
                'success': False,
                'element': None,
                'command': None,
                'message': f'Step failed: {str(e)}'
            }

    def undo(self):
        """
        Undo the last generation step and return to the previous state.
        
        Returns:
            dict: Undo result containing:
                - success: Whether undo was successful
                - message: Status message
                
        Raises:
            RuntimeError: If no steps to undo
        """
        if not self.command_history:
            return {
                'success': False,
                'message': 'No steps to undo'
            }

        try:
            # Get the last command
            last_command = self.command_history.pop()

            # Undo the command - this returns original boundary and mesh
            original_boundary, original_mesh = last_command.undo()

            # Restore previous state
            self.boundary = original_boundary
            self.mesh = original_mesh

            # Remove last generated element
            if self.generated_elements:
                self.generated_elements.pop()

            # Update step counter
            self.current_step -= 1
            if self.current_step < 0:
                self.current_step = 0

            # Update completion status
            self.is_completed = False

            return {
                'success': True,
                'message': f'Successfully undone step {self.current_step + 1}'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Undo failed: {str(e)}'
            }

    def reset(self):
        """
        Reset the mesh generator to initial state.
        
        Returns:
            dict: Reset result
        """
        try:
            # Reset to initial state
            self.boundary = deepcopy(self.initial_boundary)
            self.mesh = Mesh(self.boundary)

            # Clear history and statistics
            self.command_history = []
            self.generated_elements = []
            self.current_step = 0
            self.is_completed = False

            return {
                'success': True,
                'message': 'Successfully reset to initial state'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Reset failed: {str(e)}'
            }

    def get_status(self):
        """
        Get current status information for frontend or monitoring.
        
        Returns:
            dict: Status information containing:
                - current_step: Current generation step
                - boundary_size: Current boundary vertex count
                - generated_elements_count: Number of generated elements
                - is_completed: Whether generation is completed
                - active_predictor: Name of active predictor
                - can_undo: Whether undo is possible
                - mesh_data: Current mesh adjacency data
                - boundary_vertices: Current boundary vertices
        """
        try:
            mesh_data = self.mesh.get_adjacency_dict() if self.mesh else {}
        except Exception:
            mesh_data = {}

        return {
            'current_step': self.current_step,
            'boundary_size': self.boundary.size() if self.boundary else 0,
            'generated_elements_count': len(self.generated_elements),
            'is_completed': self.is_completed,
            'active_predictor': self.current_activated_predictor.name() if self.current_activated_predictor else None,
            'can_undo': len(self.command_history) > 0,
            'mesh_data': mesh_data,
            'boundary_vertices': self.boundary.get_vertices() if self.boundary else [],
            'total_steps_possible': max(0, self.boundary.size() - 4) if self.boundary else 0
        }
