from src.interfaces import Command
from src.rl.action import ActionType0Left, ActionType0Right, ActionType1
from copy import deepcopy


class ActionType0LeftCommand(Command):
    """
    Command wrapper for ActionType0Left that provides reversibility.
    
    This action connects 4 existing boundary vertices into a quadrilateral
    by adding an internal edge and removing 2 vertices from the boundary.
    
    Algorithm:
    - Gets vertices: v0 (ref), v1 (ref-1), v2 (ref-2), v3 (ref+1)
    - Forms quadrilateral [v0, v3, v2, v1]
    - Adds edge (v2, v3) to mesh
    - Removes vertices v0, v1 from boundary
    """

    def __init__(self, boundary, mesh, reference_vertex_idx):
        super().__init__(boundary, mesh, reference_vertex_idx)
        self._action_instance = ActionType0Left()

    def get_element(self, boundary, reference_vertex_idx):
        """
        Get the quadrilateral element that would be formed by this action.
        
        Returns:
            list: [v0, v3, v2, v1] vertices forming the quadrilateral
        """
        return self._action_instance.get_element(boundary, reference_vertex_idx)

    def is_valid(self):
        """
        Check if the action can be executed on the current boundary.
        
        Returns:
            bool: True if action is valid, False otherwise
        """
        return self._action_instance.is_valid(
            self.original_boundary,
            self.reference_vertex_idx
        )

    def execute(self):
        """
        Execute the action and return new copies of boundary and mesh.
        
        Returns:
            tuple: (new_boundary, new_mesh, generated_element)
        """
        if self.is_executed:
            return self.executed_boundary, self.executed_mesh, self.generated_element

        if not self.is_valid():
            raise ValueError("Cannot execute invalid action")

        # Create deep copies to ensure original objects are not modified
        new_boundary = deepcopy(self.original_boundary)
        new_mesh = deepcopy(self.original_mesh)

        # Execute the action on the copies
        element = self._action_instance.execute(
            new_mesh, new_boundary, self.reference_vertex_idx
        )

        if element is None:
            raise RuntimeError("Action execution failed")

        # Store the results for potential undo
        self.executed_boundary = new_boundary
        self.executed_mesh = new_mesh
        self.generated_element = element
        self.is_executed = True

        return new_boundary, new_mesh, element

    def undo(self):
        """
        Undo the action and return the original state.
        
        Returns:
            tuple: (original_boundary, original_mesh)
        """
        if not self.is_executed:
            raise RuntimeError("Cannot undo action that was not executed")

        # Reset execution state
        self.is_executed = False
        self.executed_boundary = None
        self.executed_mesh = None
        self.generated_element = None

        return self.original_boundary, self.original_mesh


class ActionType0RightCommand(Command):
    """
    Command wrapper for ActionType0Right that provides reversibility.
    
    This action connects 4 existing boundary vertices into a quadrilateral
    by adding an internal edge and removing 2 vertices from the boundary.
    
    Algorithm:
    - Gets vertices: v0 (ref), v1 (ref+1), v2 (ref+2), v3 (ref-1)
    - Forms quadrilateral [v0, v1, v2, v3]
    - Adds edge (v2, v3) to mesh
    - Removes vertices v0, v1 from boundary
    """

    def __init__(self, boundary, mesh, reference_vertex_idx):
        super().__init__(boundary, mesh, reference_vertex_idx)
        self._action_instance = ActionType0Right()

    def get_element(self, boundary, reference_vertex_idx):
        """
        Get the quadrilateral element that would be formed by this action.
        
        Returns:
            list: [v0, v1, v2, v3] vertices forming the quadrilateral
        """
        return self._action_instance.get_element(boundary, reference_vertex_idx)

    def is_valid(self):
        """
        Check if the action can be executed on the current boundary.
        
        Returns:
            bool: True if action is valid, False otherwise
        """
        return self._action_instance.is_valid(
            self.original_boundary,
            self.reference_vertex_idx
        )

    def execute(self):
        """
        Execute the action and return new copies of boundary and mesh.
        
        Returns:
            tuple: (new_boundary, new_mesh, generated_element)
        """
        if self.is_executed:
            return self.executed_boundary, self.executed_mesh, self.generated_element

        if not self.is_valid():
            raise ValueError("Cannot execute invalid action")

        # Create deep copies to ensure original objects are not modified
        new_boundary = deepcopy(self.original_boundary)
        new_mesh = deepcopy(self.original_mesh)

        # Execute the action on the copies
        element = self._action_instance.execute(
            new_mesh, new_boundary, self.reference_vertex_idx
        )

        if element is None:
            raise RuntimeError("Action execution failed")

        # Store the results for potential undo
        self.executed_boundary = new_boundary
        self.executed_mesh = new_mesh
        self.generated_element = element
        self.is_executed = True

        return new_boundary, new_mesh, element

    def undo(self):
        """
        Undo the action and return the original state.
        
        Returns:
            tuple: (original_boundary, original_mesh)
        """
        if not self.is_executed:
            raise RuntimeError("Cannot undo action that was not executed")

        # Reset execution state
        self.is_executed = False
        self.executed_boundary = None
        self.executed_mesh = None
        self.generated_element = None

        return self.original_boundary, self.original_mesh


class ActionType1Command(Command):
    """
    Command wrapper for ActionType1 that provides reversibility.
    
    This action adds a new vertex and creates a quadrilateral element
    by connecting the new vertex to existing boundary vertices.
    
    Algorithm:
    - Gets vertices: v0 (ref), v1 (ref-1), v3 (ref+1), v2 (new_vertex)
    - Forms quadrilateral [v0, v3, v2, v1]
    - Adds new vertex v2 to mesh
    - Adds edges (v1, v2) and (v2, v3) to mesh
    - Removes vertex v0 from boundary
    - Inserts vertex v2 into boundary at appropriate position
    """

    def __init__(self, boundary, mesh, reference_vertex_idx, new_vertex):
        super().__init__(boundary, mesh, reference_vertex_idx, new_vertex)
        self._action_instance = ActionType1()
        self.new_vertex = new_vertex

    def get_element(self, boundary, reference_vertex_idx, new_vertex):
        """
        Get the quadrilateral element that would be formed by this action.
        
        Returns:
            list: [v0, v3, v2, v1] vertices forming the quadrilateral
        """
        return self._action_instance.get_element(boundary, reference_vertex_idx, new_vertex)

    def is_valid(self):
        """
        Check if the action can be executed on the current boundary.
        
        Returns:
            bool: True if action is valid, False otherwise
        """
        return self._action_instance.is_valid(
            self.original_boundary,
            self.reference_vertex_idx,
            self.new_vertex
        )

    def execute(self):
        """
        Execute the action and return new copies of boundary and mesh.
        
        Returns:
            tuple: (new_boundary, new_mesh, generated_element)
        """
        if self.is_executed:
            return self.executed_boundary, self.executed_mesh, self.generated_element

        if not self.is_valid():
            raise ValueError("Cannot execute invalid action")

        # Create deep copies to ensure original objects are not modified
        new_boundary = deepcopy(self.original_boundary)
        new_mesh = deepcopy(self.original_mesh)

        # Execute the action on the copies
        element = self._action_instance.execute(
            new_mesh, new_boundary, self.reference_vertex_idx, self.new_vertex
        )

        if element is None:
            raise RuntimeError("Action execution failed")

        # Store the results for potential undo
        self.executed_boundary = new_boundary
        self.executed_mesh = new_mesh
        self.generated_element = element
        self.is_executed = True

        return new_boundary, new_mesh, element

    def undo(self):
        """
        Undo the action and return the original state.
        
        Returns:
            tuple: (original_boundary, original_mesh)
        """
        if not self.is_executed:
            raise RuntimeError("Cannot undo action that was not executed")

        # Reset execution state
        self.is_executed = False
        self.executed_boundary = None
        self.executed_mesh = None
        self.generated_element = None

        return self.original_boundary, self.original_mesh
