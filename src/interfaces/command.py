
from abc import ABC, abstractmethod
from copy import deepcopy


class Command(ABC):
    """
    Abstract base class for reversible commands in the mesh generation pipeline.
    
    Commands are used in the prediction pipeline to maintain action reversibility
    by creating new copies of mesh and boundary objects instead of modifying
    the original ones in-place.
    """
    
    def __init__(self, boundary, mesh, reference_vertex_idx, *coords):
        """
        Initialize the command with the required parameters.
        
        Args:
            boundary: The boundary object (will be copied for execution)
            mesh: The mesh object (will be copied for execution)
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Additional coordinates for the action
        """
        self.original_boundary = boundary
        self.original_mesh = mesh
        self.reference_vertex_idx = reference_vertex_idx
        self.coords = coords
        self.executed_boundary = None
        self.executed_mesh = None
        self.generated_element = None
        self.is_executed = False

    @abstractmethod
    def execute(self):
        """
        Execute the command and return copies of the modified boundary and mesh.
        
        Returns:
            tuple: (new_boundary, new_mesh, generated_element)
        """
        pass

    @abstractmethod
    def undo(self):
        """
        Undo the command and return the original state.
        
        Returns:
            tuple: (original_boundary, original_mesh)
        """
        pass
    
    @abstractmethod
    def is_valid(self):
        """
        Check if the command can be executed on the current boundary state.
        
        Returns:
            bool: True if the command is valid, False otherwise
        """
        pass