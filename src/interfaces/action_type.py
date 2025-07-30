from abc import ABC, abstractmethod
from src.quality import *


class ActionType(ABC):
    """
    Abstract base class for action types in mesh generation.
    Defines the interface for geometric operations that modify mesh and boundary.
    """
    
    QUALITY_THRESHOLD = 0.00

    @abstractmethod
    def execute(self, mesh, boundary, reference_vertex_idx, *coords):
        """
        Execute the specific geometric operation to modify mesh and boundary.
        
        Args:
            mesh: The mesh to be modified
            boundary: The boundary to be modified
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Variable coordinates for the operation
            
        Returns:
            The generated elements
        """
        pass

    @abstractmethod
    def is_valid(self, boundary, reference_vertex_idx, *coords, alpha=2, n=2):
        """
        Check if the action is valid under the current boundary conditions.
        
        Args:
            boundary: The current boundary
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Variable coordinates for validation
            alpha (int): Alpha parameter for validation (default: 2)
            n (int): N parameter for validation (default: 2)
            
        Returns:
            bool: True if the action is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_element(self, boundary, reference_vertex_idx, *coords):
        """
        Get the element for the given boundary and coordinates.
        
        Args:
            boundary: The boundary
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Variable coordinates
            
        Returns:
            The element
        """
        pass

    @staticmethod
    def element_quality(element) -> float:
        """
        Calculate hybrid quality = robust * (clamped Scaled-Jacobian)**gamma
        
        Args:
            element: Iterable of 4 vertices
            
        Returns:
            float: Quality value in [0, 1]
        """
        return quality_hybrid(element)

    @staticmethod
    def calculate_angle_quality(angle1, angle2, M_angle):
        """
        Calculate angle quality based on three angles.
        
        Args:
            angle1: First angle
            angle2: Second angle
            M_angle: Maximum angle
            
        Returns:
            float: Calculated angle quality
        """
        return min(angle1, angle2, M_angle) / M_angle

    @abstractmethod
    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        """
        Get the quality of the element for given parameters.
        
        Args:
            boundary: The boundary
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Variable coordinates
            
        Returns:
            float: Element quality
        """
        pass

    @abstractmethod
    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        """
        Get the quality of the boundary for given parameters.
        
        Args:
            boundary: The boundary
            reference_vertex_idx (int): Index of the reference vertex
            *coords: Variable coordinates
            M_angle: Maximum angle parameter
            
        Returns:
            float: Boundary quality
        """
        pass