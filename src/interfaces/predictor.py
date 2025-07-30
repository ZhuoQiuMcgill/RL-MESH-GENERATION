from abc import ABC, abstractmethod


class Predictor(ABC):
    """
    Abstract base class for prediction algorithms.
    Defines the interface for all predictor implementations.
    """
    
    def __init__(self):
        pass

    @abstractmethod
    def name(self):
        """
        Returns the name of the predictor.
        
        Returns:
            str: The name identifier for this predictor
        """
        pass

    @abstractmethod
    def predict(self, state_info):
        """
        Makes a prediction based on the given state information.
        
        Args:
            state_info: The current state information for prediction
            
        Returns:
            The prediction result
        """
        pass