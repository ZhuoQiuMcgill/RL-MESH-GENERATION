from abc import ABC, abstractmethod

class ReferencePointSelector(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def select_reference_point(boundary, **info):
        pass

