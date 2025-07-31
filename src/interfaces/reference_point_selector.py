from abc import ABC, abstractmethod

class ReferencePointSelector(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def select_reference_point(boundary, **info):
        pass

    @staticmethod
    @abstractmethod
    def get_interior_angle(boundary, ref_index, n=2):
        pass


