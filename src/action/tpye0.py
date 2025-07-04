from .action import Action
from ..geometry import Mesh, Boundary

class ActionType0(Action):
    """
    Type 0 Action in Article, only connect two point in boundary
    """

    def execute(self, mesh):
        pass

    def is_valid(self, boundary):
        pass
