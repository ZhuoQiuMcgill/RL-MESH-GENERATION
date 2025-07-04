from .action import Action
from ..geometry import Mesh, Boundary


class ActionType2(Action):
    """
    Type 2 action in Article, insert two point inside boundary and connect edges
    """

    def execute(self, mesh):
        pass

    def is_valid(self, boundary):
        pass
