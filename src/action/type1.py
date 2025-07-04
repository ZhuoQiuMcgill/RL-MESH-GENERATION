from .action import Action


class ActionType1(Action):
    """
    Type 1 action in Article, insert one vertex inside boundary and connect edges
    """

    def execute(self, mesh):
        pass

    def is_valid(self, boundary):
        pass
