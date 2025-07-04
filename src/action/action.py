from abc import ABC


class Action(ABC):

    def __init__(self):
        pass

    def execute(self, mesh):
        pass

    def is_valid(self, boundary):
        pass


