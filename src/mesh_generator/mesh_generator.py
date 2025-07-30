from src.geometry import *


class MeshGenerator:
    def __init__(self, boundary_vertices):
        self.predictors = {}
        self.current_used_predictor = None
        self.boundary = Boundary(boundary_vertices)
        self.mesh = Mesh(self.boundary)

    def set_predictor(self, predictor):
        self.predictors[predictor.name()] = predictor

    def update_used_predictor(self, name):
        self.current_used_predictor = self.predictors.get(name)
        if not self.current_used_predictor:
            raise KeyError(f"[ERROR] Predictor {name} not found.")

    def get_current_state_info(self):
        """

        :return:
        """
        pass
