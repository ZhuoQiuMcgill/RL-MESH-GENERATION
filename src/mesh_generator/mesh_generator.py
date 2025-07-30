class MeshGenerator:
    def __init__(self, boundary_vertices):
        self.predictors = {}
        self.current_used_predictor = None


    def init_env(self):
        pass

    def set_predict(self, predictor):
        self.predictors[predictor.name()] = predictor

    def update_used_predictor(self, name):
        self.current_used_predictor = self.predictors.get(name)
        if not self.current_used_predictor:
            raise KeyError(f"[ERROR] Predictor {name} not found.")

    def register(self, generator):
        pass

    def get_current_state_info(self):
        pass


