from src.interfaces import Predictor


class RLPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.agent = None

    def init_agent(self, path):
        pass

    def predict(self, state_info):
        pass

    def name(self):
        return "RL"