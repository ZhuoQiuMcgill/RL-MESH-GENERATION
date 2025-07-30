from src.interfaces import Predictor


class RLPredictor(Predictor):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    def predict(self, state_info):
        pass

    def name(self):
        return "RL"