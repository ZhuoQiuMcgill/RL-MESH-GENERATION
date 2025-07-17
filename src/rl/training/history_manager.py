import os

DATA_DIR = os.path.join(os.getcwd(), "data", "history")


def get_training_history_dir(training_id):
    return os.path.join(DATA_DIR, training_id, "history")


def save_episode_details(details, best_episode, path):
    """
    TODO: 将所有的detail都保存成一个JSON文件，格式为
    TODO: {"size": n, "best_episode“: m, "details": [detail]}
    :param details: list of detail dict, check _EpisodeCallback for the structure of detail dict
    :param best_episode: int
    :param path: path from training_manager
    :return: None
    """
    filename = os.path.join(path, "details.json")


class HistoryManager:
    def __init__(self):
        self.id = None
        self.path = None
        self.detail_data = None
        self.best_episode = 0
        self.size = 0
        self.history_dir = DATA_DIR
        self.focused = False

    def focus_on(self, training_id):
        # TODO: Handle exception, if training_id not found
        self.id = training_id
        self.path = get_training_history_dir(training_id)
        self.focused = True
        self.read_data()

    def list_training_id(self):
        """
        TODO: Return a list of dir name in self.history_dir
        :return: list of str
        """

    def read_data(self):
        """
        TODO: Read json file and update [self.detail_data, self.size, self.best_episode]
        :return:
        """
        if self.path:
            filename = os.path.join(self.path, "details.json")

    def get_episode_data(self, episode_index):
        """
        TODO: Return a single detail data, only return when is focused
        :param episode_index: int
        :return: single detail dict
        """
