import os
import json
import logging

DATA_DIR = os.path.join(os.getcwd(), "data", "history")


def get_training_history_dir(training_id):
    return os.path.join(DATA_DIR, training_id, "history")


def save_episode_details(details, best_episode, path):
    """
    将所有的detail都保存成一个JSON文件，格式为
    {"size": n, "best_episode": m, "details": [detail]}
    :param details: list of detail dict, check _EpisodeCallback for the structure of detail dict
    :param best_episode: int
    :param path: path from training_manager
    :return: None
    """
    filename = os.path.join(path, "details.json")

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # 构建保存的数据结构
    data = {
        "size": len(details),
        "best_episode": best_episode,
        "details": details
    }

    # 保存到JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


class HistoryManager:
    def __init__(self):
        self.id = None
        self.path = None
        self.detail_data = None
        self.best_episode = 0
        self.size = 0
        self.history_dir = DATA_DIR
        self.focused = False
        self.logger = logging.getLogger(__name__)

    def focus_on(self, training_id):
        """
        聚焦到指定的训练ID
        :param training_id: str
        """
        # 检查training_id是否存在
        training_path = get_training_history_dir(training_id)
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training ID not found: {training_id}")

        self.id = training_id
        self.path = training_path
        self.focused = True
        self.read_data()

    def list_training_id(self):
        """
        Return a list of dir name in self.history_dir
        :return: list of str
        """
        if not os.path.exists(self.history_dir):
            return []

        training_ids = []
        try:
            for item in os.listdir(self.history_dir):
                item_path = os.path.join(self.history_dir, item)
                if os.path.isdir(item_path):
                    training_ids.append(item)
        except Exception as e:
            self.logger.error(f"Error listing training IDs: {e}")
            return []

        return sorted(training_ids)

    def read_data(self):
        """
        Read json file and update [self.detail_data, self.size, self.best_episode]
        :return:
        """
        if not self.path:
            return

        filename = os.path.join(self.path, "details.json")

        if not os.path.exists(filename):
            # 如果文件不存在，初始化空数据
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.detail_data = data.get("details", [])
            self.size = data.get("size", len(self.detail_data))
            self.best_episode = data.get("best_episode", 0)

        except Exception as e:
            self.logger.error(f"Error reading data from {filename}: {e}")
            # 如果读取失败，初始化空数据
            self.detail_data = []
            self.size = 0
            self.best_episode = 0

    def get_episode_data(self, episode_index):
        """
        Return a single detail data, only return when is focused
        :param episode_index: int
        :return: single detail dict
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        if episode_index < 0 or episode_index >= len(self.detail_data):
            raise IndexError(f"Episode index {episode_index} out of range (0-{len(self.detail_data) - 1})")

        return self.detail_data[episode_index]

    def current_focus_id(self):
        """
        Return the current focused training ID
        :return: str or None
        """
        return self.id
