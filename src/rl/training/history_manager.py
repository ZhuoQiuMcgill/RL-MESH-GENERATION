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

    # 只保存非零步数的episode
    non_zero_step_details = []
    non_zero_step_episodes = []

    for detail in details:
        if detail.get("l", 0) > 0:  # l是step数量
            non_zero_step_details.append(detail)
            non_zero_step_episodes.append(detail.get("episode_number"))

    # 构建保存的数据结构
    data = {
        "size": len(non_zero_step_details),
        "best_episode": best_episode,
        "non_zero_step_episodes_index": non_zero_step_episodes,
        "details": non_zero_step_details
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
        self.non_zero_step_episodes = []
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
        Read json file and update [self.detail_data, self.size, self.best_episode, self.non_zero_step_episodes]
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
            self.non_zero_step_episodes = []
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.detail_data = data.get("details", [])
            self.size = data.get("size", len(self.detail_data))
            self.best_episode = data.get("best_episode", 0)
            self.non_zero_step_episodes = data.get("non_zero_step_episodes_index", [])

            self.logger.info(f"成功读取训练数据: {self.id}, size={self.size}, best_episode={self.best_episode}")
            self.logger.info(f"非零步数episodes: {len(self.non_zero_step_episodes)}个")

        except Exception as e:
            self.logger.error(f"Error reading data from {filename}: {e}")
            # 如果读取失败，初始化空数据
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            self.non_zero_step_episodes = []

    def get_episode_data(self, episode_index):
        """
        Return a single detail data, only return when is focused
        根据index获取数据，即使episode_number很大也要返回对应的数据
        :param episode_index: int 在non_zero_step_episodes中的索引
        :return: single detail dict
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        if episode_index < 0 or episode_index >= len(self.detail_data):
            raise IndexError(f"Episode index {episode_index} out of range (0-{len(self.detail_data) - 1})")

        # 根据index直接返回对应的detail数据
        # 这里的episode_index是在non_zero_step_episodes中的索引
        episode_data = self.detail_data[episode_index]

        # 记录实际的episode_number供调试使用
        actual_episode_number = episode_data.get("episode_number", "unknown")
        self.logger.debug(f"获取Episode index={episode_index}, 实际episode_number={actual_episode_number}")

        return episode_data

    def get_episode_by_number(self, episode_number):
        """
        根据episode_number获取数据
        :param episode_number: int 实际的episode编号
        :return: single detail dict or None if not found
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        # 在非零步数的episodes中查找
        for detail in self.detail_data:
            if detail.get("episode_number") == episode_number:
                return detail

        return None

    def get_episode_index_by_number(self, episode_number):
        """
        根据episode_number获取其在non_zero_step_episodes中的索引
        :param episode_number: int 实际的episode编号
        :return: int 索引位置，如果不存在返回-1
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        # 在详细数据中查找episode_number对应的索引
        for index, detail in enumerate(self.detail_data):
            if detail.get("episode_number") == episode_number:
                return index

        return -1

    def update_data(self, new_details, new_best_episode):
        """
        更新当前训练的数据并保存到文件
        :param new_details: list of detail dict
        :param new_best_episode: int
        :return: None
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        try:
            # 调用保存函数，会自动过滤非零步数的数据
            save_episode_details(new_details, new_best_episode, self.path)

            # 重新读取数据以更新内存中的状态
            self.read_data()

            self.logger.info(f"成功更新训练数据: {self.id}, new_size={self.size}, new_best_episode={self.best_episode}")

        except Exception as e:
            self.logger.error(f"Error updating data for {self.id}: {e}")
            raise

    def get_best_episode_index(self):
        """
        获取最佳episode在non_zero_step_episodes中的索引
        :return: int 最佳episode的索引，如果不存在返回-1
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        return self.get_episode_index_by_number(self.best_episode)

    def get_statistics(self):
        """
        获取训练统计信息
        :return: dict 包含统计信息
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        if self.detail_data is None:
            self.read_data()

        if not self.detail_data:
            return {
                "total_episodes": 0,
                "non_zero_episodes": 0,
                "best_episode": 0,
                "best_episode_index": -1,
                "avg_reward": 0.0,
                "avg_length": 0.0
            }

        rewards = [detail.get("r", 0) for detail in self.detail_data]
        lengths = [detail.get("l", 0) for detail in self.detail_data]

        return {
            "total_episodes": len(self.non_zero_step_episodes),  # 只计算非零步数的
            "non_zero_episodes": len(self.detail_data),
            "best_episode": self.best_episode,
            "best_episode_index": self.get_best_episode_index(),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "episode_numbers": self.non_zero_step_episodes.copy()
        }

    def current_focus_id(self):
        """
        Return the current focused training ID
        :return: str or None
        """
        return self.id

    def is_focused(self):
        """
        检查是否已聚焦到某个训练ID
        :return: bool
        """
        return self.focused

    def clear_focus(self):
        """
        清除当前聚焦状态
        :return: None
        """
        self.id = None
        self.path = None
        self.detail_data = None
        self.best_episode = 0
        self.size = 0
        self.focused = False
        self.non_zero_step_episodes = []
        self.logger.info("已清除聚焦状态")
