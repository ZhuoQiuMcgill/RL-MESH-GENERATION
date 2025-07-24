import os
import json
import logging
import time
from threading import Lock

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
        
        # 智能缓存和文件监控相关属性
        self._cache_lock = Lock()  # 线程安全锁
        self._data_loaded = False  # 数据是否已加载的标志
        self._file_last_modified = None  # 文件最后修改时间
        self._cache_valid = False  # 缓存是否有效

    def focus_on(self, training_id):
        """
        聚焦到指定的训练ID
        :param training_id: str
        """
        # 检查training_id是否存在
        training_path = get_training_history_dir(training_id)
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training ID not found: {training_id}")

        with self._cache_lock:
            # 如果切换到新的training_id，清除旧缓存
            if self.id != training_id:
                self._clear_cache()
            
            self.id = training_id
            self.path = training_path
            self.focused = True
            self._ensure_data_loaded()

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

    def _get_file_modified_time(self):
        """
        获取details.json文件的最后修改时间
        :return: float 修改时间戳，如果文件不存在返回None
        """
        if not self.path:
            return None
        
        filename = os.path.join(self.path, "details.json")
        if not os.path.exists(filename):
            return None
        
        try:
            return os.path.getmtime(filename)
        except Exception as e:
            self.logger.warning(f"无法获取文件修改时间 {filename}: {e}")
            return None

    def _is_file_modified(self):
        """
        检查文件是否被修改
        :return: bool 文件是否被修改
        """
        current_modified_time = self._get_file_modified_time()
        
        # 如果文件不存在，认为没有修改
        if current_modified_time is None:
            return False
        
        # 如果是第一次检查，记录当前时间
        if self._file_last_modified is None:
            self._file_last_modified = current_modified_time
            return True  # 第一次加载认为需要读取
        
        # 比较修改时间
        if current_modified_time != self._file_last_modified:
            self._file_last_modified = current_modified_time
            return True
        
        return False

    def _clear_cache(self):
        """
        清除缓存数据
        """
        self.detail_data = None
        self.best_episode = 0
        self.size = 0
        self.non_zero_step_episodes = []
        self._data_loaded = False
        self._file_last_modified = None
        self._cache_valid = False
        self.logger.debug("缓存已清除")

    def _ensure_data_loaded(self):
        """
        确保数据已加载，只有在需要时才重新读取文件
        """
        # 检查缓存是否有效
        if self._cache_valid and self._data_loaded and not self._is_file_modified():
            self.logger.debug("使用缓存数据，跳过文件读取")
            return
        
        # 需要重新加载数据
        self.logger.debug("重新加载数据文件")
        self._load_data_from_file()

    def _load_data_from_file(self):
        """
        从文件加载数据的内部方法
        """
        if not self.path:
            self._clear_cache()
            return

        filename = os.path.join(self.path, "details.json")

        if not os.path.exists(filename):
            # 如果文件不存在，初始化空数据
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            self.non_zero_step_episodes = []
            self._data_loaded = True
            self._cache_valid = True
            self._file_last_modified = None
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.detail_data = data.get("details", [])
            self.size = data.get("size", len(self.detail_data))
            self.best_episode = data.get("best_episode", 0)
            self.non_zero_step_episodes = data.get("non_zero_step_episodes_index", [])

            # 更新缓存状态
            self._data_loaded = True
            self._cache_valid = True
            self._file_last_modified = self._get_file_modified_time()

            self.logger.info(f"成功加载训练数据: {self.id}, size={self.size}, best_episode={self.best_episode}")
            self.logger.info(f"非零步数episodes: {len(self.non_zero_step_episodes)}个")

        except Exception as e:
            self.logger.error(f"Error reading data from {filename}: {e}")
            # 如果读取失败，初始化空数据但标记缓存无效
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            self.non_zero_step_episodes = []
            self._data_loaded = True
            self._cache_valid = False  # 标记缓存无效，下次会重试

    def read_data(self):
        """
        Read json file and update [self.detail_data, self.size, self.best_episode, self.non_zero_step_episodes]
        保留此方法以保持向后兼容性，但现在使用智能缓存
        :return:
        """
        with self._cache_lock:
            self._ensure_data_loaded()

    def get_episode_data(self, episode_index):
        """
        Return a single detail data, only return when is focused
        根据index获取数据，即使episode_number很大也要返回对应的数据
        现在使用智能缓存，避免重复读取文件
        :param episode_index: int 在non_zero_step_episodes中的索引
        :return: single detail dict
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            # 使用智能缓存机制确保数据已加载
            self._ensure_data_loaded()

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

        with self._cache_lock:
            self._ensure_data_loaded()

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

        with self._cache_lock:
            self._ensure_data_loaded()

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

            with self._cache_lock:
                # 标记缓存无效，强制重新加载
                self._cache_valid = False
                self._ensure_data_loaded()

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

        with self._cache_lock:
            self._ensure_data_loaded()
            return self.get_episode_index_by_number(self.best_episode)

    def get_statistics(self):
        """
        获取训练统计信息
        :return: dict 包含统计信息
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            self._ensure_data_loaded()

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
        with self._cache_lock:
            self.id = None
            self.path = None
            self.focused = False
            self._clear_cache()
            self.logger.info("已清除聚焦状态")

    def get_cache_status(self):
        """
        获取缓存状态信息，用于调试和监控
        :return: dict 缓存状态信息
        """
        return {
            "focused": self.focused,
            "training_id": self.id,
            "data_loaded": self._data_loaded,
            "cache_valid": self._cache_valid,
            "file_last_modified": self._file_last_modified,
            "current_file_modified": self._get_file_modified_time(),
            "data_size": len(self.detail_data) if self.detail_data else 0
        }

    def force_refresh(self):
        """
        强制刷新缓存，重新从文件加载数据
        :return: None
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")
        
        with self._cache_lock:
            self.logger.info(f"强制刷新缓存: {self.id}")
            self._cache_valid = False
            self._ensure_data_loaded()

    def is_cache_valid(self):
        """
        检查当前缓存是否有效
        :return: bool 缓存是否有效
        """
        if not self._cache_valid or not self._data_loaded:
            return False
        
        return not self._is_file_modified()
