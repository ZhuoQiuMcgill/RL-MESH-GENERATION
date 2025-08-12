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
    Save all details to a JSON file with format:
    {"size": n, "best_episode": m, "details": [detail]}
    :param details: list of detail dict, check _EpisodeCallback for the structure of detail dict
    :param best_episode: int
    :param path: path from training_manager
    :return: None
    """
    filename = os.path.join(path, "details.json")

    # Ensure directory exists
    os.makedirs(path, exist_ok=True)

    # Only save episodes with generated_elements > 0
    non_zero_generated_details = []
    non_zero_generated_episodes = []

    for detail in details:
        if detail.get("generated_elements", 0) > 0:  # generated_elements is the number of generated elements
            non_zero_generated_details.append(detail)
            non_zero_generated_episodes.append(detail.get("episode_number"))

    # Construct the data structure to save
    data = {
        "size": len(non_zero_generated_details),
        "best_episode": best_episode,
        "non_zero_generated_episodes_index": non_zero_generated_episodes,
        "details": non_zero_generated_details
    }

    # Save to JSON file
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
        self.non_zero_generated_episodes = []
        self.logger = logging.getLogger(__name__)
        
        # Smart caching and file monitoring related attributes
        self._cache_lock = Lock()  # Thread safety lock
        self._data_loaded = False  # Flag indicating whether data has been loaded
        self._file_last_modified = None  # Last modified time of file
        self._cache_valid = False  # Whether cache is valid

    def focus_on(self, training_id):
        """
        Focus on the specified training ID
        :param training_id: str
        """
        # Check if training_id exists
        training_path = get_training_history_dir(training_id)
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training ID not found: {training_id}")

        with self._cache_lock:
            # Clear old cache if switching to a new training_id
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
        Get the last modified time of details.json file
        :return: float modification timestamp, None if file doesn't exist
        """
        if not self.path:
            return None
        
        filename = os.path.join(self.path, "details.json")
        if not os.path.exists(filename):
            return None
        
        try:
            return os.path.getmtime(filename)
        except Exception as e:
            self.logger.warning(f"Unable to get file modification time {filename}: {e}")
            return None

    def _is_file_modified(self):
        """
        Check if file has been modified
        :return: bool whether file has been modified
        """
        current_modified_time = self._get_file_modified_time()
        
        # If file doesn't exist, consider it unmodified
        if current_modified_time is None:
            return False
        
        # If this is the first check, record current time
        if self._file_last_modified is None:
            self._file_last_modified = current_modified_time
            return True  # First load is considered as needing to read
        
        # Compare modification times
        if current_modified_time != self._file_last_modified:
            self._file_last_modified = current_modified_time
            return True
        
        return False

    def _clear_cache(self):
        """
        Clear cache data
        """
        self.detail_data = None
        self.best_episode = 0
        self.size = 0
        self.non_zero_generated_episodes = []
        self._data_loaded = False
        self._file_last_modified = None
        self._cache_valid = False
        self.logger.debug("Cache cleared")

    def _ensure_data_loaded(self):
        """
        Ensure data is loaded, only re-read file when necessary
        """
        # Check if cache is valid
        if self._cache_valid and self._data_loaded and not self._is_file_modified():
            self.logger.debug("Using cached data, skipping file read")
            return
        
        # Need to reload data
        self.logger.debug("Reloading data file")
        self._load_data_from_file()

    def _load_data_from_file(self):
        """
        Internal method to load data from file
        """
        if not self.path:
            self._clear_cache()
            return

        filename = os.path.join(self.path, "details.json")

        if not os.path.exists(filename):
            # If file doesn't exist, initialize empty data
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            self.non_zero_generated_episodes = []
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
            self.non_zero_generated_episodes = data.get("non_zero_generated_episodes_index", [])

            # Update cache status
            self._data_loaded = True
            self._cache_valid = True
            self._file_last_modified = self._get_file_modified_time()

            self.logger.info(f"Successfully loaded training data: {self.id}, size={self.size}, best_episode={self.best_episode}")
            self.logger.info(f"Episodes with non-zero generated_elements: {len(self.non_zero_generated_episodes)} episodes")

        except Exception as e:
            self.logger.error(f"Error reading data from {filename}: {e}")
            # If read failed, initialize empty data but mark cache as invalid
            self.detail_data = []
            self.size = 0
            self.best_episode = 0
            self.non_zero_generated_episodes = []
            self._data_loaded = True
            self._cache_valid = False  # Mark cache invalid, will retry next time

    def read_data(self):
        """
        Read json file and update [self.detail_data, self.size, self.best_episode, self.non_zero_generated_episodes]
        Keep this method for backward compatibility, but now uses smart caching
        :return:
        """
        with self._cache_lock:
            self._ensure_data_loaded()

    def get_episode_data(self, episode_index):
        """
        Return a single detail data, only return when is focused
        Get data by index, even if episode_number is large, return corresponding data
        Now uses smart caching to avoid repeated file reading
        :param episode_index: int index in non_zero_generated_episodes
        :return: single detail dict
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            # Use smart caching mechanism to ensure data is loaded
            self._ensure_data_loaded()

            if episode_index < 0 or episode_index >= len(self.detail_data):
                raise IndexError(f"Episode index {episode_index} out of range (0-{len(self.detail_data) - 1})")

            # Return corresponding detail data directly by index
            # Here episode_index is the index in non_zero_generated_episodes
            episode_data = self.detail_data[episode_index]

            # Record actual episode_number for debugging purposes
            actual_episode_number = episode_data.get("episode_number", "unknown")
            self.logger.debug(f"Get Episode index={episode_index}, actual episode_number={actual_episode_number}")

            return episode_data

    def get_episode_by_number(self, episode_number):
        """
        Get data by episode_number
        :param episode_number: int actual episode number
        :return: single detail dict or None if not found
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            self._ensure_data_loaded()

            # Search in episodes with non-zero generated_elements
            for detail in self.detail_data:
                if detail.get("episode_number") == episode_number:
                    return detail

            return None

    def get_episode_index_by_number(self, episode_number):
        """
        Get the index of episode_number in non_zero_generated_episodes
        :param episode_number: int actual episode number
        :return: int index position, return -1 if not found
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            self._ensure_data_loaded()

            # Search for the index corresponding to episode_number in detailed data
            for index, detail in enumerate(self.detail_data):
                if detail.get("episode_number") == episode_number:
                    return index

            return -1

    def update_data(self, new_details, new_best_episode):
        """
        Update current training data and save to file
        :param new_details: list of detail dict
        :param new_best_episode: int
        :return: None
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        try:
            # Call save function, will automatically filter non-zero step data
            save_episode_details(new_details, new_best_episode, self.path)

            with self._cache_lock:
                # Mark cache invalid, force reload
                self._cache_valid = False
                self._ensure_data_loaded()

            self.logger.info(f"Successfully updated training data: {self.id}, new_size={self.size}, new_best_episode={self.best_episode}")

        except Exception as e:
            self.logger.error(f"Error updating data for {self.id}: {e}")
            raise

    def get_best_episode_index(self):
        """
        Get the index of best episode in non_zero_generated_episodes
        :return: int index of best episode, return -1 if not found
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")

        with self._cache_lock:
            self._ensure_data_loaded()
            return self.get_episode_index_by_number(self.best_episode)

    def get_statistics(self):
        """
        Get training statistics information
        :return: dict containing statistics information
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
                "total_episodes": len(self.non_zero_generated_episodes),  # Only count episodes with non-zero generated_elements
                "non_zero_episodes": len(self.detail_data),
                "best_episode": self.best_episode,
                "best_episode_index": self.get_best_episode_index(),
                "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
                "episode_numbers": self.non_zero_generated_episodes.copy()
            }

    def current_focus_id(self):
        """
        Return the current focused training ID
        :return: str or None
        """
        return self.id

    def is_focused(self):
        """
        Check if focused on a training ID
        :return: bool
        """
        return self.focused

    def clear_focus(self):
        """
        Clear current focus state
        :return: None
        """
        with self._cache_lock:
            self.id = None
            self.path = None
            self.focused = False
            self._clear_cache()
            self.logger.info("Focus state cleared")

    def get_cache_status(self):
        """
        Get cache status information for debugging and monitoring
        :return: dict cache status information
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
        Force refresh cache, reload data from file
        :return: None
        """
        if not self.focused:
            raise RuntimeError("HistoryManager is not focused on any training ID")
        
        with self._cache_lock:
            self.logger.info(f"Force refresh cache: {self.id}")
            self._cache_valid = False
            self._ensure_data_loaded()

    def is_cache_valid(self):
        """
        Check if current cache is valid
        :return: bool whether cache is valid
        """
        if not self._cache_valid or not self._data_loaded:
            return False
        
        return not self._is_file_modified()
