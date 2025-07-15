"""
基础训练器抽象类

定义了训练器的通用接口和基础功能
"""
import os
import time
import threading
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List

from src.rl.training_history_manager import TrainingHistoryManager


class BaseTrainer(ABC):
    """
    训练器基类

    定义了训练器的标准接口和通用功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础训练器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 训练状态
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 训练统计
        self.training_stats = {
            'total_steps': 0,
            'episodes_completed': 0,
            'training_time': 0.0,
            'latest_reward': 0.0,
            'average_reward': 0.0,
            'episode_rewards': []
        }

        # 回调函数
        self.episode_callbacks: List[Callable] = []
        self.step_callbacks: List[Callable] = []

        # 历史管理器
        self.history_manager = TrainingHistoryManager(config=self.config)

    def add_episode_callback(self, callback: Callable):
        """
        添加episode完成回调

        Args:
            callback: 回调函数，接收episode数据字典作为参数
        """
        self.episode_callbacks.append(callback)

    def add_step_callback(self, callback: Callable):
        """
        添加训练步骤回调

        Args:
            callback: 回调函数，接收step数据字典作为参数
        """
        self.step_callbacks.append(callback)

    def _trigger_episode_callbacks(self, episode_data: Dict[str, Any]):
        """触发episode回调"""
        for callback in self.episode_callbacks:
            try:
                callback(episode_data)
            except Exception as e:
                print(f"Episode回调执行失败: {e}")

    def _trigger_step_callbacks(self, step_data: Dict[str, Any]):
        """触发step回调"""
        for callback in self.step_callbacks:
            try:
                callback(step_data)
            except Exception as e:
                print(f"Step回调执行失败: {e}")

    def _update_training_stats(self, episode_reward: float, episode_length: int):
        """更新训练统计信息"""
        self.training_stats['episodes_completed'] += 1
        self.training_stats['latest_reward'] = episode_reward
        self.training_stats['episode_reward'] = episode_reward  # 添加此字段以匹配前端期望
        self.training_stats['episode_length'] = episode_length  # 添加此字段
        self.training_stats['episode_rewards'].append(episode_reward)

        # 计算平均奖励（最近100个episodes）
        recent_rewards = self.training_stats['episode_rewards'][-100:]
        self.training_stats['average_reward'] = sum(recent_rewards) / len(recent_rewards)

    def _create_episode_data(self, episode: int, episode_reward: float,
                             episode_length: int, ref_info: Optional[Dict] = None) -> Dict[str, Any]:
        """创建episode数据字典"""
        episode_data = {
            'episode': episode,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_steps': self.training_stats['total_steps'],
            'average_reward': self.training_stats['average_reward'],
            'timestamp': time.time()
        }

        # 添加参考点信息
        if ref_info:
            episode_data['reference_point_info'] = ref_info

        # 添加边界信息
        if hasattr(self, 'initial_boundary'):
            episode_data['boundary_vertices'] = len(self.initial_boundary.get_vertices())
            episode_data['boundary_vertices_data'] = self.initial_boundary.get_vertices()

        # 添加mesh数据（如果可用）
        if hasattr(self, 'env') and hasattr(self.env, 'get_mesh_data'):
            try:
                episode_data['mesh_data'] = self.env.get_mesh_data()
            except Exception as e:
                print(f"获取mesh数据失败: {e}")

        return episode_data

    def start_training_async(self, **kwargs) -> str:
        """
        异步启动训练

        Args:
            **kwargs: 训练参数，如max_timesteps, boundary_source等

        Returns:
            str: 训练ID
        """
        if self.is_training:
            raise RuntimeError("训练已在进行中")

        # 重置停止事件
        self.stop_event.clear()

        # 开始新的训练会话
        description = kwargs.get('description', '')
        boundary_source = kwargs.get('boundary_source', '')

        # 获取mesh名称（如果可用）
        mesh_name = None
        if boundary_source and isinstance(boundary_source, str):
            mesh_name = boundary_source

        training_id = self.history_manager.start_training_session(
            mesh_name=mesh_name,
            config_overrides={
                'max_timesteps': kwargs.get('max_timesteps', 100000),
                'boundary_source': boundary_source,
                'backend_type': getattr(self, 'backend_type', 'unknown')
            },
            description=description
        )

        # 启动训练线程，只传递train()方法需要的参数
        self.training_thread = threading.Thread(
            target=self._train_worker,
            kwargs=self._filter_train_kwargs(kwargs)
        )
        self.training_thread.start()

        return training_id

    def _filter_train_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤出train()方法需要的参数

        Args:
            kwargs: 原始参数字典

        Returns:
            Dict[str, Any]: 过滤后的参数字典
        """
        # 定义train()方法接受的参数列表
        valid_train_params = {
            'max_timesteps',
            'max_steps_per_episode',
            'save_frequency',
            'log_frequency',
            'evaluation_frequency'
        }

        # 只保留有效的训练参数
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in valid_train_params
        }

        return filtered_kwargs

    def _train_worker(self, **kwargs):
        """训练工作线程"""
        try:
            self.is_training = True
            self.train(**kwargs)
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False

    def stop_training(self):
        """停止训练"""
        if not self.is_training:
            print("训练未在进行中")
            return

        print("正在停止训练...")
        self.stop_event.set()

        # 等待训练线程结束
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10.0)

        self.is_training = False
        print("训练已停止")

    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态

        Returns:
            Dict[str, Any]: 训练状态信息，包含running, status, stats等字段
        """
        try:
            # 基础状态信息
            status_info = {
                "running": self.is_training,
                "status": "running" if self.is_training else "idle",
                "stats": None
            }

            # 如果正在训练或有统计数据，添加stats
            if self.training_stats['episodes_completed'] > 0 or self.is_training:
                stats = self.training_stats.copy()

                # 确保字段名与前端期望一致
                stats['episode'] = stats.get('episodes_completed', 0)
                stats['episode_reward'] = stats.get('latest_reward', 0.0)

                # 添加其他必需字段的默认值
                default_fields = {
                    'boundary_vertices': 0,
                    'buffer_size': 0,
                    'training_id': '',
                    'online_learning_mode': False,
                    'mesh_data': {},
                    'boundary_vertices_data': [],
                    'reference_point_info': {}
                }

                for field, default_value in default_fields.items():
                    if field not in stats:
                        stats[field] = default_value

                status_info["stats"] = stats

            return status_info

        except Exception as e:
            print(f"获取训练状态失败: {e}")
            return {
                "running": False,
                "status": "error",
                "stats": None,
                "error": str(e)
            }

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        执行训练主循环（子类实现）

        Args:
            **kwargs: 训练参数，通常包括：
                - max_timesteps (int): 最大训练步数
                - max_steps_per_episode (int): 每episode最大步数
                - save_frequency (int): 保存频率
                - log_frequency (int): 日志频率
                - evaluation_frequency (int): 评估频率

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        pass

    def _save_model(self, path: str):
        """
        保存模型（子类可覆盖）

        Args:
            path: 保存路径
        """
        print(f"基类不支持模型保存: {path}")

    def _load_model(self, path: str):
        """
        加载模型（子类可覆盖）

        Args:
            path: 模型路径
        """
        print(f"基类不支持模型加载: {path}")
