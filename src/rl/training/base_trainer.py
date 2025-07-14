"""
基础训练器抽象类

定义了所有SAC实现共同的训练接口和行为
"""
import os
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from collections import deque
import numpy as np

from ..training_history_manager import TrainingHistoryManager
from ..config import load_config


class BaseTrainer(ABC):
    """
    训练器基础抽象类

    定义了训练流程的标准接口，具体的SAC实现只需要实现抽象方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础训练器

        Args:
            config: 配置字典
        """
        self.config = config if config is not None else load_config()

        # 训练状态管理
        self.is_training = False
        self.stop_event = threading.Event()
        self.training_thread = None

        # 训练统计
        self.training_stats = {
            'total_steps': 0,
            'episodes_completed': 0,
            'training_time': 0.0,
            'episode_rewards': [],
            'average_reward': 0.0,
            'latest_reward': 0.0
        }

        # 历史管理
        self.history_manager = TrainingHistoryManager(config=self.config)

        # 回调系统
        self.episode_callbacks: List[Callable] = []
        self.step_callbacks: List[Callable] = []

        # 用于前端展示的最近奖励队列
        self.recent_rewards = deque(maxlen=100)

        # 获取配置参数
        training_config = self.config.get("training", {})
        self.log_frequency = training_config.get("log_frequency", 1000)
        self.save_frequency = training_config.get("save_frequency", 10000)
        self.evaluation_frequency = training_config.get("evaluation_frequency", 5000)
        self.history_save_frequency = training_config.get("history_save_frequency", 10000)

    @abstractmethod
    def _initialize_agent(self):
        """初始化智能体（子类实现）"""
        pass

    @abstractmethod
    def _select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作（子类实现）"""
        pass

    @abstractmethod
    def _train_step(self, **kwargs) -> Dict[str, Any]:
        """执行一步训练（子类实现）"""
        pass

    @abstractmethod
    def _save_model(self, path: str):
        """保存模型（子类实现）"""
        pass

    @abstractmethod
    def _load_model(self, path: str):
        """加载模型（子类实现）"""
        pass

    def add_episode_callback(self, callback: Callable):
        """添加episode完成回调"""
        self.episode_callbacks.append(callback)

    def add_step_callback(self, callback: Callable):
        """添加训练步骤回调"""
        self.step_callbacks.append(callback)

    def _trigger_episode_callbacks(self, episode_data: Dict[str, Any]):
        """触发episode回调"""
        for callback in self.episode_callbacks:
            try:
                callback(episode_data)
            except Exception as e:
                print(f"Episode回调执行失败: {e}")

    def _trigger_step_callbacks(self, step_data: Dict[str, Any]):
        """触发步骤回调"""
        for callback in self.step_callbacks:
            try:
                callback(step_data)
            except Exception as e:
                print(f"Step回调执行失败: {e}")

    def _update_training_stats(self, episode_reward: float = None,
                               episode_length: int = None):
        """更新训练统计信息"""
        if episode_reward is not None:
            self.training_stats['episode_rewards'].append(episode_reward)
            self.recent_rewards.append(episode_reward)
            self.training_stats['latest_reward'] = episode_reward

            # 计算平均奖励
            if self.recent_rewards:
                self.training_stats['average_reward'] = np.mean(list(self.recent_rewards))

        if episode_length is not None:
            self.training_stats['episodes_completed'] += 1

    def _create_episode_data(self, episode: int, episode_reward: float,
                             episode_length: int, ref_info: Dict = None) -> Dict[str, Any]:
        """创建标准的episode数据格式"""
        current_time = time.time()
        current_timestep = self.training_stats['total_steps']

        episode_data = {
            'episode': episode,
            'timestep': current_timestep,
            'episode_reward': float(episode_reward),
            'episode_length': int(episode_length),
            'average_reward': self.training_stats['average_reward'],
            'timestamp': current_time,
            'training_id': self.history_manager.get_current_training_id()
        }

        # 添加参考信息
        if ref_info:
            episode_data['ref_info'] = ref_info

        return episode_data

    def _should_save_history(self, current_timestep: int, last_save_timestep: int) -> bool:
        """判断是否应该保存历史数据"""
        return current_timestep - last_save_timestep >= self.history_save_frequency

    def _should_log_progress(self, current_timestep: int, last_log_timestep: int) -> bool:
        """判断是否应该输出训练日志"""
        return current_timestep - last_log_timestep >= self.log_frequency

    def _should_evaluate(self, current_timestep: int, last_eval_timestep: int) -> bool:
        """判断是否应该进行评估"""
        return current_timestep - last_eval_timestep >= self.evaluation_frequency

    def _log_training_progress(self, episode: int, episode_reward: float):
        """输出训练进度日志"""
        training_id = self.history_manager.get_current_training_id()
        avg_reward = self.training_stats['average_reward']
        total_steps = self.training_stats['total_steps']

        print(f"Timestep {total_steps} Episode {episode} [{training_id}]: "
              f"最新奖励={episode_reward:.3f}, 平均奖励={avg_reward:.3f}")

    def start_training_async(self, **kwargs) -> str:
        """
        异步启动训练

        Returns:
            str: 训练ID
        """
        if self.is_training:
            raise RuntimeError("训练已在进行中")

        # 重置停止事件
        self.stop_event.clear()

        # 启动训练会话
        training_id = self.history_manager.start_training_session(
            mesh_name=kwargs.get('mesh_name', None),
            config_overrides={
                'max_timesteps': kwargs.get('max_timesteps', 100000),
                'max_steps': kwargs.get('max_steps', 1000),
                'batch_size': kwargs.get('batch_size', None),
                'history_save_frequency': self.history_save_frequency
            },
            description=kwargs.get('description', None)
        )

        # 启动训练线程 - 移除stop_event参数
        self.training_thread = threading.Thread(
            target=self._train_worker,
            kwargs=kwargs  # 直接传递kwargs，_train_worker会过滤掉stop_event
        )
        self.training_thread.start()

        return training_id

    def _train_worker(self, **kwargs):
        """训练工作线程"""
        try:
            self.is_training = True
            # 移除stop_event参数，因为它已经存储在self.stop_event中
            train_kwargs = {k: v for k, v in kwargs.items() if k != 'stop_event'}
            self.train(**train_kwargs)
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
        """获取训练状态"""
        return {
            'running': self.is_training,
            'status': 'running' if self.is_training else 'stopped',
            'stats': {
                'total_steps': self.training_stats['total_steps'],
                'episodes_completed': self.training_stats['episodes_completed'],
                'training_time': self.training_stats['training_time'],
                'latest_reward': self.training_stats['latest_reward'],
                'average_reward': self.training_stats['average_reward'],
                'total_episodes': len(self.training_stats['episode_rewards'])
            }
        }

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        执行训练主循环（子类实现）

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        pass
