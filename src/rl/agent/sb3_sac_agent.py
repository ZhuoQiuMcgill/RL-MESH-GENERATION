import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
import json

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as th

SB3_AVAILABLE = True

from ..config import load_config


class HistoryCallback(BaseCallback):
    """
    自定义回调函数，用于收集训练过程中的数据并触发episode回调
    修复版本：正确处理episode统计和定期输出训练日志
    """

    def __init__(self, trainer_instance, verbose=0):
        super(HistoryCallback, self).__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0
        self.last_log_timestep = 0
        self.log_frequency = 1000  # 每1000步输出一次日志

    def _on_training_start(self) -> None:
        """训练开始时的初始化"""
        super()._on_training_start()
        print(f"SB3训练回调已启动，目标timesteps: {self.model.num_timesteps + self.model._total_timesteps}")

    def _on_step(self) -> bool:
        """在每个训练步骤后调用"""
        # 检查是否完成了一个episode
        dones = self.locals.get('dones', [False])

        # 处理episode结束
        if any(dones):
            self._on_episode_end()

        # 定期输出训练日志
        current_timestep = self.model.num_timesteps
        if current_timestep - self.last_log_timestep >= self.log_frequency:
            self._log_training_progress(current_timestep)
            self.last_log_timestep = current_timestep

        return True

    def _log_training_progress(self, current_timestep):
        """输出训练进度日志"""
        try:
            # 获取最近的episode统计
            if hasattr(self.trainer, 'recent_rewards') and self.trainer.recent_rewards:
                avg_reward = np.mean(list(self.trainer.recent_rewards))
                latest_reward = list(self.trainer.recent_rewards)[-1]
            else:
                avg_reward = 0.0
                latest_reward = 0.0

            # 获取训练ID
            training_id = ""
            if hasattr(self.trainer, 'history_manager'):
                training_id = f"[{self.trainer.history_manager.get_current_training_id()}]"

            print(f"SB3 Timestep {current_timestep} Episode {self.episode_count} {training_id}: "
                  f"最新奖励={latest_reward:.3f}, 平均奖励={avg_reward:.3f}")

        except Exception as e:
            print(f"SB3训练日志输出错误: {e}")

    def _on_episode_end(self):
        """episode结束时的处理"""
        self.episode_count += 1

        # 获取episode统计信息 - 直接访问环境
        episode_reward = 0.0
        episode_length = 0

        try:
            # 直接访问环境对象（SB3的环境通常是单环境）
            env = self.training_env

            # 尝试多种方式获取episode统计
            if hasattr(env, 'episode_reward'):
                episode_reward = float(env.episode_reward)
            elif hasattr(env, 'get_wrapper_attr'):
                episode_reward = float(env.get_wrapper_attr('episode_reward'))

            if hasattr(env, 'episode_length'):
                episode_length = int(env.episode_length)
            elif hasattr(env, 'get_wrapper_attr'):
                episode_length = int(env.get_wrapper_attr('episode_length'))

            print(f"SB3回调: Episode {self.episode_count} 结束，奖励={episode_reward:.3f}, 长度={episode_length}")

        except Exception as e:
            print(f"SB3回调获取episode统计失败: {e}")
            # 使用默认值
            episode_reward = 0.0
            episode_length = 0

        # 获取当前环境的参考信息
        ref_info = None
        try:
            env = self.training_env
            if hasattr(env, 'get_last_reference_info'):
                ref_info = env.get_last_reference_info()
        except Exception as e:
            print(f"SB3回调获取参考信息失败: {e}")
            ref_info = None

        # 强制更新trainer的recent_rewards用于平均奖励计算
        if hasattr(self.trainer, 'recent_rewards'):
            self.trainer.recent_rewards.append(episode_reward)

        # 创建episode数据
        episode_data = self.trainer._create_sb3_episode_data(
            episode=self.episode_count,
            episode_reward=float(episode_reward),
            episode_length=int(episode_length),
            ref_info=ref_info
        )

        # 触发回调
        self.trainer._trigger_episode_callbacks(episode_data)

        # 缓存到历史管理器
        if hasattr(self.trainer, 'history_manager'):
            self.trainer.history_manager.cache_episode_data(episode_data)


class SB3SACAgent:
    """
    使用Stable-Baselines3的SAC智能体包装器
    提供与自制SAC相同的接口，但内部使用SB3的实现
    """

    def __init__(self, env, device, config=None):
        """
        初始化SB3 SAC智能体

        Args:
            env: gymnasium环境
            device: 训练设备
            config: 配置字典
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 未安装。请运行: pip install stable-baselines3[extra]"
            )

        self.env = env
        self.device = device
        self.config = config if config is not None else load_config()

        # 获取SB3配置
        sb3_config = self.config.get("sb3_sac", {})

        # 设置网络架构
        policy_kwargs = dict(
            activation_fn=th.ReLU,
            net_arch=sb3_config.get("net_arch", [128, 128, 128])
        )

        # 创建SAC模型
        self.model = SAC(
            policy='MlpPolicy',
            env=env,
            learning_rate=float(sb3_config.get("learning_rate", 3e-4)),
            buffer_size=int(sb3_config.get("buffer_size", 1000000)),
            learning_starts=int(sb3_config.get("learning_starts", 10000)),
            batch_size=int(sb3_config.get("batch_size", 100)),
            tau=float(sb3_config.get("tau", 0.005)),
            gamma=float(sb3_config.get("gamma", 0.99)),
            train_freq=int(sb3_config.get("train_freq", 1)),
            gradient_steps=int(sb3_config.get("gradient_steps", 1)),
            policy_kwargs=policy_kwargs,
            seed=sb3_config.get("seed", None),
            device=str(device),
            verbose=sb3_config.get("verbose", 0)
        )

        # 用于与自制SAC接口兼容的属性
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # 训练统计
        self.training_stats = {
            'total_timesteps': 0,
            'episodes_completed': 0
        }

    def select_action(self, state, deterministic=True):
        """
        选择动作

        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略

        Returns:
            np.ndarray: 选择的动作
        """
        # 确保state是正确的形状
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def train(self, total_timesteps, callback=None):
        """
        训练模型

        Args:
            total_timesteps: 总训练步数
            callback: 回调函数

        Returns:
            dict: 训练统计信息
        """
        # 开始训练
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False  # 不重置timestep计数
        )

        # 更新统计信息
        self.training_stats['total_timesteps'] = self.model.num_timesteps

        return {
            'total_timesteps': self.training_stats['total_timesteps'],
            'episodes_completed': self.training_stats['episodes_completed']
        }

    def evaluate(self, eval_env, n_eval_episodes=10):
        """
        评估模型性能

        Args:
            eval_env: 评估环境
            n_eval_episodes: 评估episode数量

        Returns:
            tuple: (平均奖励, 奖励标准差)
        """
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            return_episode_rewards=False
        )

        return float(mean_reward), float(std_reward)

    def save(self, path):
        """
        保存模型

        Args:
            path: 保存路径
        """
        self.model.save(path)

        # 保存额外的统计信息
        stats_path = path + "_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2)

    def load(self, path):
        """
        加载模型

        Args:
            path: 模型路径
        """
        self.model = SAC.load(path, env=self.env)

        # 加载统计信息
        stats_path = path + "_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.training_stats = json.load(f)

    @classmethod
    def load_from_path(cls, path, env, device, config=None):
        """
        从路径加载预训练模型

        Args:
            path: 模型路径
            env: 环境
            device: 设备
            config: 配置

        Returns:
            SB3SACAgent: 加载的智能体
        """
        agent = cls(env, device, config)
        agent.load(path)
        return agent

    def get_model(self):
        """
        获取内部的SB3模型

        Returns:
            SAC: SB3的SAC模型
        """
        return self.model

    def set_env(self, env):
        """
        设置新的环境

        Args:
            env: 新环境
        """
        self.env = env
        self.model.set_env(env)
