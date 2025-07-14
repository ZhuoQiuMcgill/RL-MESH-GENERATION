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
    """

    def __init__(self, trainer_instance, verbose=0):
        super(HistoryCallback, self).__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_obs = None

    def _on_step(self) -> bool:
        """在每个训练步骤后调用"""
        # 获取当前环境信息
        if hasattr(self.training_env, 'get_attr'):
            # 处理VecEnv
            infos = self.training_env.get_attr('get_last_reference_info')
            ref_info = infos[0]() if infos and callable(infos[0]) else None
        else:
            # 处理单个环境
            ref_info = getattr(self.training_env, 'get_last_reference_info', lambda: None)()

        # 更新episode统计
        self.current_episode_length += 1

        # 检查是否完成了一个episode
        dones = self.locals.get('dones', [False])
        if any(dones):
            self._on_episode_end(ref_info)

        return True

    def _on_episode_end(self, ref_info):
        """episode结束时的处理"""
        self.episode_count += 1

        # 获取episode奖励
        if hasattr(self.training_env, 'get_attr'):
            # VecEnv
            episode_rewards = self.training_env.get_attr('episode_reward')
            episode_reward = episode_rewards[0] if episode_rewards else 0
        else:
            # 单个环境
            episode_reward = getattr(self.training_env, 'episode_reward', 0)

        # 创建episode数据
        episode_data = self.trainer._create_sb3_episode_data(
            episode=self.episode_count,
            episode_reward=float(episode_reward),
            episode_length=self.current_episode_length,
            ref_info=ref_info
        )

        # 触发回调
        self.trainer._trigger_episode_callbacks(episode_data)

        # 缓存到历史管理器
        if hasattr(self.trainer, 'history_manager'):
            self.trainer.history_manager.cache_episode_data(episode_data)

        # 重置episode统计
        self.current_episode_length = 0


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
            activation_fn=th.nn.ReLU,
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
        self.training_stats['total_timesteps'] += total_timesteps

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
