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


class SB3SACAgent:
    """
    使用Stable-Baselines3的SAC智能体包装器
    提供与自制SAC相同的接口，但内部使用SB3的实现
    """

    def __init__(self, env, device="cuda", config=None):
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
            train_freq=sb3_config.get("train_freq", 1),
            gradient_steps=sb3_config.get("gradient_steps", 1),
            policy_kwargs=policy_kwargs,
            verbose=sb3_config.get("verbose", 0),
            seed=sb3_config.get("seed", None),
            device=device
        )

    def predict(self, state, deterministic=True):
        """
        预测动作

        Args:
            state: 状态
            deterministic: 是否使用确定性策略

        Returns:
            action: 预测的动作
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, path):
        """
        保存模型

        Args:
            path: 保存路径
        """
        self.model.save(path)

    def load(self, path):
        """
        加载模型

        Args:
            path: 模型路径
        """
        self.model = SAC.load(path, env=self.env)

    def learn(self, total_timesteps, callback=None):
        """
        开始学习

        Args:
            total_timesteps: 总时间步数
            callback: 回调函数

        Returns:
            学习后的模型
        """
        return self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def get_parameters(self):
        """
        获取模型参数

        Returns:
            dict: 模型参数字典
        """
        return {
            'policy': self.model.policy.state_dict() if self.model.policy else None,
            'learning_rate': self.model.learning_rate,
            'gamma': self.model.gamma,
            'tau': self.model.tau,
        }

    def set_parameters(self, parameters):
        """
        设置模型参数

        Args:
            parameters: 参数字典
        """
        if 'policy' in parameters and parameters['policy'] is not None:
            self.model.policy.load_state_dict(parameters['policy'])

    def evaluate(self, eval_env, n_eval_episodes=10, deterministic=True):
        """
        评估智能体

        Args:
            eval_env: 评估环境
            n_eval_episodes: 评估episodes数量
            deterministic: 是否使用确定性策略

        Returns:
            tuple: (平均奖励, 奖励标准差)
        """
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=False
        )
        return mean_reward, std_reward

    def get_action_probabilities(self, state):
        """
        获取动作概率分布

        Args:
            state: 状态

        Returns:
            动作概率分布
        """
        # SB3的SAC是连续动作空间，返回动作分布的参数
        obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions, log_probs = self.model.policy.actor.get_action_log_prob(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @property
    def replay_buffer(self):
        """获取经验回放缓冲区"""
        return self.model.replay_buffer if hasattr(self.model, 'replay_buffer') else None

    @property
    def num_timesteps(self):
        """获取当前时间步数"""
        return self.model.num_timesteps

    def get_env(self):
        """获取环境"""
        return self.env

    def set_env(self, env):
        """设置环境"""
        self.env = env
        self.model.set_env(env)

    def __getattr__(self, name):
        """代理到内部SAC模型的属性访问"""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
