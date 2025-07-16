import os
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.agent.sb3_sac_agent import SB3SACAgent
from src.rl.config import load_config


class _EpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.infos: List[Dict[str, Any]] = []
        self.meshes = []
        self.boundaries = []
        self.ref_points = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            ep_stats = info.get("episode", {})
            self.rewards.append(float(ep_stats.get("r", 0.0)))
            self.lengths.append(int(ep_stats.get("l", 0)))

            geom = info.get("geometry", {})
            self.meshes.append(geom.get("mesh_data"))
            self.boundaries.append(geom.get("boundary_vertices_data"))
            self.ref_points.append(geom.get("last_ref_point"))

            self.infos.append(info)

        return True


class SB3SACTrainer:
    """
    Stable-Baselines3 SAC训练器

    支持从config.yaml读取所有训练参数，确保参数配置的一致性。
    """

    def __init__(self, env, device: str = "cuda", config=None):
        """
        初始化SB3 SAC训练器

        Args:
            env: 训练环境
            device: 训练设备
            config: 配置字典，如果为None则从config.yaml加载
        """
        self.env = env
        self.device = device
        self.config = config if config is not None else load_config()

        # 创建SAC智能体，传入完整配置
        self.agent = SB3SACAgent(env, device, self.config)

        # 创建回调函数
        self._cb = _EpisodeCallback()

    def train(self, total_timesteps: int):
        """
        开始训练

        Args:
            total_timesteps: 总训练时间步数
        """
        self.agent.learn(total_timesteps=total_timesteps, callback=self._cb)

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)

    def load(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        self.agent.load(path)

    def evaluate(self, eval_env=None, n_eval_episodes: int = 10):
        """
        评估智能体

        Args:
            eval_env: 评估环境
            n_eval_episodes: 评估episode数量

        Returns:
            tuple: (平均奖励, 奖励标准差)
        """
        env = eval_env if eval_env is not None else self.env
        return self.agent.evaluate(env, n_eval_episodes=n_eval_episodes)

    def get_policy_optimizers(self):
        """
        获取策略的优化器列表 - 修复版本

        Returns:
            list: 优化器列表 [actor_optimizer, critic_optimizer, ent_coef_optimizer]
        """
        optimizers = []
        try:
            model = self.agent.model
            policy = model.policy

            # SB3 SAC的优化器通常存储在policy中
            if hasattr(policy, 'actor') and hasattr(policy.actor, 'optimizer'):
                optimizers.append(policy.actor.optimizer)
            elif hasattr(policy, 'actor_optimizer'):
                optimizers.append(policy.actor_optimizer)

            if hasattr(policy, 'critic') and hasattr(policy.critic, 'optimizer'):
                optimizers.append(policy.critic.optimizer)
            elif hasattr(policy, 'critic_optimizer'):
                optimizers.append(policy.critic_optimizer)

            if hasattr(policy, 'ent_coef_optimizer'):
                optimizers.append(policy.ent_coef_optimizer)

        except Exception as e:
            print(f"获取优化器时出错: {e}")

        return optimizers

    @property
    def model(self):
        """获取内部SB3模型"""
        return self.agent.model

    @property
    def average_reward(self) -> float:
        """获取平均奖励"""
        return float(np.mean(self._cb.rewards)) if self._cb.rewards else 0.0

    @property
    def total_steps(self) -> int:
        """获取总步数"""
        return int(self.agent.num_timesteps)

    @property
    def total_episodes(self) -> int:
        """获取总episode数"""
        return len(self._cb.rewards)

    def get_episode_infos(self) -> List[Dict[str, Any]]:
        """获取episode信息列表"""
        return list(self._cb.infos)

    def summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            "avg_reward": self.average_reward,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        获取训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        latest_reward = self._cb.rewards[-1] if self._cb.rewards else None
        latest_length = self._cb.lengths[-1] if self._cb.lengths else None

        latest_mesh = self._cb.meshes[-1] if self._cb.meshes else None
        latest_boundary = self._cb.boundaries[-1] if self._cb.boundaries else None
        latest_ref_point = self._cb.ref_points[-1] if self._cb.ref_points else None

        latest_info = self._cb.infos[-1] if self._cb.infos else None

        return {
            "timesteps": self.total_steps,
            "episodes": self.total_episodes,
            "latest_reward": latest_reward,
            "latest_length": latest_length,
            "latest_mesh": latest_mesh,
            "latest_boundary": latest_boundary,
            "latest_ref_point": latest_ref_point,
            "avg_reward_100": float(np.mean(self._cb.rewards[-100:])) if self._cb.rewards else 0.0,
            "latest_info": latest_info,
        }

    def __getattr__(self, name):
        """代理到agent的属性访问"""
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
