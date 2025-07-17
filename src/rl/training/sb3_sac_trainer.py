import os
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.agent.sb3_sac_agent import SB3SACAgent
from src.rl.config import load_config
from src.utils.rl_ploter import plot_reward_change


class _EpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._current_episodes = 0
        self._current_timesteps = 0
        self.details = []
        self.data = {"r": [],
                     "l": [],
                     "mesh_data": [],
                     "boundary_vertices_data": [],
                     "last_ref_point": [],
                     "is_completed": []}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            detail = info.get("detail", {})
            self._current_episodes += 1
            self._current_timesteps += detail['l']
            self.details.append(detail)
            self.data['r'].append(detail['r'])
            self.data['l'].append(detail['l'])
            self.data['mesh_data'].append(detail['mesh_data'])
            self.data['boundary_vertices_data'].append(detail['boundary_vertices_data'])
            self.data['last_ref_point'].append(detail['last_ref_point'])
            self.data['last_ref_point'].append(detail['last_ref_point'])

        return True

    def get_last_detail(self):
        return self.details[-1]

    def get_details(self):
        return self.details

    def get_data(self, key):
        return [d.get(key) for d in self.details]

    def current_episodes(self):
        return self._current_episodes

    def current_timesteps(self):
        return self._current_timesteps

    def avg_reward_100(self):
        rewards = self.get_data('r')
        if not rewards:
            return 0.0
        return float(np.mean(rewards[-100:]))


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
        rewards = self._cb.get_data('r')
        return float(np.mean(rewards) if rewards else 0.0)

    @property
    def total_steps(self) -> int:
        """获取总步数"""
        return int(self.agent.num_timesteps)

    @property
    def total_episodes(self) -> int:
        """获取总episode数"""
        return self._cb.current_episodes()

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
        last_detail = self._cb.get_last_detail()

        return {
            "timesteps": self._cb.current_timesteps(),
            "episodes": self._cb.current_episodes(),
            "latest_reward": last_detail.get('r'),
            "latest_length": last_detail.get('l'),
            "latest_mesh": last_detail.get('mesh_data'),
            "latest_boundary": last_detail.get('boundary_vertices_data'),
            "latest_ref_point": last_detail.get('last_ref_point'),
            "avg_reward_100": self._cb.avg_reward_100(),
            "is_completed": last_detail.get('is_completed'),
        }

    def __getattr__(self, name):
        """代理到agent的属性访问"""
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def plot_reward(self, path):
        plot_reward_change(self._cb.get_data('r'), self._cb.get_data('l'), path)

    def best_mesh(self):
        """
        TODO: RETURN ONLY THE MESH DATA WITH THE BEST REWARD
        :return:
        """

