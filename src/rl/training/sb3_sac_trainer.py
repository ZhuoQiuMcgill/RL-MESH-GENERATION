import os
from typing import Any, Dict
from math import inf

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.agent.sb3_sac_agent import SB3SACAgent
from src.rl.config import load_config
from src.utils.rl_ploter import plot_reward_change, plot_training_metrics, plot_action_distribution, plot_action_reward_distribution
from src.rl.training.history_manager import save_episode_details


class _EpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._current_episode = 0
        self._current_timesteps = 0
        self.details = []
        self.data = {"r": [],
                     "l": [],
                     "mesh_data": [],
                     "boundary_vertices_data": [],
                     "last_ref_point": [],
                     "is_completed": [],
                     "generated_elements": [],
                     "action_counts": []}
        self._best_reward = -inf
        self._best_episode = 0
        
        # 新增：训练过程中的loss和alpha数据收集
        self.training_metrics = {
            "actor_losses": [],
            "critic_losses": [],
            "alphas": [],
            "timesteps": []  # 记录对应的timestep
        }
        
        # 当前最新的metrics值（用于实时显示）
        self._latest_actor_loss = 0.0
        self._latest_critic_loss = 0.0
        self._latest_alpha = 0.0

    def _on_step(self) -> bool:
        # 收集训练过程中的loss和alpha数据
        self._collect_training_metrics()
        
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            detail = info.get("detail", {})
            detail["episode_number"] = self._current_episode
            self.details.append(detail)
            self.data['r'].append(detail['r'])
            self.data['l'].append(detail['l'])
            self.data['mesh_data'].append(detail['mesh_data'])
            self.data['boundary_vertices_data'].append(detail['boundary_vertices_data'])
            self.data['last_ref_point'].append(detail['last_ref_point'])
            self.data['is_completed'].append(detail['is_completed'])
            self.data['generated_elements'].append(detail['generated_elements'])
            self.data['action_counts'].append(detail.get('action_count', {}))

            if detail['r'] > self._best_reward:
                self._best_reward = detail['r']
                self._best_episode = self._current_episode

            self._current_episode += 1
            self._current_timesteps += detail['l']
            if not detail['is_completed']:
                self._current_timesteps += 1

        return True

    def _collect_training_metrics(self):
        """
        收集训练过程中的loss和alpha数据
        """
        try:
            if self.model and hasattr(self.model, 'logger') and self.model.logger:
                # 获取logger中记录的值
                name_to_value = getattr(self.model.logger, 'name_to_value', {})
                
                if name_to_value:
                    current_timestep = self.model.num_timesteps
                    
                    # 获取actor loss
                    if 'train/actor_loss' in name_to_value:
                        actor_loss = name_to_value['train/actor_loss']
                        self._latest_actor_loss = actor_loss
                        self.training_metrics["actor_losses"].append(actor_loss)
                        self.training_metrics["timesteps"].append(current_timestep)
                    
                    # 获取critic loss
                    if 'train/critic_loss' in name_to_value:
                        critic_loss = name_to_value['train/critic_loss']
                        self._latest_critic_loss = critic_loss
                        self.training_metrics["critic_losses"].append(critic_loss)
                    
                    # 获取alpha (entropy coefficient)
                    if 'train/ent_coef' in name_to_value:
                        alpha = name_to_value['train/ent_coef']
                        self._latest_alpha = alpha
                        self.training_metrics["alphas"].append(alpha)
            
            # 如果logger中没有数据，尝试直接从policy中获取alpha
            if self._latest_alpha == 0.0 and self.model and hasattr(self.model, 'policy'):
                try:
                    if hasattr(self.model.policy, 'log_ent_coef'):
                        import torch
                        alpha = torch.exp(self.model.policy.log_ent_coef).item()
                        self._latest_alpha = alpha
                        if len(self.training_metrics["alphas"]) == 0 or self.training_metrics["alphas"][-1] != alpha:
                            self.training_metrics["alphas"].append(alpha)
                            if len(self.training_metrics["timesteps"]) == 0:
                                self.training_metrics["timesteps"].append(self.model.num_timesteps)
                    elif hasattr(self.model.policy, 'ent_coef'):
                        alpha = float(self.model.policy.ent_coef)
                        self._latest_alpha = alpha
                        if len(self.training_metrics["alphas"]) == 0 or self.training_metrics["alphas"][-1] != alpha:
                            self.training_metrics["alphas"].append(alpha)
                            if len(self.training_metrics["timesteps"]) == 0:
                                self.training_metrics["timesteps"].append(self.model.num_timesteps)
                except Exception as e:
                    pass  # 静默处理，避免影响训练
        except Exception as e:
            # 静默处理异常，避免影响训练过程
            pass

    def get_latest_training_metrics(self):
        """
        获取最新的训练指标，用于实时显示
        
        Returns:
            dict: 包含最新的actor_loss, critic_loss, alpha
        """
        return {
            "actor_loss": self._latest_actor_loss,
            "critic_loss": self._latest_critic_loss,
            "alpha": self._latest_alpha
        }

    def get_training_metrics(self):
        """
        获取完整的训练指标历史数据
        
        Returns:
            dict: 包含所有训练指标的历史数据
        """
        return self.training_metrics.copy()

    def get_last_detail(self):
        return self.details[-1]

    def get_details(self):
        return self.details

    def get_data(self, key):
        return self.data.get(key, [])

    def current_episodes(self):
        return self._current_episode

    def current_timesteps(self):
        return self._current_timesteps

    def avg_reward_100(self):
        rewards = self.get_data('r')
        if not rewards:
            return 0.0
        return float(np.mean(rewards[-100:]))

    def current_best_episode(self):
        return self._best_episode

    def get_non_zero_generated_details(self):
        non_zero_generated_details = []
        for detail in self.details:
            if detail.get("generated_elements", 0) > 0:
                non_zero_generated_details.append(detail)
        return non_zero_generated_details


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
        
        # 获取最新的训练指标
        latest_metrics = self._cb.get_latest_training_metrics()

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
            # 新增：训练指标
            "recent_actor_loss": latest_metrics.get("actor_loss", 0.0),
            "recent_critic_loss": latest_metrics.get("critic_loss", 0.0),
            "current_alpha": latest_metrics.get("alpha", 0.0),
        }

    def __getattr__(self, name):
        """代理到agent的属性访问"""
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def plot_reward(self, path):
        plot_reward_change(self._cb.get_data('r'), self._cb.get_data('l'), path)

    def plot_training_metrics(self, save_dir: str):
        """
        绘制训练过程中的loss和alpha图表
        
        Args:
            save_dir: 保存目录路径
            
        Returns:
            dict: 包含生成的图片路径
        """
        # 获取训练指标数据
        metrics = self._cb.get_training_metrics()
        
        actor_losses = metrics.get("actor_losses", [])
        critic_losses = metrics.get("critic_losses", [])
        alphas = metrics.get("alphas", [])
        timesteps = metrics.get("timesteps", [])
        
        # 生成图表
        saved_plots = plot_training_metrics(
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            alphas=alphas,
            timesteps=timesteps,
            save_dir=save_dir
        )
        
        return saved_plots

    def plot_action_distribution(self, save_path: str) -> str:
        """
        绘制动作分布图，显示每种动作类型的valid/invalid统计
        
        Args:
            save_path: 图表保存路径
            
        Returns:
            str: 最终保存的文件路径
        """
        action_counts_list = self._cb.get_data('action_counts')
        return plot_action_distribution(action_counts_list, save_path)

    def plot_action_reward_distribution(self, save_path: str) -> str:
        """
        绘制动作奖励分布图，显示每种动作类型的奖励分布情况
        
        Args:
            save_path: 图表保存路径
            
        Returns:
            str: 最终保存的文件路径
        """
        action_counts_list = self._cb.get_data('action_counts')
        return plot_action_reward_distribution(action_counts_list, save_path)

    def save_history(self, path):
        details = self._cb.get_non_zero_generated_details()
        best_episode_number = self._cb.current_best_episode()
        best_episode_index = 0
        for i, detail in enumerate(details):
            if detail.get("episode_number") == best_episode_number:
                best_episode_index = i
                break
        save_episode_details(details, best_episode_index, path)
