import os
from typing import Any, Dict
from math import inf

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.agent.sb3_sac_agent import SB3SACAgent
from src.rl.config import load_config
from src.utils.rl_ploter import plot_reward_change, plot_training_metrics, plot_action_distribution, \
    plot_action_reward_distribution, plot_avg_element_quality
from src.rl.training.history_manager import save_episode_details


class _EpisodeCallback(BaseCallback):
    def __init__(self, evaluation_frequency=10, n_eval_episodes=10, training_session_dir=None, stop_event=None, enable_verbose_logging=False, require_completed_for_save=True):
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
                     "action_counts": [],
                     "avg_element_quality": []}
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

        # 新增：自动评估相关参数
        self.evaluation_frequency = evaluation_frequency
        self.n_eval_episodes = n_eval_episodes
        self.training_session_dir = training_session_dir
        self.stop_event = stop_event  # 停止事件

        # 评估状态跟踪
        self._eval_count = 0
        self._last_eval_reward = 0.0
        self._best_eval_reward = -inf
        self._eval_rewards_history = []
        self._best_model_path = None
        
        # 日志控制开关
        self.enable_verbose_logging = enable_verbose_logging
        
        # 保存条件控制开关
        self.require_completed_for_save = require_completed_for_save

    def _on_step(self) -> bool:
        # 检查停止事件
        if self.stop_event and self.stop_event.is_set():
            import logging
            logger = logging.getLogger(__name__)
            logger.info("检测到停止信号，终止训练")
            return False  # 返回False停止训练

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
            self.data['avg_element_quality'].append(detail.get('avg_element_quality', 0.0))

            if detail['r'] > self._best_reward:
                self._best_reward = detail['r']
                self._best_episode = self._current_episode

            self._current_episode += 1
            self._current_timesteps += detail['l']
            if not detail['is_completed']:
                self._current_timesteps += 1

            # 检查是否需要进行自动评估（只有在训练未停止时才进行）
            if (self.evaluation_frequency > 0 and
                    self._current_episode % self.evaluation_frequency == 0 and
                    (not self.stop_event or not self.stop_event.is_set())):
                self._perform_evaluation()

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

    def _perform_evaluation(self):
        """
        执行自动评估并保存最佳模型
        """
        try:
            import logging
            logger = logging.getLogger(__name__)

            # 检查停止事件
            if self.stop_event and self.stop_event.is_set():
                logger.info("检测到停止信号，跳过评估")
                return

            if self.enable_verbose_logging:
                logger.info(f"开始第{self._eval_count + 1}次自动评估 (Episode {self._current_episode})")

            # 使用当前环境进行评估
            eval_env = self.model.env
            if eval_env is None:
                logger.warning("评估环境不可用，跳过评估")
                return

            # 执行评估
            episode_rewards = []
            completed_episodes = []  # 记录每个episode是否完成
            
            for i in range(self.n_eval_episodes):
                # 在每个评估episode开始前检查停止事件
                if self.stop_event and self.stop_event.is_set():
                    logger.info("检测到停止信号，终止评估")
                    return

                # 兼容不同版本的gymnasium/gym API
                reset_result = eval_env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result

                total_reward = 0.0
                done = False
                episode_completed = False  # 追踪整个episode的完成状态

                while not done:
                    # 在evaluation步骤中也检查停止事件
                    if self.stop_event and self.stop_event.is_set():
                        logger.info("检测到停止信号，终止当前评估episode")
                        return

                    action, _ = self.model.predict(obs, deterministic=True)

                    # 兼容不同版本的step返回值
                    step_result = eval_env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:
                        logger.error(f"意外的step返回值数量: {len(step_result)}")
                        break

                    total_reward += float(reward)
                    
                    # 检查episode是否在任何时刻完成任务（修复：追踪整个过程）
                    if info and 'detail' in info:
                        current_completed = info['detail'].get('is_completed', False)
                        # 一旦检测到完成状态，就保持为True（不会被后续的False覆盖）
                        if current_completed:
                            episode_completed = True
                    
                    # 如果episode因为任务完成而终止，立即停止（避免继续执行无效动作）
                    if done and episode_completed:
                        break

                episode_rewards.append(total_reward)
                completed_episodes.append(episode_completed)
                if self.enable_verbose_logging:
                    logger.debug(f"评估Episode {i + 1}/{self.n_eval_episodes}: {total_reward:.3f}, 完成: {episode_completed}")

            # 计算平均奖励
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            
            # 统计完成情况
            completed_count = sum(completed_episodes)
            completion_rate = completed_count / len(completed_episodes) if completed_episodes else 0.0

            # 更新评估统计
            self._eval_count += 1
            self._last_eval_reward = mean_reward
            self._eval_rewards_history.append({
                'episode': self._current_episode,
                'eval_count': self._eval_count,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'rewards': episode_rewards.copy(),
                'completed_episodes': completed_episodes.copy(),
                'completed_count': completed_count,
                'completion_rate': completion_rate
            })

            if self.enable_verbose_logging:
                logger.info(f"评估完成: 平均奖励={mean_reward:.3f}±{std_reward:.3f}, 完成率={completion_rate:.2%} ({completed_count}/{len(completed_episodes)})")
                # 调试日志：显示每个episode的完成状态
                logger.debug(f"Episode完成状态: {completed_episodes}")
                logger.debug(f"Episode奖励: {[f'{r:.2f}' for r in episode_rewards]}")
            
            # 检查是否满足保存条件：根据配置决定是否要求完成任务
            should_save = False
            save_reason = ""
            
            # 检查完成条件（如果启用了要求完成的配置）
            completion_check_passed = True
            if self.require_completed_for_save and completed_count == 0:
                completion_check_passed = False
                save_reason = "没有episode完成任务，不保存模型"
            
            # 检查奖励提升条件
            if completion_check_passed:
                if mean_reward <= self._best_eval_reward:
                    save_reason = f"奖励未提升 ({mean_reward:.3f} <= {self._best_eval_reward:.3f})，不保存模型"
                else:
                    should_save = True
                    if self.require_completed_for_save:
                        save_reason = f"发现更好的模型! 奖励提升: {self._best_eval_reward:.3f} -> {mean_reward:.3f}, 完成率: {completion_rate:.2%}"
                    else:
                        save_reason = f"发现更好的模型! 奖励提升: {self._best_eval_reward:.3f} -> {mean_reward:.3f} (不要求完成)"
            
            if should_save:
                self._best_eval_reward = mean_reward
                logger.info(f"🎉 {save_reason}")  # 保存成功总是记录

                # 保存最佳模型
                if self.training_session_dir:
                    self._save_best_model(mean_reward, completed_count, completion_rate)
            else:
                if self.enable_verbose_logging:
                    logger.info(save_reason)  # 不保存的原因只在verbose模式下记录

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"自动评估过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_best_model(self, reward, completed_count=0, completion_rate=0.0):
        """
        保存最佳模型
        
        Args:
            reward: 当前评估奖励
            completed_count: 完成任务的episode数量
            completion_rate: 任务完成率
        """
        try:
            import os
            import logging
            import glob
            logger = logging.getLogger(__name__)

            if not self.training_session_dir:
                logger.warning("训练会话目录未设置，无法保存最佳模型")
                return

            # 创建最佳模型目录
            best_model_dir = os.path.join(self.training_session_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)

            # 删除之前的最佳模型文件
            if self._best_model_path:
                try:
                    # 删除之前的SB3模型文件
                    if 'sb3_path' in self._best_model_path and os.path.exists(self._best_model_path['sb3_path']):
                        os.remove(self._best_model_path['sb3_path'])
                        if self.enable_verbose_logging:
                            logger.info(f"已删除旧的SB3模型: {self._best_model_path['sb3_path']}")

                    # 删除之前的Checkpoint文件
                    if 'checkpoint_path' in self._best_model_path and os.path.exists(
                            self._best_model_path['checkpoint_path']):
                        os.remove(self._best_model_path['checkpoint_path'])
                        if self.enable_verbose_logging:
                            logger.info(f"已删除旧的Checkpoint: {self._best_model_path['checkpoint_path']}")

                    # 删除之前的评估信息文件
                    if 'eval_info_path' in self._best_model_path and os.path.exists(
                            self._best_model_path['eval_info_path']):
                        os.remove(self._best_model_path['eval_info_path'])
                        if self.enable_verbose_logging:
                            logger.info(f"已删除旧的评估信息: {self._best_model_path['eval_info_path']}")

                except Exception as e:
                    logger.warning(f"删除旧的最佳模型文件时发生错误: {e}")
            else:
                # 如果没有记录的最佳模型路径，尝试清理目录中的所有最佳模型文件
                try:
                    for pattern in ["best_model_*.zip", "best_model_*.pth", "best_model_*_eval_info.json"]:
                        for file_path in glob.glob(os.path.join(best_model_dir, pattern)):
                            os.remove(file_path)
                            if self.enable_verbose_logging:
                                logger.info(f"已删除旧的最佳模型文件: {file_path}")
                except Exception as e:
                    logger.warning(f"清理旧的最佳模型文件时发生错误: {e}")

            # 生成新的文件名，包含完成信息
            timestamp = self._current_episode
            best_model_name = f"best_model_ep{timestamp}_reward{reward:.3f}_completed{completed_count}_rate{completion_rate:.0%}"

            # 保存SB3模型
            sb3_path = os.path.join(best_model_dir, f"{best_model_name}.zip")
            self.model.save(sb3_path)

            # 保存PyTorch模型参数
            import torch
            checkpoint = {
                'episode': self._current_episode,
                'eval_count': self._eval_count,
                'eval_reward': reward,
                'training_timesteps': self.model.num_timesteps,
                'learning_rate': self.model.learning_rate,
                'gamma': self.model.gamma,
                'tau': self.model.tau,
                'actor_state_dict': self.model.policy.actor.state_dict(),
                'critic_state_dict': self.model.policy.critic.state_dict(),
                'critic_target_state_dict': self.model.policy.critic_target.state_dict(),
            }

            # 保存温度参数
            if hasattr(self.model.policy, 'log_ent_coef'):
                checkpoint['log_ent_coef'] = self.model.policy.log_ent_coef.data.clone()
            elif hasattr(self.model.policy, 'ent_coef'):
                checkpoint['ent_coef'] = self.model.policy.ent_coef

            # 保存checkpoint
            checkpoint_path = os.path.join(best_model_dir, f"{best_model_name}.pth")
            torch.save(checkpoint, checkpoint_path)

            # 保存评估信息
            import json
            eval_info = {
                'episode': self._current_episode,
                'eval_count': self._eval_count,
                'eval_reward': float(reward),
                'best_eval_reward': float(self._best_eval_reward),
                'completed_count': completed_count,
                'completion_rate': float(completion_rate),
                'n_eval_episodes': self.n_eval_episodes,
                'evaluation_frequency': self.evaluation_frequency,
                'timestamp': self.model.num_timesteps,
                'eval_history': self._eval_rewards_history,
                'save_criteria': {
                    'requires_completion': True,
                    'requires_better_reward': True,
                    'description': 'Model saved only if episodes completed tasks AND reward improved'
                }
            }

            eval_info_path = os.path.join(best_model_dir, f"{best_model_name}_eval_info.json")
            with open(eval_info_path, 'w', encoding='utf-8') as f:
                json.dump(eval_info, f, indent=2, default=str)

            # 更新最佳模型路径
            self._best_model_path = {
                'sb3_path': sb3_path,
                'checkpoint_path': checkpoint_path,
                'eval_info_path': eval_info_path,
                'reward': reward,
                'episode': self._current_episode
            }

            logger.info(f"✓ 最佳模型已保存 (奖励: {reward:.3f}, 完成: {completed_count}/{self.n_eval_episodes}, 完成率: {completion_rate:.1%})")
            if self.enable_verbose_logging:
                logger.info(f"  - SB3模型: {sb3_path}")
                logger.info(f"  - Checkpoint: {checkpoint_path}")
                logger.info(f"  - 评估信息: {eval_info_path}")

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"保存最佳模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

    def get_evaluation_info(self):
        """
        获取评估相关信息
        
        Returns:
            dict: 包含评估统计信息
        """
        return {
            'eval_count': self._eval_count,
            'last_eval_reward': self._last_eval_reward,
            'best_eval_reward': self._best_eval_reward,
            'evaluation_frequency': self.evaluation_frequency,
            'n_eval_episodes': self.n_eval_episodes,
            'eval_history': self._eval_rewards_history.copy(),
            'best_model_path': self._best_model_path
        }

    def get_last_detail(self):
        if self.details:
            return self.details[-1]
        else:
            # 返回默认值以避免异常
            return {
                'r': 0.0,
                'l': 0,
                'mesh_data': {},
                'boundary_vertices_data': [],
                'last_ref_point': None,
                'is_completed': False,
                'avg_element_quality': 0.0
            }

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

    def __init__(self, env, device: str = "cuda", config=None, training_session_dir=None, stop_event=None, enable_verbose_logging=False):
        """
        初始化SB3 SAC训练器

        Args:
            env: 训练环境
            device: 训练设备
            config: 配置字典，如果为None则从config.yaml加载
            training_session_dir: 训练会话目录，用于保存最佳模型
            stop_event: 停止事件，用于停止训练
            enable_verbose_logging: 是否启用详细日志输出
        """
        self.env = env
        self.device = device
        self.config = config if config is not None else load_config()
        self.training_session_dir = training_session_dir
        self.stop_event = stop_event
        self.enable_verbose_logging = enable_verbose_logging

        # 创建SAC智能体，传入完整配置
        self.agent = SB3SACAgent(env, device, self.config)

        # 获取评估配置
        training_config = self.config.get("training", {})
        evaluation_frequency = training_config.get("evaluation_frequency", 10)
        n_eval_episodes = training_config.get("n_eval_episodes", 10)
        require_completed_for_save = training_config.get("require_completed_for_save", True)

        # 创建回调函数，传入评估参数和停止事件
        self._cb = _EpisodeCallback(
            evaluation_frequency=evaluation_frequency,
            n_eval_episodes=n_eval_episodes,
            training_session_dir=training_session_dir,
            stop_event=stop_event,
            enable_verbose_logging=enable_verbose_logging,
            require_completed_for_save=require_completed_for_save
        )

    def set_verbose_logging(self, enable: bool):
        """
        动态设置详细日志开关
        
        Args:
            enable: 是否启用详细日志
        """
        self.enable_verbose_logging = enable
        self._cb.enable_verbose_logging = enable

    def set_require_completed_for_save(self, require: bool):
        """
        动态设置是否要求evaluation中必须有completed episode才能保存模型
        
        Args:
            require: 是否要求completed episode
        """
        self._cb.require_completed_for_save = require

    def train(self, total_timesteps: int):
        """
        开始训练

        Args:
            total_timesteps: 总训练时间步数
        """
        try:
            # 在开始训练前检查停止事件
            if self.stop_event and self.stop_event.is_set():
                import logging
                logger = logging.getLogger(__name__)
                logger.info("训练开始前检测到停止信号，取消训练")
                return

            self.agent.learn(total_timesteps=total_timesteps, callback=self._cb)

        except KeyboardInterrupt:
            # 处理键盘中断
            import logging
            logger = logging.getLogger(__name__)
            logger.info("检测到键盘中断，停止训练")
        except Exception as e:
            # 检查是否因为停止事件而异常
            if self.stop_event and self.stop_event.is_set():
                import logging
                logger = logging.getLogger(__name__)
                logger.info("训练因停止信号而终止")
            else:
                # 重新抛出其他异常
                raise

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

    def _sanitize_value(self, value):
        """
        清理数值，确保JSON序列化兼容
        """
        import math
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return 0.0
            return float(value)
        return value

    def get_status(self) -> Dict[str, Any]:
        """
        获取训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        last_detail = self._cb.get_last_detail()

        # 获取最新的训练指标
        latest_metrics = self._cb.get_latest_training_metrics()

        # 获取评估信息
        eval_info = self._cb.get_evaluation_info()

        result = {
            "timesteps": self._cb.current_timesteps(),
            "episodes": self._cb.current_episodes(),
            "latest_reward": self._sanitize_value(last_detail.get('r')),
            "latest_length": last_detail.get('l', 0),
            "latest_mesh": last_detail.get('mesh_data', {}),
            "latest_boundary": last_detail.get('boundary_vertices_data', []),
            "latest_ref_point": last_detail.get('last_ref_point'),
            "avg_reward_100": self._sanitize_value(self._cb.avg_reward_100()),
            "is_completed": last_detail.get('is_completed', False),
            "avg_element_quality": self._sanitize_value(last_detail.get('avg_element_quality', 0.0)),
            # 训练指标
            "recent_actor_loss": self._sanitize_value(latest_metrics.get("actor_loss", 0.0)),
            "recent_critic_loss": self._sanitize_value(latest_metrics.get("critic_loss", 0.0)),
            "current_alpha": self._sanitize_value(latest_metrics.get("alpha", 0.0)),
            # 评估信息
            "eval_count": eval_info.get("eval_count", 0),
            "last_eval_reward": self._sanitize_value(eval_info.get("last_eval_reward", 0.0)),
            "best_eval_reward": self._sanitize_value(eval_info.get("best_eval_reward", 0.0)),
            "evaluation_frequency": eval_info.get("evaluation_frequency", 10),
            "n_eval_episodes": eval_info.get("n_eval_episodes", 10),
        }

        return result

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

    def plot_avg_element_quality(self, save_path: str) -> str:
        """
        绘制训练过程中平均元素质量变化图表
        
        Args:
            save_path: 图表保存路径
            
        Returns:
            str: 最终保存的文件路径
        """
        avg_qualities = self._cb.get_data('avg_element_quality')
        episode_lengths = self._cb.get_data('l')
        return plot_avg_element_quality(avg_qualities, episode_lengths, save_path)

    def save_history(self, path):
        details = self._cb.get_non_zero_generated_details()
        best_episode_number = self._cb.current_best_episode()
        best_episode_index = 0
        for i, detail in enumerate(details):
            if detail.get("episode_number") == best_episode_number:
                best_episode_index = i
                break
        save_episode_details(details, best_episode_index, path)
