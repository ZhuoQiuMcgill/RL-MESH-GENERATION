import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config


class TrainingHistoryManager:
    """
    训练历史管理器

    负责管理训练过程中的历史数据存储，包括：
    - 为每次训练生成唯一ID
    - 基于timestep间隔存储episode的详细信息
    - 提供历史数据的查询和管理功能
    - 支持训练会话的恢复和分析
    - 生成训练结果图表
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化历史管理器

        Args:
            config: 配置字典，如果为None则从config.yaml加载
        """
        self.config = config if config is not None else load_config()

        # 设置历史数据根目录 - 修改为data/history
        paths_config = self.config.get("paths", {})
        data_root = self._get_absolute_path(paths_config.get("data_root", "data"))
        self.history_root = os.path.join(data_root, "history")

        # 确保历史目录存在
        os.makedirs(self.history_root, exist_ok=True)

        # 当前训练会话信息
        self.current_training_id = None
        self.current_training_dir = None
        self.training_metadata = {}
        self.episode_count = 0

        # 训练统计数据收集
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
        self.episode_rewards = []
        self.episodes = []

        # 基于timestep的保存机制
        self.save_frequency_timesteps = self.config.get("training", {}).get("history_save_frequency", 10000)
        self.last_save_timestep = 0
        self.episode_data_cache = []  # 缓存episode数据
        self.total_timesteps = 0

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        将相对路径转换为绝对路径（基于项目根目录）

        Args:
            relative_path: 相对于项目根目录的路径

        Returns:
            str: 绝对路径
        """
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(os.getcwd(), relative_path)

    def _create_training_subdirectories(self):
        """创建训练会话的子目录结构"""
        subdirs = [
            "episodes",  # 存储episode的详细信息（按timestep间隔保存）
            "checkpoints",  # 存储模型检查点
            "models",  # 存储最终模型权重
            "plots",  # 存储训练曲线图片
            "logs",  # 存储详细日志
            "config"  # 存储配置文件
        ]

        for subdir in subdirs:
            os.makedirs(os.path.join(self.current_training_dir, subdir), exist_ok=True)

    def start_training_session(self,
                               mesh_name: Optional[str] = None,
                               config_overrides: Optional[Dict[str, Any]] = None,
                               description: Optional[str] = None) -> str:
        """
        开始一个新的训练会话

        Args:
            mesh_name: 使用的mesh名称
            config_overrides: 配置覆盖
            description: 训练描述

        Returns:
            str: 生成的训练ID
        """
        # 生成唯一的训练ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_suffix = f"_{mesh_name}" if mesh_name else ""
        unique_suffix = str(uuid.uuid4())[:8]
        self.current_training_id = f"train_{timestamp}{mesh_suffix}_{unique_suffix}"

        # 创建训练目录
        self.current_training_dir = os.path.join(self.history_root, self.current_training_id)
        os.makedirs(self.current_training_dir, exist_ok=True)

        # 创建子目录
        self._create_training_subdirectories()

        # 初始化训练元数据
        self.training_metadata = {
            "training_id": self.current_training_id,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "mesh_name": mesh_name,
            "description": description,
            "config_overrides": config_overrides or {},
            "status": "running",
            "episodes_completed": 0,
            "total_steps": 0,
            "best_reward": None,
            "final_stats": None,
            "end_time": None,
            "end_datetime": None,
            "duration_seconds": None
        }

        # 重置episode计数和统计数据
        self.episode_count = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
        self.episode_rewards = []
        self.episodes = []

        # 重置基于timestep的保存机制
        self.last_save_timestep = 0
        self.episode_data_cache = []
        self.total_timesteps = 0

        # 保存初始元数据
        self._save_metadata()

        self.logger.info(f"开始新的训练会话: {self.current_training_id}")
        self.logger.info(f"History保存频率: 每{self.save_frequency_timesteps}个timesteps")
        return self.current_training_id

    def cache_episode_data(self, episode_data: Dict[str, Any]) -> bool:
        """
        缓存episode数据，而不是立即保存

        Args:
            episode_data: episode数据字典

        Returns:
            bool: 缓存是否成功
        """
        if not self.current_training_id:
            self.logger.warning("没有活动的训练会话，无法缓存episode数据")
            return False

        try:
            self.episode_count += 1
            episode_num = episode_data.get('episode', self.episode_count)

            # 添加额外的元数据
            enhanced_data = episode_data.copy()
            enhanced_data.update({
                "training_id": self.current_training_id,
                "cache_timestamp": time.time(),
                "cache_datetime": datetime.now().isoformat(),
                "episode_index": self.episode_count
            })

            # 深度清理数据以确保JSON兼容性
            clean_data = self._deep_clean_for_json(enhanced_data)

            # 缓存数据
            self.episode_data_cache.append(clean_data)

            # 收集训练统计数据用于图表生成
            self._collect_training_stats(clean_data)

            # 更新训练元数据
            self._update_training_metadata(clean_data)

            # 更新总timesteps
            self.total_timesteps = clean_data.get('total_steps', self.total_timesteps)

            # 检查是否需要保存
            if self._should_save_history():
                self._flush_episode_cache()

            return True

        except Exception as e:
            self.logger.error(f"缓存episode数据失败: {e}")
            return False

    def _should_save_history(self) -> bool:
        """
        判断是否应该保存历史数据（基于timestep间隔）

        Returns:
            bool: 是否应该保存
        """
        timesteps_since_last_save = self.total_timesteps - self.last_save_timestep
        return timesteps_since_last_save >= self.save_frequency_timesteps

    def _flush_episode_cache(self) -> bool:
        """
        将缓存的episode数据批量写入磁盘

        Returns:
            bool: 保存是否成功
        """
        if not self.episode_data_cache:
            return True

        try:
            # 创建批次文件名，基于timestep范围
            batch_filename = f"episodes_batch_{self.last_save_timestep}_{self.total_timesteps}.json"
            batch_filepath = os.path.join(
                self.current_training_dir,
                "episodes",
                batch_filename
            )

            # 准备批次数据
            batch_data = {
                "training_id": self.current_training_id,
                "timestep_range": [self.last_save_timestep, self.total_timesteps],
                "episode_count": len(self.episode_data_cache),
                "save_timestamp": time.time(),
                "save_datetime": datetime.now().isoformat(),
                "episodes": self.episode_data_cache
            }

            # 保存批次数据
            with open(batch_filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"批量保存了{len(self.episode_data_cache)}个episodes "
                             f"(timesteps {self.last_save_timestep}-{self.total_timesteps})")

            # 清空缓存并更新保存时间戳
            self.episode_data_cache = []
            self.last_save_timestep = self.total_timesteps

            # 保存元数据
            self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"批量保存episode数据失败: {e}")
            return False

    def save_episode_data(self, episode_data: Dict[str, Any]) -> bool:
        """
        保存单个episode的数据（为了兼容性保留，内部调用cache_episode_data）

        Args:
            episode_data: episode数据字典

        Returns:
            bool: 保存是否成功
        """
        return self.cache_episode_data(episode_data)

    def force_save_cache(self) -> bool:
        """
        强制保存当前缓存的数据（用于训练结束或紧急情况）

        Returns:
            bool: 保存是否成功
        """
        if self.episode_data_cache:
            return self._flush_episode_cache()
        return True

    def _collect_training_stats(self, episode_data: Dict[str, Any]):
        """
        收集训练统计数据用于图表生成

        Args:
            episode_data: episode数据
        """
        episode_num = episode_data.get('episode', len(self.episodes))
        self.episodes.append(episode_num)

        # 收集奖励数据
        episode_reward = episode_data.get('episode_reward', 0.0)
        self.episode_rewards.append(float(episode_reward))

        # 收集损失数据
        if 'recent_actor_loss' in episode_data:
            self.actor_losses.append(float(episode_data['recent_actor_loss']))
        elif len(self.actor_losses) > 0:
            # 如果当前episode没有损失数据，使用上一个值
            self.actor_losses.append(self.actor_losses[-1])
        else:
            self.actor_losses.append(0.0)

        if 'recent_critic_loss' in episode_data:
            self.critic_losses.append(float(episode_data['recent_critic_loss']))
        elif len(self.critic_losses) > 0:
            self.critic_losses.append(self.critic_losses[-1])
        else:
            self.critic_losses.append(0.0)

        if 'current_alpha' in episode_data:
            self.alpha_values.append(float(episode_data['current_alpha']))
        elif len(self.alpha_values) > 0:
            self.alpha_values.append(self.alpha_values[-1])
        else:
            self.alpha_values.append(0.0)

    def _update_training_metadata(self, episode_data: Dict[str, Any]):
        """更新训练元数据"""
        self.training_metadata["episodes_completed"] = self.episode_count
        self.training_metadata["total_steps"] = episode_data.get("total_steps", 0)

        # 更新最佳奖励
        episode_reward = episode_data.get("episode_reward", 0)
        if self.training_metadata["best_reward"] is None or episode_reward > self.training_metadata["best_reward"]:
            self.training_metadata["best_reward"] = episode_reward

    def finish_training_session(self, final_stats: Optional[Dict[str, Any]] = None, stopped_early: bool = False):
        """
        结束当前训练会话并生成图表

        Args:
            final_stats: 最终训练统计信息
            stopped_early: 是否提前停止
        """
        if not self.current_training_id:
            self.logger.warning("没有活动的训练会话可以结束")
            return

        # 强制保存剩余的缓存数据
        self.force_save_cache()

        # 更新元数据
        end_time = time.time()
        status = "stopped_early" if stopped_early else "completed"

        self.training_metadata.update({
            "status": status,
            "end_time": end_time,
            "end_datetime": datetime.now().isoformat(),
            "duration_seconds": end_time - self.training_metadata["start_time"],
            "final_stats": final_stats,
            "stopped_early": stopped_early
        })

        # 保存最终元数据
        self._save_metadata()

        # 异步生成训练图表，避免阻塞主线程和产生matplotlib报错
        try:
            import threading
            plot_thread = threading.Thread(target=self._generate_training_plots_safe, daemon=True)
            plot_thread.start()
            # 不等待线程完成，允许主程序继续执行
        except Exception as e:
            self.logger.warning(f"无法启动图表生成线程: {e}")

        # 保存最终统计报告
        if final_stats:
            self._save_final_report(final_stats)

        # 保存模型权重到history目录（如果有的话）
        self._save_final_models()

        self.logger.info(f"训练会话结束: {self.current_training_id}, 状态: {status}")
        self.logger.info(f"总共保存了{self.episode_count}个episodes的历史数据")

        # 清理当前会话信息
        self.current_training_id = None
        self.current_training_dir = None
        self.training_metadata = {}
        self.episode_count = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
        self.episode_rewards = []
        self.episodes = []
        self.episode_data_cache = []
        self.last_save_timestep = 0
        self.total_timesteps = 0

    def _generate_training_plots(self):
        """
        生成训练过程的图表，包括actor loss、critic loss、α值和平均回报
        """
        if not self.current_training_dir or len(self.episodes) == 0:
            self.logger.warning("没有足够的数据生成图表")
            return

        try:
            # 设置matplotlib后端为Agg（非交互式），避免GUI相关问题
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # 禁用matplotlib的详细日志输出
            import logging
            matplotlib_logger = logging.getLogger('matplotlib')
            original_level = matplotlib_logger.getEffectiveLevel()
            matplotlib_logger.setLevel(logging.WARNING)

            # 设置matplotlib的中文字体支持，添加异常处理
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                # 如果字体设置失败，使用默认字体
                plt.rcParams['font.family'] = 'sans-serif'

            # 创建2x2的子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Analysis - {self.current_training_id}', fontsize=16, fontweight='bold')

            # 计算滑动平均
            def moving_average(data, window=50):
                if len(data) < window:
                    window = max(1, len(data) // 5)
                return np.convolve(data, np.ones(window) / window, mode='valid')

            episodes_array = np.array(self.episodes)

            # 1. Actor Loss
            if len(self.actor_losses) > 0:
                ax1.plot(episodes_array, self.actor_losses, alpha=0.3, color='blue', label='Raw Data')
                if len(self.actor_losses) > 10:
                    smoothed = moving_average(self.actor_losses)
                    ax1.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='blue', linewidth=2, label='Moving Average')
                ax1.set_title('Actor Loss', fontweight='bold')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. Critic Loss
            if len(self.critic_losses) > 0:
                ax2.plot(episodes_array, self.critic_losses, alpha=0.3, color='green', label='Raw Data')
                if len(self.critic_losses) > 10:
                    smoothed = moving_average(self.critic_losses)
                    ax2.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='green', linewidth=2, label='Moving Average')
                ax2.set_title('Critic Loss', fontweight='bold')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Alpha Values
            if len(self.alpha_values) > 0:
                ax3.plot(episodes_array, self.alpha_values, alpha=0.7, color='purple', linewidth=2)
                ax3.set_title('Alpha Values', fontweight='bold')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Alpha')
                ax3.grid(True, alpha=0.3)

            # 4. Episode Rewards and Average Rewards
            if len(self.episode_rewards) > 0:
                ax4.plot(episodes_array, self.episode_rewards, alpha=0.3, color='orange', label='Episode Reward')
                if len(self.episode_rewards) > 10:
                    smoothed = moving_average(self.episode_rewards)
                    ax4.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='orange', linewidth=2, label='Moving Average Reward')

                # Calculate cumulative average reward
                cumulative_avg = np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1)
                ax4.plot(episodes_array, cumulative_avg, color='darkred', linewidth=2,
                         linestyle='--', label='Cumulative Average Reward')

                ax4.set_title('Reward Analysis', fontweight='bold')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Reward')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save plots
            plots_dir = os.path.join(self.current_training_dir, "plots")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save high-resolution PNG
            png_path = os.path.join(plots_dir, f"training_analysis_{timestamp}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

            # Save PDF version
            pdf_path = os.path.join(plots_dir, f"training_analysis_{timestamp}.pdf")
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

            # Close figure to release memory and avoid threading issues
            plt.close(fig)
            plt.close('all')  # 确保关闭所有图形

            # 恢复matplotlib日志级别
            matplotlib_logger.setLevel(original_level)

            # Generate training statistics summary
            self._save_training_statistics()

            self.logger.info(f"Training plot generated: {png_path}")
            self.logger.info(f"Training plot PDF generated: {pdf_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate training plots: {e}")
            # 不打印详细的traceback，避免产生更多日志噪音
            pass

    def _save_training_statistics(self):
        """
        保存训练统计数据到JSON文件
        """
        try:
            stats_data = {
                "training_id": self.current_training_id,
                "episodes": self.episodes,
                "actor_losses": self.actor_losses,
                "critic_losses": self.critic_losses,
                "alpha_values": self.alpha_values,
                "episode_rewards": self.episode_rewards,
                "summary": {
                    "total_episodes": len(self.episodes),
                    "final_actor_loss": self.actor_losses[-1] if self.actor_losses else None,
                    "final_critic_loss": self.critic_losses[-1] if self.critic_losses else None,
                    "final_alpha": self.alpha_values[-1] if self.alpha_values else None,
                    "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
                    "best_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                    "final_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
                },
                "save_mechanism": {
                    "type": "timestep_based",
                    "save_frequency_timesteps": self.save_frequency_timesteps,
                    "total_timesteps": self.total_timesteps
                },
                "generated_at": datetime.now().isoformat()
            }

            stats_path = os.path.join(self.current_training_dir, "training_statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"训练统计数据已保存: {stats_path}")

        except Exception as e:
            self.logger.error(f"保存训练统计数据失败: {e}")

    def _save_final_models(self):
        """
        保存最终模型权重到history目录
        """
        try:
            models_dir = os.path.join(self.current_training_dir, "models")

            # 创建一个标记文件，表示模型应该保存在这里
            model_info = {
                "note": "模型权重应保存在此目录",
                "training_id": self.current_training_id,
                "completion_time": datetime.now().isoformat(),
                "recommended_files": [
                    "final_actor.pth",
                    "final_critic.pth",
                    "final_actor_optimizer.pth",
                    "final_critic_optimizer.pth"
                ]
            }

            info_path = os.path.join(models_dir, "model_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"保存模型信息失败: {e}")

    def get_training_plots_path(self, training_id: Optional[str] = None) -> str:
        """
        获取训练图表的保存路径

        Args:
            training_id: 训练ID，如果为None则使用当前训练ID

        Returns:
            str: 图表目录路径
        """
        if training_id is None:
            training_id = self.current_training_id

        if not training_id:
            return ""

        return os.path.join(self.history_root, training_id, "plots")

    def get_training_models_path(self, training_id: Optional[str] = None) -> str:
        """
        获取训练模型的保存路径

        Args:
            training_id: 训练ID，如果为None则使用当前训练ID

        Returns:
            str: 模型目录路径
        """
        if training_id is None:
            training_id = self.current_training_id

        if not training_id:
            return ""

        return os.path.join(self.history_root, training_id, "models")

    def get_training_checkpoints_path(self, training_id: Optional[str] = None) -> str:
        """
        获取训练检查点的保存路径

        Args:
            training_id: 训练ID，如果为None则使用当前训练ID

        Returns:
            str: 检查点目录路径
        """
        if training_id is None:
            training_id = self.current_training_id

        if not training_id:
            return ""

        return os.path.join(self.history_root, training_id, "checkpoints")

    def _save_metadata(self):
        """保存训练元数据"""
        if not self.current_training_dir:
            return

        metadata_path = os.path.join(self.current_training_dir, "training_metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存训练元数据失败: {e}")

    def _save_final_report(self, final_stats: Dict[str, Any]):
        """保存最终训练报告"""
        report_path = os.path.join(self.current_training_dir, "final_report.json")

        report = {
            "training_summary": self.training_metadata,
            "final_statistics": final_stats,
            "save_mechanism": {
                "type": "timestep_based",
                "save_frequency_timesteps": self.save_frequency_timesteps,
                "total_batches_saved": len([f for f in os.listdir(os.path.join(self.current_training_dir, "episodes"))
                                            if f.endswith('.json')]) if os.path.exists(
                    os.path.join(self.current_training_dir, "episodes")) else 0
            },
            "generated_at": datetime.now().isoformat()
        }

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存最终报告失败: {e}")

    def _deep_clean_for_json(self, data):
        """
        递归清理数据，确保所有数据都是JSON安全的

        Args:
            data: 要清理的数据

        Returns:
            清理后的数据
        """
        if data is None:
            return None
        elif isinstance(data, (bool, int, float, str)):
            return data
        elif hasattr(data, 'item'):  # numpy类型
            try:
                return float(data.item())
            except:
                return str(data)
        elif isinstance(data, dict):
            cleaned_dict = {}
            for key, value in data.items():
                try:
                    # 确保键是字符串
                    clean_key = str(key)
                    cleaned_dict[clean_key] = self._deep_clean_for_json(value)
                except Exception as e:
                    self.logger.warning(f"清理字典项时出错: {e}, key: {key}")
                    cleaned_dict[str(key)] = None
            return cleaned_dict
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for item in data:
                try:
                    cleaned_list.append(self._deep_clean_for_json(item))
                except Exception as e:
                    self.logger.warning(f"清理列表项时出错: {e}")
                    cleaned_list.append(None)
            return cleaned_list
        else:
            # 对于其他类型，尝试转换为字符串
            try:
                return str(data)
            except:
                return None

    def get_training_history(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取训练历史信息

        Args:
            training_id: 训练ID，如果为None则返回当前训练信息

        Returns:
            Dict[str, Any]: 训练历史信息
        """
        if training_id is None:
            training_id = self.current_training_id

        if not training_id:
            return {"error": "没有指定的训练ID"}

        training_dir = os.path.join(self.history_root, training_id)
        if not os.path.exists(training_dir):
            return {"error": f"训练记录不存在: {training_id}"}

        # 读取元数据
        metadata_path = os.path.join(training_dir, "training_metadata.json")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            return {"error": f"读取训练元数据失败: {e}"}

        # 统计episode批次文件数量
        episodes_dir = os.path.join(training_dir, "episodes")
        batch_files = []
        total_episodes = 0
        if os.path.exists(episodes_dir):
            batch_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]
            # 计算总episode数量（从批次文件中）
            for batch_file in batch_files:
                try:
                    batch_path = os.path.join(episodes_dir, batch_file)
                    with open(batch_path, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        total_episodes += batch_data.get("episode_count", 0)
                except Exception as e:
                    self.logger.warning(f"读取批次文件失败: {batch_file}, {e}")

        return {
            "metadata": metadata,
            "episode_count": total_episodes,
            "batch_files": sorted(batch_files),
            "training_directory": training_dir,
            "save_mechanism": "timestep_based"
        }

    def list_all_trainings(self) -> List[Dict[str, Any]]:
        """
        列出所有训练记录

        Returns:
            List[Dict[str, Any]]: 所有训练记录的列表
        """
        if not os.path.exists(self.history_root):
            return []

        trainings = []
        for item in os.listdir(self.history_root):
            item_path = os.path.join(self.history_root, item)
            if os.path.isdir(item_path) and item.startswith("train_"):
                training_info = self.get_training_history(item)
                if "error" not in training_info:
                    trainings.append({
                        "training_id": item,
                        "metadata": training_info["metadata"],
                        "episode_count": training_info["episode_count"]
                    })

        # 按开始时间排序
        trainings.sort(key=lambda x: x["metadata"].get("start_time", 0), reverse=True)
        return trainings

    def get_episode_data(self, training_id: str, episode_num: int) -> Optional[Dict[str, Any]]:
        """
        获取特定episode的数据（从批次文件中搜索）

        Args:
            training_id: 训练ID
            episode_num: episode编号

        Returns:
            Optional[Dict[str, Any]]: episode数据，如果不存在则返回None
        """
        episodes_dir = os.path.join(self.history_root, training_id, "episodes")
        if not os.path.exists(episodes_dir):
            return None

        # 搜索所有批次文件
        batch_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]

        for batch_file in batch_files:
            try:
                batch_path = os.path.join(episodes_dir, batch_file)
                with open(batch_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)

                # 在批次中搜索指定的episode
                episodes = batch_data.get("episodes", [])
                for episode_data in episodes:
                    if episode_data.get("episode") == episode_num:
                        return episode_data

            except Exception as e:
                self.logger.error(f"读取批次文件失败: {batch_file}, {e}")
                continue

        return None

    def delete_training_history(self, training_id: str) -> bool:
        """
        删除指定的训练历史

        Args:
            training_id: 要删除的训练ID

        Returns:
            bool: 删除是否成功
        """
        import shutil

        training_dir = os.path.join(self.history_root, training_id)
        if not os.path.exists(training_dir):
            self.logger.warning(f"训练记录不存在，无法删除: {training_id}")
            return False

        try:
            shutil.rmtree(training_dir)
            self.logger.info(f"成功删除训练历史: {training_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除训练历史失败: {e}")
            return False

    def export_training_summary(self, training_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """
        导出训练摘要报告

        Args:
            training_id: 训练ID
            export_path: 导出路径，如果为None则使用默认路径

        Returns:
            Optional[str]: 导出的文件路径，失败时返回None
        """
        training_info = self.get_training_history(training_id)
        if "error" in training_info:
            self.logger.error(f"无法导出训练摘要: {training_info['error']}")
            return None

        if export_path is None:
            export_path = os.path.join(os.getcwd(), f"{training_id}_summary.json")

        try:
            summary = {
                "training_id": training_id,
                "export_datetime": datetime.now().isoformat(),
                "training_metadata": training_info["metadata"],
                "episode_count": training_info["episode_count"],
                "batch_files_count": len(training_info.get("batch_files", [])),
                "save_mechanism": training_info.get("save_mechanism", "timestep_based"),
                "export_path": export_path
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"训练摘要已导出: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"导出训练摘要失败: {e}")
            return None

    def get_current_training_id(self) -> Optional[str]:
        """
        获取当前训练会话ID

        Returns:
            Optional[str]: 当前训练ID，如果没有活动会话则返回None
        """
        return self.current_training_id

    def is_training_active(self) -> bool:
        """
        检查是否有活动的训练会话

        Returns:
            bool: 是否有活动的训练会话
        """
        return self.current_training_id is not None

    def get_save_frequency(self) -> int:
        """
        获取当前的保存频率（timesteps）

        Returns:
            int: 保存频率
        """
        return self.save_frequency_timesteps

    def set_save_frequency(self, frequency: int):
        """
        设置保存频率（timesteps）

        Args:
            frequency: 新的保存频率
        """
        if frequency > 0:
            self.save_frequency_timesteps = frequency
            self.logger.info(f"History保存频率已更新为: 每{frequency}个timesteps")
        else:
            self.logger.warning("保存频率必须大于0")

    def get_cache_status(self) -> Dict[str, Any]:
        """
        获取当前缓存状态

        Returns:
            Dict[str, Any]: 缓存状态信息
        """
        return {
            "cached_episodes": len(self.episode_data_cache),
            "last_save_timestep": self.last_save_timestep,
            "current_timestep": self.total_timesteps,
            "timesteps_since_last_save": self.total_timesteps - self.last_save_timestep,
            "save_frequency": self.save_frequency_timesteps,
            "will_save_next": self._should_save_history()
        }

    def _generate_training_plots_safe(self):
        """
        安全地生成训练图表，用于在单独线程中执行
        """
        try:
            self._generate_training_plots()
        except Exception as e:
            self.logger.warning(f"图表生成失败: {e}")
            # 静默处理，不影响主要的训练停止流程
