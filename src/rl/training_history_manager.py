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
    - 存储每个episode的详细信息
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

        # 保存初始元数据
        self._save_metadata()

        self.logger.info(f"开始新的训练会话: {self.current_training_id}")
        return self.current_training_id

    def _create_training_subdirectories(self):
        """创建训练会话的子目录结构"""
        subdirs = [
            "episodes",  # 存储每个episode的详细信息
            "checkpoints",  # 存储模型检查点
            "models",  # 存储最终模型权重
            "plots",  # 存储训练曲线图片
            "logs",  # 存储详细日志
            "config"  # 存储配置文件
        ]

        for subdir in subdirs:
            os.makedirs(os.path.join(self.current_training_dir, subdir), exist_ok=True)

    def save_episode_data(self, episode_data: Dict[str, Any]) -> bool:
        """
        保存单个episode的数据并收集统计信息

        Args:
            episode_data: episode数据字典

        Returns:
            bool: 保存是否成功
        """
        if not self.current_training_id:
            self.logger.warning("没有活动的训练会话，无法保存episode数据")
            return False

        try:
            self.episode_count += 1
            episode_num = episode_data.get('episode', self.episode_count)

            # 准备episode文件路径
            episode_filename = f"episode_{episode_num:06d}.json"
            episode_filepath = os.path.join(
                self.current_training_dir,
                "episodes",
                episode_filename
            )

            # 添加额外的元数据
            enhanced_data = episode_data.copy()
            enhanced_data.update({
                "training_id": self.current_training_id,
                "save_timestamp": time.time(),
                "save_datetime": datetime.now().isoformat(),
                "episode_index": self.episode_count
            })

            # 深度清理数据以确保JSON兼容性
            clean_data = self._deep_clean_for_json(enhanced_data)

            # 保存episode数据
            with open(episode_filepath, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)

            # 收集训练统计数据用于图表生成
            self._collect_training_stats(clean_data)

            # 更新训练元数据
            self._update_training_metadata(clean_data)

            # 定期保存元数据（每10个episode或重要里程碑）
            if self.episode_count % 10 == 0 or clean_data.get('episode_reward', 0) > self.training_metadata.get(
                    'best_reward', float('-inf')):
                self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"保存episode数据失败: {e}")
            return False

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

        # 更新元数据
        end_time = time.time()
        status = "stopped" if stopped_early else "completed"

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

        # 生成训练图表
        self._generate_training_plots()

        # 保存最终统计报告
        if final_stats:
            self._save_final_report(final_stats)

        # 保存模型权重到history目录（如果有的话）
        self._save_final_models()

        self.logger.info(f"训练会话结束: {self.current_training_id}, 状态: {status}")

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

    def _generate_training_plots(self):
        """
        生成训练过程的图表，包括actor loss、critic loss、α值和平均回报
        """
        if not self.current_training_dir or len(self.episodes) == 0:
            self.logger.warning("没有足够的数据生成图表")
            return

        try:
            # 设置matplotlib的中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建2x2的子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'训练过程分析 - {self.current_training_id}', fontsize=16, fontweight='bold')

            # 计算滑动平均
            def moving_average(data, window=50):
                if len(data) < window:
                    window = max(1, len(data) // 5)
                return np.convolve(data, np.ones(window) / window, mode='valid')

            episodes_array = np.array(self.episodes)

            # 1. Actor Loss
            if len(self.actor_losses) > 0:
                ax1.plot(episodes_array, self.actor_losses, alpha=0.3, color='blue', label='原始数据')
                if len(self.actor_losses) > 10:
                    smoothed = moving_average(self.actor_losses)
                    ax1.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='blue', linewidth=2, label='滑动平均')
                ax1.set_title('Actor Loss', fontweight='bold')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. Critic Loss
            if len(self.critic_losses) > 0:
                ax2.plot(episodes_array, self.critic_losses, alpha=0.3, color='red', label='原始数据')
                if len(self.critic_losses) > 10:
                    smoothed = moving_average(self.critic_losses)
                    ax2.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='red', linewidth=2, label='滑动平均')
                ax2.set_title('Critic Loss', fontweight='bold')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Alpha值
            if len(self.alpha_values) > 0:
                ax3.plot(episodes_array, self.alpha_values, alpha=0.3, color='green', label='原始数据')
                if len(self.alpha_values) > 10:
                    smoothed = moving_average(self.alpha_values)
                    ax3.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='green', linewidth=2, label='滑动平均')
                ax3.set_title('Alpha (α) 值', fontweight='bold')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Alpha')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # 4. 平均回报
            if len(self.episode_rewards) > 0:
                ax4.plot(episodes_array, self.episode_rewards, alpha=0.3, color='orange', label='Episode奖励')
                if len(self.episode_rewards) > 10:
                    smoothed = moving_average(self.episode_rewards)
                    ax4.plot(episodes_array[len(episodes_array) - len(smoothed):], smoothed,
                             color='orange', linewidth=2, label='滑动平均奖励')

                # 计算累积平均奖励
                cumulative_avg = np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1)
                ax4.plot(episodes_array, cumulative_avg, color='darkred', linewidth=2,
                         linestyle='--', label='累积平均奖励')

                ax4.set_title('回报分析', fontweight='bold')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Reward')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            # 调整布局
            plt.tight_layout()

            # 保存图表
            plots_dir = os.path.join(self.current_training_dir, "plots")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存高分辨率PNG
            png_path = os.path.join(plots_dir, f"training_analysis_{timestamp}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

            # 保存PDF版本
            pdf_path = os.path.join(plots_dir, f"training_analysis_{timestamp}.pdf")
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

            plt.close()  # 关闭图形以释放内存

            # 同时生成训练统计摘要
            self._save_training_statistics()

            self.logger.info(f"训练图表已生成: {png_path}")
            self.logger.info(f"训练图表PDF已生成: {pdf_path}")

        except Exception as e:
            self.logger.error(f"生成训练图表失败: {e}")
            import traceback
            traceback.print_exc()

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

        # 统计episode数量
        episodes_dir = os.path.join(training_dir, "episodes")
        episode_files = []
        if os.path.exists(episodes_dir):
            episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]

        return {
            "metadata": metadata,
            "episode_count": len(episode_files),
            "episode_files": sorted(episode_files),
            "training_directory": training_dir
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
        获取特定episode的数据

        Args:
            training_id: 训练ID
            episode_num: episode编号

        Returns:
            Optional[Dict[str, Any]]: episode数据，如果不存在则返回None
        """
        episode_filename = f"episode_{episode_num:06d}.json"
        episode_filepath = os.path.join(
            self.history_root,
            training_id,
            "episodes",
            episode_filename
        )

        if not os.path.exists(episode_filepath):
            return None

        try:
            with open(episode_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"读取episode数据失败: {e}")
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
