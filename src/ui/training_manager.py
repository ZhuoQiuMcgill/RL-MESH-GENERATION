"""
训练管理器

负责管理强化学习训练会话，包括启动、停止、状态监控等功能。
提供与前端API的桥接功能。
"""

import os
import threading
import time
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.rl.config import load_config
from src.utils import create_default_importer
from src.rl.environment import MeshEnv
from src.rl.training.sb3_sac_trainer import SB3SACTrainer


BASE_DIR = os.getcwd()
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")


class TrainingManager:
    """
    训练管理器

    负责管理强化学习训练会话的生命周期，包括启动、停止、状态监控等。
    """

    def __init__(self, config=None):
        """
        初始化训练管理器

        Args:
            config: 配置字典，如果为None则从config.yaml加载
        """
        self.config = config if config is not None else load_config()
        self.logger = logging.getLogger(__name__)

        # 训练状态
        self._is_running = False
        self._training_thread = None
        self._stop_event = threading.Event()

        # 训练实例
        self.trainer = None
        self.env = None
        self.mesh_importer = create_default_importer()

        # 当前训练会话信息
        self.current_training_config = None
        self.training_start_time = None
        self.training_id = None

        # 训练统计信息
        self.current_stats = {
            "episode": 0,
            "total_steps": 0,
            "episode_reward": 0.0,
            "average_reward": 0.0,
            "episode_length": 0,
            "boundary_vertices": 0,
            "buffer_size": 0,
            "training_id": None,
            "online_learning_mode": False,
            "recent_actor_loss": 0.0,
            "recent_critic_loss": 0.0,
            "current_alpha": 0.0,
            "mesh_data": {},
            "boundary_vertices_data": [],
            "reference_point_info": None
        }

        self.logger.info("训练管理器初始化完成")

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        启动训练会话

        Args:
            config: 训练配置
                - mesh_name: mesh文件名
                - subfolder: 子文件夹名称，默认为"mesh"
                - max_timesteps: 最大训练时间步数（前端参数，优先级最高）
                - max_steps: 每episode最大步数（前端参数，优先级最高）
                - description: 训练描述

        Returns:
            Dict[str, Any]: 启动结果

        Raises:
            RuntimeError: 当训练已在运行时
            ValueError: 当配置无效时
        """
        if self._is_running:
            raise RuntimeError("Training already running")

        try:
            # 合并配置：前端参数优先，然后是config.yaml中的参数
            merged_config = self._merge_config(config)

            # 验证配置
            self._validate_config(merged_config)

            # 生成训练ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mesh_name = merged_config.get("mesh_name", "default")
            self.training_id = f"train_{timestamp}_{mesh_name}"

            # 更新统计信息中的training_id
            self.current_stats["training_id"] = self.training_id

            # 创建环境
            self._create_environment(merged_config)

            # 创建训练器
            self._create_trainer(merged_config)

            # 保存配置
            self.current_training_config = merged_config.copy()
            self.training_start_time = datetime.now()

            # 启动训练线程
            self._stop_event.clear()
            self._training_thread = threading.Thread(
                target=self._training_loop,
                args=(merged_config,),
                daemon=True
            )
            self._training_thread.start()

            self._is_running = True

            self.logger.info(f"训练已启动，ID: {self.training_id}")

            return {
                "message": "training_started",
                "success": True,
                "config": merged_config,
                "training_id": self.training_id
            }

        except Exception as e:
            self.logger.error(f"启动训练失败: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def stop_training(self) -> Dict[str, Any]:
        """
        停止当前训练会话

        Returns:
            Dict[str, Any]: 停止结果
        """
        if not self._is_running:
            return {
                "message": "stop_requested",
                "success": True
            }

        try:
            # 设置停止事件
            self._stop_event.set()

            # 等待训练线程结束
            if self._training_thread and self._training_thread.is_alive():
                self._training_thread.join(timeout=5.0)

            self._is_running = False

            self.logger.info("训练已停止")

            return {
                "message": "stop_requested",
                "success": True
            }

        except Exception as e:
            self.logger.error(f"停止训练失败: {e}")
            return {
                "message": f"stop_failed: {str(e)}",
                "success": False
            }

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        if not self._is_running:
            return {
                "running": False,
                "status": "idle",
                "stats": None,
                "progress": None,
                "timestamp": time.time()
            }

        # 更新统计信息
        self._update_stats()

        # 构建进度信息
        progress_info = {
            "current_episode": self.current_stats.get("episode", 0),
            "total_steps": self.current_stats.get("total_steps", 0),
            "latest_reward": self.current_stats.get("episode_reward", 0.0),
            "average_reward": self.current_stats.get("average_reward", 0.0),
            "buffer_utilization": self.current_stats.get("buffer_size", 0)
        }

        return {
            "running": True,
            "status": "running",
            "stats": self.current_stats.copy(),
            "progress": progress_info,
            "timestamp": time.time()
        }

    def is_running(self) -> bool:
        """
        检查训练是否正在运行

        Returns:
            bool: 训练运行状态
        """
        return self._is_running

    def get_current_training_id(self) -> Optional[str]:
        """
        获取当前训练会话ID

        Returns:
            Optional[str]: 训练会话ID，如果没有运行的训练则返回None
        """
        return self.training_id if self._is_running else None

    def _merge_config(self, frontend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置：前端参数优先，config.yaml作为默认值

        Args:
            frontend_config: 前端传入的配置

        Returns:
            Dict[str, Any]: 合并后的配置
        """
        # 从config.yaml获取默认配置
        training_config = self.config.get("training", {})
        environment_config = self.config.get("environment", {})

        # 构建合并后的配置
        merged_config = {}

        # mesh相关参数（前端必须提供）
        merged_config["mesh_name"] = frontend_config.get("mesh_name")
        merged_config["subfolder"] = frontend_config.get("subfolder", "mesh")

        # 训练参数：前端优先，然后使用config.yaml的默认值
        merged_config["max_timesteps"] = (
                frontend_config.get("max_timesteps") or
                training_config.get("max_timesteps", 1000000)
        )

        merged_config["max_steps"] = (
                frontend_config.get("max_steps") or
                environment_config.get("max_steps", 1000)
        )

        # 其他可选参数
        merged_config["description"] = frontend_config.get("description")

        # 添加其他配置项，这些不会被前端修改
        merged_config.update({
            "save_frequency": training_config.get("save_frequency", 10000),
            "log_frequency": training_config.get("log_frequency", 1000),
            "evaluation_frequency": training_config.get("evaluation_frequency", 5000),
            "history_save_frequency": training_config.get("history_save_frequency", 10000)
        })

        self.logger.info(
            f"配置合并完成: max_timesteps={merged_config['max_timesteps']}, max_steps={merged_config['max_steps']}")

        return merged_config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证训练配置

        Args:
            config: 训练配置

        Raises:
            ValueError: 当配置无效时
        """
        # 检查必要的参数
        mesh_name = config.get("mesh_name")
        if not mesh_name:
            raise ValueError("mesh_name参数是必需的")

        # 验证mesh文件是否存在
        subfolder = config.get("subfolder", "mesh")
        try:
            mesh_info = self.mesh_importer.get_mesh_info(mesh_name, subfolder)
            if not mesh_info.get("exists", False):
                raise ValueError(f"Mesh文件不存在: {mesh_name}")
        except Exception as e:
            raise ValueError(f"无法验证mesh文件: {str(e)}")

        # 验证数值参数
        max_timesteps = config.get("max_timesteps")
        if max_timesteps is not None:
            if not isinstance(max_timesteps, int) or max_timesteps <= 0:
                raise ValueError("max_timesteps必须是正整数")

        max_steps = config.get("max_steps")
        if max_steps is not None:
            if not isinstance(max_steps, int) or max_steps <= 0:
                raise ValueError("max_steps必须是正整数")

    def _create_environment(self, config: Dict[str, Any]) -> None:
        """
        创建训练环境

        Args:
            config: 训练配置
        """
        mesh_name = config["mesh_name"]
        subfolder = config.get("subfolder", "mesh")

        # 加载边界
        boundary = self.mesh_importer.load_boundary_by_name(mesh_name, subfolder)

        # 获取环境参数
        max_steps = config.get("max_steps")

        # 创建环境，传入完整的config以确保所有环境参数都能被读取
        self.env = MeshEnv(
            initial_boundary=boundary,
            max_steps=max_steps,
            config=self.config
        )

        self.logger.info(f"环境已创建，使用mesh: {mesh_name}, max_steps: {max_steps}")

    def _create_trainer(self, config: Dict[str, Any]) -> None:
        """
        创建训练器

        Args:
            config: 训练配置
        """
        if self.env is None:
            raise RuntimeError("环境未创建")

        # 确定设备
        device = "cuda"  # TODO: 可以从config中获取

        # 创建SB3 SAC训练器，传入完整的config以确保所有训练参数都能被读取
        self.trainer = SB3SACTrainer(
            env=self.env,
            device=device,
            config=self.config
        )

        self.logger.info(f"训练器已创建，设备: {device}")

    def _training_loop(self, config: Dict[str, Any]) -> None:
        """
        训练主循环

        Args:
            config: 训练配置
        """
        try:
            max_timesteps = config.get("max_timesteps")

            self.logger.info(f"开始训练，最大时间步数: {max_timesteps}")

            # 开始训练
            self.trainer.train(total_timesteps=max_timesteps)

            self.logger.info("训练完成")

        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self._is_running = False
            self._stop_event.set()

    def _update_stats(self) -> None:
        """
        更新训练统计信息
        """
        if not self.trainer:
            return

        try:
            # 获取训练器状态
            trainer_status = self.trainer.get_status()

            # 更新基础统计信息
            self.current_stats.update({
                "episode": trainer_status.get("episodes", 0),
                "total_steps": trainer_status.get("timesteps", 0),
                "episode_reward": trainer_status.get("latest_reward", 0.0) or 0.0,
                "average_reward": trainer_status.get("avg_reward_100", 0.0) or 0.0,
                "episode_length": trainer_status.get("latest_length", 0) or 0,
                "training_id": self.training_id,
                "online_learning_mode": False,  # SB3默认使用经验回放
            })

            mesh_data = trainer_status.get("latest_mesh")
            boundary_vertices = trainer_status.get("latest_boundary")
            ref_info = trainer_status.get("latest_ref_point")

            self.current_stats["boundary_vertices"] = len(boundary_vertices)
            self.current_stats["mesh_data"] = mesh_data if mesh_data else {}
            self.current_stats["boundary_vertices_data"] = boundary_vertices if boundary_vertices else []
            self.current_stats["reference_point_info"] = ref_info if ref_info else (0.0, 0.0)

            # 获取缓冲区大小
            if hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'replay_buffer'):
                replay_buffer = self.trainer.model.replay_buffer
                if replay_buffer and hasattr(replay_buffer, 'size'):
                    self.current_stats["buffer_size"] = replay_buffer.size()
                else:
                    self.current_stats["buffer_size"] = 0
            else:
                self.current_stats["buffer_size"] = 0

        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取训练管理器健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "status": "healthy",
            "service": "training-api",
            "manager_running": self._is_running,
            "current_training_id": self.training_id,
            "timestamp": time.time()
        }

    def cleanup(self) -> None:
        """
        清理资源
        """
        if self._is_running:
            self.stop_training()

        self.trainer = None
        self.env = None
        self.current_training_config = None
        self.training_start_time = None
        self.training_id = None

        self.logger.info("训练管理器资源已清理")


# 全局训练管理器实例
_training_manager = None


def get_training_manager(config=None) -> TrainingManager:
    """
    获取全局训练管理器实例

    Args:
        config: 配置字典

    Returns:
        TrainingManager: 训练管理器实例
    """
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager(config)
    return _training_manager


def reset_training_manager() -> None:
    """
    重置全局训练管理器实例
    """
    global _training_manager
    if _training_manager:
        _training_manager.cleanup()
    _training_manager = None
