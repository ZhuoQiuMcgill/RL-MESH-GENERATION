"""
训练管理器 - 重构版本

提供统一的训练管理接口，支持自制SAC和SB3 SAC的无缝切换
"""
import os
import time
import threading
import traceback
from typing import Dict, Any, Optional, Callable

from src.rl.training.unified_trainer import UnifiedTrainer, get_available_backends
from src.rl.config import load_config


class TrainingManager:
    """
    重构的训练管理器

    通过统一训练器接口管理两种SAC实现，提供一致的前端API
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化训练管理器

        Args:
            config: 配置字典
        """
        self.config = config if config is not None else load_config()

        # 训练器实例
        self.trainer: Optional[UnifiedTrainer] = None

        # 训练状态
        self._is_training = False
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 统计信息缓存
        self._last_stats = {}

        # 回调函数
        self._stats_callbacks = []

        print("训练管理器初始化完成")
        print(f"可用的SAC后端: {get_available_backends()}")

    def initialize_trainer(self, boundary_source=None, device=None,
                           backend=None) -> Dict[str, Any]:
        """
        初始化训练器

        Args:
            boundary_source: 边界数据源
            device: 训练设备
            backend: 强制指定SAC后端

        Returns:
            Dict[str, Any]: 初始化结果
        """
        try:
            # 如果有正在运行的训练，先停止
            if self._is_training:
                self.stop_training()

            # 创建新的训练器
            config = self.config.copy() if backend is None else self.config.copy()
            if backend is not None:
                if "sac_backend" not in config:
                    config["sac_backend"] = {}
                config["sac_backend"]["type"] = backend

            self.trainer = UnifiedTrainer(
                boundary_source=boundary_source,
                config=config,
                device=device
            )

            # 设置回调
            self.trainer.add_episode_callback(self._on_episode_complete)
            self.trainer.add_step_callback(self._on_training_step)

            trainer_info = self.trainer.get_trainer_info()

            return {
                "success": True,
                "message": "训练器初始化成功",
                "trainer_info": trainer_info
            }

        except Exception as e:
            error_msg = f"训练器初始化失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def start_training(self, **training_params) -> Dict[str, Any]:
        """
        启动训练

        Args:
            **training_params: 训练参数，包括：
                - boundary_source: 边界数据源
                - backend: SAC后端类型
                - device: 训练设备
                - max_timesteps: 最大训练步数
                - 等等

        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            # 检查训练器是否已初始化
            if self.trainer is None:
                # 从training_params中提取初始化参数
                boundary_source = training_params.get('boundary_source')
                backend = training_params.get('backend')
                device = training_params.get('device')

                print(
                    f"训练器未初始化，正在使用参数初始化: boundary_source={boundary_source}, backend={backend}, device={device}")

                # 使用传入的参数初始化训练器
                init_result = self.initialize_trainer(
                    boundary_source=boundary_source,
                    backend=backend,
                    device=device
                )
                if not init_result["success"]:
                    return init_result

            # 检查是否已在训练
            if self._is_training:
                return {
                    "success": False,
                    "error": "训练已在进行中"
                }

            # 重置停止事件
            self._stop_event.clear()

            # 异步启动训练 - 不传递stop_event参数，因为它已经在trainer内部管理
            training_id = self.trainer.start_training_async(**training_params)

            self._is_training = True

            return {
                "success": True,
                "message": "训练启动成功",
                "training_id": training_id
            }

        except Exception as e:
            error_msg = f"启动训练失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def stop_training(self) -> Dict[str, Any]:
        """
        停止训练

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            if not self._is_training:
                return {
                    "success": True,
                    "message": "训练未在进行中"
                }

            # 设置停止事件
            self._stop_event.set()

            # 如果有训练器，调用其停止方法
            if self.trainer:
                self.trainer.stop_training()

            # 等待训练完全停止
            self._wait_for_training_stop()

            self._is_training = False

            return {
                "success": True,
                "message": "训练已停止"
            }

        except Exception as e:
            error_msg = f"停止训练失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def _wait_for_training_stop(self, timeout: float = 10.0):
        """等待训练完全停止"""
        if hasattr(self.trainer, 'training_thread') and self.trainer.training_thread:
            self.trainer.training_thread.join(timeout=timeout)

    def get_status(self) -> Dict[str, Any]:
        """
        获取训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        try:
            if self.trainer is None:
                return {
                    "running": False,
                    "status": "idle",
                    "stats": None,
                    "backend_type": None,
                    "timestamp": time.time()
                }

            # 获取训练器状态
            trainer_status = self.trainer.get_training_status()

            # 清理统计数据以确保JSON兼容性
            stats = trainer_status.get("stats", {})
            if stats:
                stats = self._clean_stats_for_json(stats)

                # 确保包含可视化数据
                stats = self._ensure_visualization_data(stats)

            return {
                "running": trainer_status.get("running", False),
                "status": trainer_status.get("status", "stopped"),
                "stats": stats,
                "backend_type": trainer_status.get("backend_type"),
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"获取训练状态失败: {e}")
            traceback.print_exc()
            return {
                "running": False,
                "status": "error",
                "stats": None,
                "backend_type": None,
                "error": str(e),
                "timestamp": time.time()
            }

    def _ensure_visualization_data(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        确保统计数据包含前端需要的可视化数据

        Args:
            stats: 统计数据字典

        Returns:
            Dict[str, Any]: 包含可视化数据的统计信息
        """
        # 如果trainer有环境，尝试获取可视化数据
        if self.trainer and hasattr(self.trainer, 'trainer'):
            trainer_instance = self.trainer.trainer

            # 获取边界数据
            if hasattr(trainer_instance, 'initial_boundary') and trainer_instance.initial_boundary:
                boundary_vertices = trainer_instance.initial_boundary.get_vertices()
                stats["boundary_vertices"] = len(boundary_vertices)
                stats["boundary_vertices_data"] = boundary_vertices
            else:
                stats["boundary_vertices"] = stats.get("boundary_vertices", 0)
                stats["boundary_vertices_data"] = stats.get("boundary_vertices_data", [])

            # 获取网格数据
            if hasattr(trainer_instance, 'env') and trainer_instance.env:
                try:
                    # 尝试从环境获取网格数据
                    if hasattr(trainer_instance.env, 'get_mesh_data'):
                        mesh_data = trainer_instance.env.get_mesh_data()
                        stats["mesh_data"] = mesh_data if mesh_data else {}
                    elif hasattr(trainer_instance.env, 'mesh') and trainer_instance.env.mesh:
                        # 手动构建网格数据
                        mesh_data = trainer_instance.env.mesh.get_adjacency_dict()
                        stats["mesh_data"] = mesh_data if mesh_data else {}
                    else:
                        stats["mesh_data"] = stats.get("mesh_data", {})

                    # 获取参考点信息
                    if hasattr(trainer_instance.env, 'get_last_reference_info'):
                        ref_info = trainer_instance.env.get_last_reference_info()
                        stats["reference_point_info"] = ref_info if ref_info else {}
                    elif hasattr(trainer_instance.env, 'last_reference_info'):
                        stats["reference_point_info"] = trainer_instance.env.last_reference_info or {}
                    else:
                        stats["reference_point_info"] = stats.get("reference_point_info", {})

                except Exception as e:
                    print(f"获取环境可视化数据失败: {e}")
                    stats["mesh_data"] = stats.get("mesh_data", {})
                    stats["reference_point_info"] = stats.get("reference_point_info", {})
            else:
                stats["mesh_data"] = stats.get("mesh_data", {})
                stats["reference_point_info"] = stats.get("reference_point_info", {})

            # 获取缓冲区大小
            if hasattr(trainer_instance, 'replay_buffer') and trainer_instance.replay_buffer:
                try:
                    buffer_size = len(trainer_instance.replay_buffer)
                    stats["buffer_size"] = buffer_size
                except:
                    stats["buffer_size"] = stats.get("buffer_size", 0)
            else:
                stats["buffer_size"] = stats.get("buffer_size", 0)

            # 确保训练ID存在
            if hasattr(trainer_instance, 'history_manager'):
                training_id = trainer_instance.history_manager.get_current_training_id()
                stats["training_id"] = training_id or stats.get("training_id", "")
            else:
                stats["training_id"] = stats.get("training_id", "")

            # 确保在线学习模式标志存在
            if hasattr(trainer_instance, 'online_learning_mode'):
                stats["online_learning_mode"] = trainer_instance.online_learning_mode
            else:
                stats["online_learning_mode"] = stats.get("online_learning_mode", False)

        # 修复关键字段映射问题
        if 'latest_reward' in stats and 'episode_reward' not in stats:
            stats['episode_reward'] = stats['latest_reward']
        elif 'episode_reward' not in stats:
            stats['episode_reward'] = 0.0

        if 'episodes_completed' in stats and 'episode' not in stats:
            stats['episode'] = stats['episodes_completed']
        elif 'episode' not in stats:
            stats['episode'] = 0

        # 确保episode_length字段存在
        if 'episode_length' not in stats:
            stats['episode_length'] = 0

        # 确保所有必需字段都存在
        required_fields = {
            "mesh_data": {},
            "boundary_vertices_data": [],
            "reference_point_info": {},
            "boundary_vertices": 0,
            "buffer_size": 0,
            "training_id": "",
            "online_learning_mode": False
        }

        for field, default_value in required_fields.items():
            if field not in stats:
                stats[field] = default_value

        return stats

    def _clean_stats_for_json(self, data):
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
                clean_key = str(key)
                cleaned_dict[clean_key] = self._clean_stats_for_json(value)
            return cleaned_dict
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for item in data:
                cleaned_list.append(self._clean_stats_for_json(item))
            return cleaned_list
        else:
            try:
                return str(data)
            except:
                return None

    def load_boundary(self, boundary_source) -> Dict[str, Any]:
        """
        加载新边界

        Args:
            boundary_source: 边界数据源

        Returns:
            Dict[str, Any]: 加载结果
        """
        try:
            if self.trainer is None:
                return {
                    "success": False,
                    "error": "训练器未初始化"
                }

            # 如果正在训练，需要先停止
            if self._is_training:
                stop_result = self.stop_training()
                if not stop_result["success"]:
                    return {
                        "success": False,
                        "error": f"无法停止当前训练: {stop_result.get('error', 'unknown')}"
                    }

            # 加载新边界
            self.trainer.load_boundary(boundary_source)

            return {
                "success": True,
                "message": "边界加载成功"
            }

        except Exception as e:
            error_msg = f"加载边界失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def get_trainer_info(self) -> Dict[str, Any]:
        """获取训练器信息"""
        try:
            if self.trainer is None:
                return {
                    "is_training": self._is_training,
                    "available_backends": get_available_backends(),
                    "timestamp": time.time()
                }

            trainer_info = self.trainer.get_trainer_info()
            trainer_info.update({
                "is_training": self._is_training,
                "available_backends": get_available_backends(),
                "timestamp": time.time()
            })

            return trainer_info

        except Exception as e:
            return {
                "error": str(e),
                "is_training": self._is_training,
                "available_backends": get_available_backends(),
                "timestamp": time.time()
            }

    def switch_backend(self, backend_type: str, boundary_source=None) -> Dict[str, Any]:
        """
        切换SAC后端

        Args:
            backend_type: 后端类型 ("custom" 或 "sb3")
            boundary_source: 边界数据源（可选，保持当前边界）

        Returns:
            Dict[str, Any]: 切换结果
        """
        try:
            # 验证后端类型
            available_backends = get_available_backends()
            if backend_type not in available_backends:
                return {
                    "success": False,
                    "error": f"不支持的后端类型: {backend_type}"
                }

            if not available_backends[backend_type]:
                return {
                    "success": False,
                    "error": f"后端 {backend_type} 不可用（可能缺少依赖）"
                }

            # 停止当前训练
            if self._is_training:
                stop_result = self.stop_training()
                if not stop_result["success"]:
                    return {
                        "success": False,
                        "error": f"无法停止当前训练: {stop_result.get('error', 'unknown')}"
                    }

            # 保留当前边界（如果未指定新边界）
            if boundary_source is None and self.trainer is not None:
                boundary_source = self.trainer.initial_boundary

            # 重新初始化训练器
            init_result = self.initialize_trainer(
                boundary_source=boundary_source,
                backend=backend_type
            )

            if init_result["success"]:
                return {
                    "success": True,
                    "message": f"已切换到 {backend_type} 后端",
                    "trainer_info": init_result["trainer_info"]
                }
            else:
                return init_result

        except Exception as e:
            error_msg = f"切换后端失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def _on_episode_complete(self, episode_data: Dict[str, Any]):
        """处理episode完成事件"""
        self._last_stats = episode_data

    def _on_training_step(self, step_data: Dict[str, Any]):
        """处理训练步骤事件"""
        # 可以在这里处理步骤级别的数据
        pass

    def is_training_active(self) -> bool:
        """
        检查训练是否处于活跃状态

        Returns:
            bool: 如果训练正在进行返回True，否则返回False
        """
        return self._is_training


# 全局训练管理器实例
training_manager = TrainingManager()
