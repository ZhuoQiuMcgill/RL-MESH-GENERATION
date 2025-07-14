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
            **training_params: 训练参数

        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            # 检查训练器是否已初始化
            if self.trainer is None:
                # 尝试使用默认配置初始化
                init_result = self.initialize_trainer()
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

            # 强制设置状态
            self._is_training = False

            return {
                "success": False,
                "error": error_msg
            }

    def _wait_for_training_stop(self, timeout=5.0):
        """等待训练停止"""
        start_time = time.time()
        while self._is_training and (time.time() - start_time) < timeout:
            time.sleep(0.1)

            # 检查训练器状态
            if self.trainer:
                status = self.trainer.get_training_status()
                if not status.get('running', False):
                    self._is_training = False
                    break

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
                    "status": "not_initialized",
                    "stats": None,
                    "backend_type": None
                }

            # 获取训练器状态
            status = self.trainer.get_training_status()

            # 更新本地状态
            actual_running = status.get('running', False)
            if not actual_running and self._is_training:
                self._is_training = False

            # 清理统计数据，确保JSON安全
            stats = status.get('stats', {})
            if stats:
                stats = self._clean_stats_for_json(stats)

            result = {
                "running": actual_running,
                "status": status.get('status', 'unknown'),
                "stats": stats,
                "backend_type": status.get('backend_type', None),
                "timestamp": time.time()
            }

            # 缓存最新统计
            self._last_stats = result

            return result

        except Exception as e:
            print(f"获取训练状态失败: {e}")
            return {
                "running": False,
                "status": "error",
                "stats": None,
                "backend_type": None,
                "error": str(e)
            }

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
        """
        获取训练器信息

        Returns:
            Dict[str, Any]: 训练器信息
        """
        if self.trainer is None:
            return {
                "initialized": False,
                "available_backends": get_available_backends()
            }

        try:
            trainer_info = self.trainer.get_trainer_info()
            trainer_info["initialized"] = True
            trainer_info["available_backends"] = get_available_backends()
            return trainer_info
        except Exception as e:
            return {
                "initialized": False,
                "error": str(e),
                "available_backends": get_available_backends()
            }

    def add_stats_callback(self, callback: Callable):
        """
        添加统计回调函数

        Args:
            callback: 回调函数
        """
        self._stats_callbacks.append(callback)

    def remove_stats_callback(self, callback: Callable):
        """
        移除统计回调函数

        Args:
            callback: 回调函数
        """
        if callback in self._stats_callbacks:
            self._stats_callbacks.remove(callback)

    def _on_episode_complete(self, episode_data: Dict[str, Any]):
        """
        Episode完成回调

        Args:
            episode_data: Episode数据
        """
        try:
            # 触发统计回调
            for callback in self._stats_callbacks:
                try:
                    callback(episode_data)
                except Exception as e:
                    print(f"统计回调执行失败: {e}")
        except Exception as e:
            print(f"Episode回调处理失败: {e}")

    def _on_training_step(self, step_data: Dict[str, Any]):
        """
        训练步骤回调

        Args:
            step_data: 步骤数据
        """
        try:
            # 这里可以添加步骤级别的处理
            pass
        except Exception as e:
            print(f"训练步骤回调处理失败: {e}")

    def save_model(self, path: str) -> Dict[str, Any]:
        """
        保存模型

        Args:
            path: 保存路径

        Returns:
            Dict[str, Any]: 保存结果
        """
        try:
            if self.trainer is None:
                return {
                    "success": False,
                    "error": "训练器未初始化"
                }

            self.trainer.save_model(path)

            return {
                "success": True,
                "message": f"模型已保存到: {path}"
            }

        except Exception as e:
            error_msg = f"保存模型失败: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def load_model(self, path: str) -> Dict[str, Any]:
        """
        加载模型

        Args:
            path: 模型路径

        Returns:
            Dict[str, Any]: 加载结果
        """
        try:
            if self.trainer is None:
                return {
                    "success": False,
                    "error": "训练器未初始化"
                }

            # 检查文件是否存在
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"模型文件不存在: {path}"
                }

            self.trainer.load_model(path)

            return {
                "success": True,
                "message": f"模型已从 {path} 加载"
            }

        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def reset(self):
        """重置训练管理器"""
        try:
            # 停止训练
            if self._is_training:
                self.stop_training()

            # 清理训练器
            self.trainer = None

            # 重置状态
            self._is_training = False
            self._training_thread = None
            self._stop_event.clear()
            self._last_stats = {}
            self._stats_callbacks = []

            print("训练管理器已重置")

        except Exception as e:
            print(f"重置训练管理器失败: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "status": "healthy",
            "trainer_initialized": self.trainer is not None,
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


# 全局训练管理器实例
training_manager = TrainingManager()
