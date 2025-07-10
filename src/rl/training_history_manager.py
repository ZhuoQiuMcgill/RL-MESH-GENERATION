import threading
from typing import Optional, Dict, Any

try:
    from src.rl.trainer import MeshTrainer
except ImportError:
    # 如果trainer模块不存在，创建一个模拟的trainer
    class MeshTrainer:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_mesh_name(cls, *args, **kwargs):
            return cls()

        def train(self, *args, **kwargs):
            return {"episode_rewards": [], "message": "训练模拟完成"}

        def set_current_mesh_name(self, mesh_name: str):
            pass

try:
    from src.utils import MeshImporter
except ImportError:
    # 如果utils模块不存在，创建一个模拟的importer
    class MeshImporter:
        def __init__(self, *args, **kwargs):
            pass

        def list_available_meshes(self, *args, **kwargs):
            return ["simple_square", "triangle", "pentagon"]

        def get_mesh_info(self, *args, **kwargs):
            return {"vertex_count": 4, "file_size": 100, "exists": True}


class TrainingManager:
    """
    管理异步训练会话

    现在集成了训练历史管理功能，会自动为每次训练创建唯一ID并保存详细历史记录
    """

    def __init__(self) -> None:
        self._trainer: Optional[MeshTrainer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status: str = "idle"
        self._stats: Optional[Dict[str, Any]] = None
        self.importer = MeshImporter()

    @property
    def running(self) -> bool:
        """检查训练是否正在运行"""
        return self._thread is not None and self._thread.is_alive()

    def start_training(
            self,
            mesh_name: Optional[str] = None,
            subfolder: str = "mesh",
            max_episodes: Optional[int] = None,
            max_steps: Optional[int] = None,
            description: Optional[str] = None,  # 新增参数：训练描述
    ) -> None:
        """
        启动训练过程

        Args:
            mesh_name: 网格名称
            subfolder: 子文件夹名称
            max_episodes: 最大训练轮数
            max_steps: 每轮最大步数
            description: 训练描述

        Raises:
            RuntimeError: 如果训练已在运行
        """
        if self.running:
            raise RuntimeError("Training already running")

        # 创建训练器
        try:
            if mesh_name is None:
                self._trainer = MeshTrainer()
            else:
                self._trainer = MeshTrainer.from_mesh_name(mesh_name, subfolder=subfolder)
                # 设置当前mesh名称用于历史记录
                self._trainer.set_current_mesh_name(mesh_name)
        except Exception as e:
            # 如果创建训练器失败，使用模拟训练器
            print(f"警告: 无法创建实际训练器，使用模拟训练器。错误: {e}")
            self._trainer = MeshTrainer()
            if mesh_name and hasattr(self._trainer, 'set_current_mesh_name'):
                self._trainer.set_current_mesh_name(mesh_name)

        # 添加回调函数来实时更新统计信息
        self._trainer.add_episode_callback(self._update_stats_callback)

        self._stop_event.clear()
        # 初始化统计信息为可JSON序列化的格式
        self._stats = {
            'episode': 0,
            'total_steps': 0,
            'episode_reward': 0.0,
            'average_reward': 0.0,
            'episode_length': 0,
            'boundary_vertices': 0,
            'buffer_size': 0,
            'mesh_data': {},
            'boundary_vertices_data': [],
            'reference_point_info': None,
            'training_id': None  # 新增字段：训练ID
        }
        self._status = "running"

        def _run() -> None:
            """训练线程函数"""
            try:
                final_stats = self._trainer.train(
                    max_episodes=max_episodes if max_episodes is not None else 100,
                    max_steps=max_steps if max_steps is not None else 1000,
                    stop_event=self._stop_event,
                    description=description,  # 传递训练描述
                )
                # 训练完成后更新最终统计信息
                if self._stats:
                    self._stats.update(final_stats)
                    # 添加训练ID到最终统计信息中
                    if hasattr(self._trainer, 'history_manager'):
                        training_id = self._trainer.history_manager.get_current_training_id()
                        if training_id:
                            self._stats['training_id'] = training_id
                else:
                    self._stats = final_stats

                if self._stop_event.is_set():
                    self._status = "stopped"
                else:
                    self._status = "completed"
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                self._status = "error"
                self._stats = {"error": str(e)}

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _update_stats_callback(self, episode_data: Dict[str, Any]) -> None:
        """
        训练过程中的回调函数，用于实时更新统计信息

        Args:
            episode_data: episode完成时的数据
        """
        try:
            if self._stats is None:
                self._stats = {}

            # 处理mesh_data，将元组键转换为JSON字符串键以便序列化
            mesh_data = episode_data.get('mesh_data', {})
            serializable_mesh_data = {}
            try:
                # 使用json.dumps来确保键是有效的JSON格式
                import json
                for vertex, neighbors in mesh_data.items():
                    # 将顶点坐标元组转换为JSON数组格式的字符串, 并通过separators参数移除空格
                    vertex_key = json.dumps(list(vertex), separators=(',', ':'))

                    # 确保邻居列表也是可序列化的
                    serializable_neighbors = []
                    for neighbor in neighbors:
                        # 确保邻居坐标是Python原生浮点数列表
                        clean_neighbor = [float(coord) for coord in neighbor]
                        serializable_neighbors.append(clean_neighbor)

                    serializable_mesh_data[vertex_key] = serializable_neighbors
            except Exception as mesh_error:
                print(f"处理mesh_data时发生错误: {mesh_error}")
                serializable_mesh_data = {}

            # 处理boundary_vertices_data，确保是可序列化的
            boundary_vertices = episode_data.get('boundary_vertices', [])
            serializable_boundary_vertices = []
            try:
                for vertex in boundary_vertices:
                    # 确保顶点坐标是Python原生浮点数列表
                    clean_vertex = [float(coord) for coord in vertex]
                    serializable_boundary_vertices.append(clean_vertex)
            except Exception as boundary_error:
                print(f"处理boundary_vertices时发生错误: {boundary_error}")
                serializable_boundary_vertices = []

            # 处理参考点信息
            ref_info = episode_data.get('reference_point_info')
            try:
                serializable_ref_info = self._deep_clean_for_json(ref_info)
            except Exception as ref_err:
                print(f"处理reference_point_info时发生错误: {ref_err}")
                serializable_ref_info = None

            # 获取当前训练ID
            training_id = None
            if hasattr(self._trainer, 'history_manager'):
                training_id = self._trainer.history_manager.get_current_training_id()

            # 更新实时统计信息
            self._stats.update({
                'episode': episode_data.get('episode', 0),
                'total_steps': episode_data.get('total_steps', 0),
                'episode_reward': float(episode_data.get('episode_reward', 0.0)),
                'average_reward': float(episode_data.get('average_reward', 0.0)),
                'episode_length': episode_data.get('episode_length', 0),
                'boundary_vertices': episode_data.get('boundary_size', 0),
                'buffer_size': episode_data.get('buffer_size', 0),
                'mesh_data': serializable_mesh_data,
                'boundary_vertices_data': serializable_boundary_vertices,
                'reference_point_info': serializable_ref_info,
                'training_id': training_id  # 新增字段
            })

            # 添加最近的损失信息，确保是可序列化的浮点数
            if 'recent_actor_loss' in episode_data:
                self._stats['recent_actor_loss'] = float(episode_data['recent_actor_loss'])
            if 'recent_critic_loss' in episode_data:
                self._stats['recent_critic_loss'] = float(episode_data['recent_critic_loss'])
            if 'current_alpha' in episode_data:
                self._stats['current_alpha'] = float(episode_data['current_alpha'])

        except Exception as e:
            print(f"更新统计信息时发生错误: {e}")
            # 在出错时重置统计信息为安全的默认值
            self._stats = {
                'episode': 0,
                'total_steps': 0,
                'episode_reward': 0.0,
                'average_reward': 0.0,
                'episode_length': 0,
                'boundary_vertices': 0,
                'buffer_size': 0,
                'mesh_data': {},
                'boundary_vertices_data': [],
                'reference_point_info': None,
                'training_id': None
            }

    def stop_training(self) -> None:
        """停止训练过程"""
        try:
            if not self.running:
                return

            self._stop_event.set()
            self._status = "stopping"

            # 移除回调函数
            if self._trainer and hasattr(self._trainer, 'remove_episode_callback'):
                try:
                    self._trainer.remove_episode_callback(self._update_stats_callback)
                except Exception as e:
                    print(f"移除回调函数时发生错误: {e}")

            # 等待训练线程结束（最多等待5秒）
            if self._thread:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    print("警告: 训练线程未能在5秒内正常结束")

            # 确保状态被正确设置
            self._status = "stopped"

        except Exception as e:
            print(f"停止训练时发生错误: {e}")
            self._status = "error"

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
                    print(f"清理字典项时出错: {e}, key: {key}, value: {value}")
                    cleaned_dict[str(key)] = None
            return cleaned_dict
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for item in data:
                try:
                    cleaned_list.append(self._deep_clean_for_json(item))
                except Exception as e:
                    print(f"清理列表项时出错: {e}, item: {item}")
                    cleaned_list.append(None)
            return cleaned_list
        else:
            # 对于其他类型，尝试转换为字符串
            try:
                return str(data)
            except:
                return None

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前训练状态

        Returns:
            包含训练状态信息的字典
        """
        try:
            # 检查线程状态
            actual_running = self._thread is not None and self._thread.is_alive()

            # 如果线程已停止但状态仍然显示运行中，更新状态
            if not actual_running and self._status == "running":
                self._status = "stopped"

            # 确保统计信息是可序列化的
            safe_stats = self._stats if self._stats is not None else {}

            # 验证统计信息的类型，进行深度清理
            if isinstance(safe_stats, dict):
                # 递归清理所有数据，确保JSON安全
                safe_stats = self._deep_clean_for_json(safe_stats)

            return {
                "running": actual_running,
                "status": self._status,
                "stats": safe_stats,
            }

        except Exception as e:
            print(f"获取状态时发生错误: {e}")
            # 返回安全的默认状态
            return {
                "running": False,
                "status": "error",
                "stats": {
                    'episode': 0,
                    'total_steps': 0,
                    'episode_reward': 0.0,
                    'average_reward': 0.0,
                    'episode_length': 0,
                    'boundary_vertices': 0,
                    'buffer_size': 0,
                    'mesh_data': {},
                    'boundary_vertices_data': [],
                    'reference_point_info': None,
                    'training_id': None
                }
            }

    # 新增方法：历史管理相关
    def get_training_history(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取训练历史信息

        Args:
            training_id: 训练ID，如果为None则返回当前训练信息

        Returns:
            Dict[str, Any]: 训练历史信息
        """
        if self._trainer and hasattr(self._trainer, 'get_training_history'):
            return self._trainer.get_training_history(training_id)
        else:
            return {"error": "训练器不支持历史查询功能"}

    def list_all_training_history(self):
        """
        列出所有历史训练记录

        Returns:
            List[Dict[str, Any]]: 所有训练记录的列表
        """
        if self._trainer and hasattr(self._trainer, 'list_all_training_history'):
            return self._trainer.list_all_training_history()
        else:
            return []

    def export_training_summary(self, training_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """
        导出指定训练的摘要报告

        Args:
            training_id: 训练ID
            export_path: 导出路径

        Returns:
            Optional[str]: 导出的文件路径，失败时返回None
        """
        if self._trainer and hasattr(self._trainer, 'export_training_summary'):
            return self._trainer.export_training_summary(training_id, export_path)
        else:
            print("训练器不支持导出功能")
            return None

    def get_current_training_id(self) -> Optional[str]:
        """
        获取当前训练会话ID

        Returns:
            Optional[str]: 当前训练ID，如果没有活动会话则返回None
        """
        if self._trainer and hasattr(self._trainer, 'history_manager'):
            return self._trainer.history_manager.get_current_training_id()
        else:
            return None
