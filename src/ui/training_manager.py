import threading
from typing import Optional, Dict, Any, List

# 直接导入，不使用try-except来掩盖错误
from src.rl.trainer import MeshTrainer
from src.utils import MeshImporter


class TrainingManager:
    """管理异步训练会话"""

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
    ) -> None:
        """
        启动训练过程

        Args:
            mesh_name: 网格名称
            subfolder: 子文件夹名称
            max_episodes: 最大训练轮数
            max_steps: 每轮最大步数

        Raises:
            RuntimeError: 如果训练已在运行
        """
        if self.running:
            raise RuntimeError("Training already running")

        # 创建训练器 - 如果失败就让程序报错，不掩盖问题
        if mesh_name is None:
            self._trainer = MeshTrainer()
        else:
            self._trainer = MeshTrainer.from_mesh_name(mesh_name, subfolder=subfolder)

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
            'reference_point_info': None
        }
        self._status = "running"

        def _run() -> None:
            """训练线程函数"""
            try:
                final_stats = self._trainer.train(
                    max_episodes=max_episodes if max_episodes is not None else 100,
                    max_steps=max_steps if max_steps is not None else 1000,
                    stop_event=self._stop_event,
                )
                # 训练完成后更新最终统计信息
                if self._stats:
                    self._stats.update(final_stats)
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
                # 重新抛出异常，不掩盖错误
                raise

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
                        if isinstance(neighbor, (tuple, list)) and len(neighbor) == 2:
                            neighbor_key = json.dumps(list(neighbor), separators=(',', ':'))
                            serializable_neighbors.append(neighbor_key)
                        else:
                            # 如果neighbor不是坐标格式，直接添加
                            serializable_neighbors.append(str(neighbor))

                    serializable_mesh_data[vertex_key] = serializable_neighbors

            except Exception as json_error:
                print(f"处理mesh_data时发生错误: {json_error}")
                # 如果转换失败，使用空字典
                serializable_mesh_data = {}

            # 获取边界顶点数据并确保可序列化
            boundary_vertices_data = episode_data.get('boundary_vertices', [])
            serializable_boundary_data = []
            try:
                for vertex in boundary_vertices_data:
                    if isinstance(vertex, (tuple, list)) and len(vertex) == 2:
                        serializable_boundary_data.append([float(vertex[0]), float(vertex[1])])
                    else:
                        # 跳过不符合格式的顶点
                        continue
            except Exception as boundary_error:
                print(f"处理boundary_vertices时发生错误: {boundary_error}")
                serializable_boundary_data = []

            # 安全地更新统计信息
            safe_episode_data = self._deep_clean_for_json(episode_data)

            # 更新核心统计信息 - 按照前端期望的字段名
            self._stats.update({
                'episode': int(safe_episode_data.get('episode', 0)),
                'total_steps': int(safe_episode_data.get('total_steps', 0)),
                'episode_reward': float(safe_episode_data.get('episode_reward', 0.0)),
                'average_reward': float(safe_episode_data.get('average_reward', 0.0)),
                'episode_length': int(safe_episode_data.get('episode_length', 0)),
                'boundary_vertices': len(serializable_boundary_data),
                'buffer_size': int(safe_episode_data.get('buffer_size', 0)),
                # 前端期望的确切字段名
                'mesh_data': serializable_mesh_data,
                'boundary_vertices_data': serializable_boundary_data,
                'reference_point_info': safe_episode_data.get('reference_point_info')
            })

            # 可选的额外统计信息
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
                'reference_point_info': None
            }

    def stop_training(self) -> None:
        """停止训练过程"""
        if not self.running:
            return

        self._stop_event.set()
        self._status = "stopping"

        # 移除回调函数
        if self._trainer:
            self._trainer.remove_episode_callback(self._update_stats_callback)

        # 等待训练线程结束（最多等待5秒）
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                print("警告: 训练线程未能在5秒内正常结束")

        # 确保状态被正确设置
        self._status = "stopped"

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
                # 确保键是字符串
                clean_key = str(key)
                clean_value = self._deep_clean_for_json(value)
                cleaned_dict[clean_key] = clean_value
            return cleaned_dict
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for item in data:
                clean_item = self._deep_clean_for_json(item)
                cleaned_list.append(clean_item)
            return cleaned_list
        else:
            # 对于其他类型，尝试转换为字符串
            try:
                return str(data)
            except:
                return "无法序列化的数据"

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
                    'reference_point_info': None
                }
            }

    def get_training_status(self) -> str:
        """获取训练状态"""
        return self._status

    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """获取训练统计信息"""
        return self._stats

    def list_available_meshes(self, subfolder: str = "mesh") -> List[str]:
        """
        列出可用的网格文件

        Args:
            subfolder: 子文件夹名称

        Returns:
            List[str]: 可用的网格文件名列表
        """
        return self.importer.list_available_meshes(subfolder)

    def get_mesh_info(self, mesh_name: str, subfolder: str = "mesh") -> dict:
        """
        获取网格文件信息

        Args:
            mesh_name: 网格文件名
            subfolder: 子文件夹名称

        Returns:
            dict: 网格文件信息
        """
        return self.importer.get_mesh_info(mesh_name, subfolder)
