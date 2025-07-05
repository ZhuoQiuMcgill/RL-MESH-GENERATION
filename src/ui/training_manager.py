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

        # 创建训练器
        try:
            if mesh_name is None:
                self._trainer = MeshTrainer()
            else:
                self._trainer = MeshTrainer.from_mesh_name(mesh_name, subfolder=subfolder)
        except Exception as e:
            # 如果创建训练器失败，使用模拟训练器
            print(f"警告: 无法创建实际训练器，使用模拟训练器。错误: {e}")
            self._trainer = MeshTrainer()

        # 添加回调函数来实时更新统计信息
        self._trainer.add_episode_callback(self._update_stats_callback)

        self._stop_event.clear()
        self._stats = {
            'episode': 0,
            'total_steps': 0,
            'episode_reward': 0.0,
            'average_reward': 0.0,
            'episode_length': 0,
            'boundary_vertices': 0,
            'buffer_size': 0,
            'mesh_data': {},
            'boundary_vertices_data': []
        }
        self._status = "running"

        def _run() -> None:
            """训练线程函数"""
            try:
                final_stats = self._trainer.train(
                    max_episodes=max_episodes or 100,
                    max_steps=max_steps or 1000,
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

            # 更新实时统计信息
            self._stats.update({
                'episode': episode_data.get('episode', 0),
                'total_steps': episode_data.get('total_steps', 0),
                'episode_reward': episode_data.get('episode_reward', 0.0),
                'average_reward': episode_data.get('average_reward', 0.0),
                'episode_length': episode_data.get('episode_length', 0),
                'boundary_vertices': episode_data.get('boundary_size', 0),
                'buffer_size': episode_data.get('buffer_size', 0),
                'mesh_data': episode_data.get('mesh_data', {}),
                'boundary_vertices_data': episode_data.get('boundary_vertices', [])
            })

            # 添加最近的损失信息
            if 'recent_actor_loss' in episode_data:
                self._stats['recent_actor_loss'] = episode_data['recent_actor_loss']
            if 'recent_critic_loss' in episode_data:
                self._stats['recent_critic_loss'] = episode_data['recent_critic_loss']
            if 'current_alpha' in episode_data:
                self._stats['current_alpha'] = episode_data['current_alpha']

        except Exception as e:
            print(f"更新统计信息时发生错误: {e}")

    def stop_training(self) -> None:
        """停止训练过程"""
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

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前训练状态

        Returns:
            包含训练状态信息的字典
        """
        return {
            "running": self.running,
            "status": self._status,
            "stats": self._stats,
        }
