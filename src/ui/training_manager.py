import threading
from typing import Optional, Dict, Any

from src.rl.trainer import MeshTrainer
from src.utils import MeshImporter


class TrainingManager:
    """Manage asynchronous training sessions."""

    def __init__(self) -> None:
        self._trainer: Optional[MeshTrainer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status: str = "idle"
        self._stats: Optional[Dict[str, Any]] = None
        self.importer = MeshImporter()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_training(
        self,
        mesh_name: Optional[str] = None,
        subfolder: str = "mesh",
        max_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        if self.running:
            raise RuntimeError("Training already running")

        if mesh_name is None:
            self._trainer = MeshTrainer()
        else:
            self._trainer = MeshTrainer.from_mesh_name(mesh_name, subfolder=subfolder)

        self._stop_event.clear()
        self._stats = None
        self._status = "running"

        def _run() -> None:
            try:
                self._stats = self._trainer.train(
                    max_episodes=max_episodes,
                    max_steps=max_steps,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    self._status = "stopped"
                else:
                    self._status = "completed"
            finally:
                pass

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop_training(self) -> None:
        if not self.running:
            return
        self._stop_event.set()

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "status": self._status,
            "stats": self._stats,
        }
