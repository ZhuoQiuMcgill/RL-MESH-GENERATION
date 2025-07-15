"""
SB3 SAC trainer – revised version
Uses MeshEnv's true step counter (info["real_step"]) when available.
"""
# ---------- SB3 availability flag ---------------------------------
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except ImportError as err:
    SB3_AVAILABLE = False
    raise ImportError(
        "Stable‑Baselines3 未正确安装，或其依赖（gymnasium / torch）缺失。\n"
        "请在已激活的 conda 环境中执行：\n"
        "   conda install -c conda-forge stable-baselines3 gymnasium pytorch\n"
        "或：\n"
        "   pip install stable-baselines3\n"
    ) from err
# ------------------------------------------------------------------

import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Union

from .base_trainer import BaseTrainer
from ..environment import MeshEnv
from src.geometry import Boundary
from src.utils import MeshImporter

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as th


class SB3TrainingCallback(BaseCallback):
    def __init__(self, trainer_instance, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0
        self.last_log_timestep = 0

    def _on_training_start(self) -> None:
        total_ts = self.model.num_timesteps + self.model._total_timesteps
        print(f"SB3 training started, target timesteps: {total_ts}")

    def _on_step(self) -> bool:
        self.trainer.training_stats["total_steps"] = self.num_timesteps
        infos = self.locals.get("infos", [])
        print(infos)

        for info in infos:
            if "episode" not in info:
                continue

            self.episode_count += 1

            raw_len = int(info["episode"]["l"])
            ep_r = float(info["episode"]["r"])

            self.trainer._update_training_stats(ep_r, 0)

            episode_data = self.trainer._create_sb3_episode_data(
                episode=self.episode_count,
                episode_reward=ep_r,
                episode_length=0,
                raw_episode_length=raw_len,
                ref_info=info.get("reference_point_info", None),
            )

            if "mesh_data" in info:
                episode_data["mesh_data"] = info["mesh_data"]
            if "boundary_vertices_data" in info:
                episode_data["boundary_vertices_data"] = info["boundary_vertices_data"]
            if "generated_elements" in info:
                episode_data["generated_elements"] = info["generated_elements"]

            self.trainer._trigger_episode_callbacks(episode_data)
            if hasattr(self.trainer, "history_manager"):
                self.trainer.history_manager.cache_episode_data(episode_data)

        if (
                self.trainer.training_stats["total_steps"] - self.last_log_timestep
                >= self.trainer.log_frequency
        ):
            elapsed = time.time() - self.trainer.training_start_time
            print(
                f"SB3 progress – step {self.num_timesteps} | episode {self.episode_count} | "
                f"reward {self.trainer.training_stats.get('episode_reward', 0):.3f} | "
                f"mean {self.trainer.training_stats.get('average_reward', 0):.3f} | "
                f"elapsed {elapsed:.1f}s"
            )
            self.last_log_timestep = self.num_timesteps

        if hasattr(self.trainer, "stop_event") and self.trainer.stop_event.is_set():
            print("Stop event received, terminating SB3 training")
            return False

        return True


class SB3SACTrainer(BaseTrainer):
    def __init__(
            self,
            boundary_source: Union[Boundary, str, Dict[str, Any]] = None,
            config: Optional[Dict[str, Any]] = None,
            device: str = "auto",
    ):
        super().__init__(config)
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"SB3 SAC trainer initialised on {self.device}")

        training_cfg = self.config.get("training", {})
        self.log_frequency = training_cfg.get("log_frequency", 1000)

        self.initial_boundary = (
            self._validate_boundary_source(boundary_source) if boundary_source else None
        )

        self.agent = None
        self.env = None
        self.eval_env = None

    # ---------- helpers ----------

    def _validate_boundary_source(self, boundary_source) -> Optional[Boundary]:
        if boundary_source is None:
            return None
        if isinstance(boundary_source, Boundary):
            return boundary_source
        if isinstance(boundary_source, str):
            return MeshImporter().load_boundary_by_name(boundary_source)
        print(f"Unsupported boundary source type: {type(boundary_source)}")
        return None

    def _create_sb3_episode_data(
            self,
            episode: int,
            episode_reward: float,
            episode_length: int,
            raw_episode_length: int,
            ref_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_length": episode_length,  # true length from MeshEnv
            "raw_episode_length": raw_episode_length,  # Monitor length
            "total_steps": self.training_stats.get("total_steps", 0),
            "average_reward": self.training_stats.get("average_reward", 0.0),
            "mesh_data": {},
            "reference_point_info": ref_info or {},
            "boundary_vertices": self.initial_boundary.get_vertices()
            if self.initial_boundary
            else [],
            "training_backend": "sb3",
        }

    # ---------- environments ----------

    def _init_environments(self, max_steps: Optional[int] = None):
        if max_steps is None:
            max_steps = self.config.get("environment", {}).get("max_steps", None)

        def make_env() -> Monitor:
            env = MeshEnv(
                initial_boundary=self.initial_boundary,
                max_steps=max_steps,
                config=self.config,
            )
            return Monitor(env, info_keywords=("term_reason", "trunc_reason"))

        self.env = make_env()
        self.eval_env = make_env()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        print(f"Environment ready: state_dim={self.state_dim}, action_dim={self.action_dim}")

    # ---------- agent ----------

    def _initialize_agent(self):
        cfg = self.config.get("sb3_sac", {})
        policy_kwargs = dict(
            activation_fn=th.ReLU, net_arch=cfg.get("net_arch", [128, 128, 128])
        )

        self.agent = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=cfg.get("learning_rate", 3e-4),
            buffer_size=cfg.get("buffer_size", 1_000_000),
            learning_starts=cfg.get("learning_starts", 10_000),
            batch_size=cfg.get("batch_size", 100),
            tau=cfg.get("tau", 0.005),
            gamma=cfg.get("gamma", 0.99),
            train_freq=cfg.get("train_freq", 1),
            gradient_steps=cfg.get("gradient_steps", 1),
            policy_kwargs=policy_kwargs,
            verbose=cfg.get("verbose", 0),
            seed=cfg.get("seed", None),
            device=self.device,
        )
        print("SB3 SAC agent initialised")

    # ---------- training ----------

    def _prepare_for_training(self, boundary_source=None, **kwargs):
        if self.initial_boundary is None or boundary_source is not None:
            if boundary_source is not None:
                self.initial_boundary = self._validate_boundary_source(boundary_source)

            if self.initial_boundary is None:
                default_vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
                self.initial_boundary = Boundary(default_vertices)

        self._init_environments(kwargs.get("max_steps_per_episode", None))
        self._initialize_agent()
        print(f"Training setup complete, boundary vertices: {len(self.initial_boundary.get_vertices())}")

    def train(self, max_timesteps: int = 100_000, **kwargs) -> Dict[str, Any]:
        print(f"Starting SB3 SAC training, timesteps={max_timesteps}")

        if self.agent is None:
            self._prepare_for_training(kwargs.get("boundary_source", None), **kwargs)

        self.training_start_time = time.time()
        callback = SB3TrainingCallback(self)

        self.agent.learn(total_timesteps=max_timesteps, callback=callback, progress_bar=False)

        self.training_stats["training_time"] = time.time() - self.training_start_time
        self.history_manager.force_save_cache()
        self._save_final_model()

        stopped = self.stop_event.is_set()
        self.history_manager.finish_training_session(
            final_stats=self.training_stats, stopped_early=stopped
        )

        status = "stopped early" if stopped else "finished"
        print(
            f"SB3 training {status}: steps={self.training_stats['total_steps']}, "
            f"episodes={callback.episode_count}"
        )
        return self.training_stats

    # ---------- utilities ----------

    def _save_model(self, path: str):
        self.agent.save(path)

    def _load_model(self, path: str):
        self.agent.load(path, env=self.env)

    def _save_final_model(self):
        if not self.agent:
            return
        models_dir = os.path.join(os.getcwd(), "models", "sb3_sac")
        os.makedirs(models_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f"sb3_sac_final_{timestamp}")
        self.agent.save(model_path)
        print(f"Final model saved at: {model_path}.zip")

    def get_training_stats(self) -> Dict[str, Any]:
        stats = self.training_stats.copy()
        if self.agent:
            stats["num_timesteps"] = self.agent.num_timesteps
        return stats

    def stop_training(self):
        self.stop_event.set()
        print("Stop signal sent to SB3 training")
