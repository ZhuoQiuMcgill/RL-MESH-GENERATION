import os
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.agent.sb3_sac_agent import SB3SACAgent
from src.rl.config import load_config


class _EpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.infos: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                ep = info["episode"]
                self.rewards.append(float(ep["r"]))
                self.lengths.append(int(ep["l"]))
                self.infos.append(info)
        return True


class SB3SACTrainer:
    def __init__(self, env, device: str = "cuda", config=None):
        self.env = env
        self.device = device
        self.config = config if config is not None else load_config()
        self.agent = SB3SACAgent(env, device, self.config)
        self._cb = _EpisodeCallback()

    def train(self, total_timesteps: int):
        self.agent.learn(total_timesteps=total_timesteps, callback=self._cb)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)

    def load(self, path: str):
        self.agent.load(path)

    def evaluate(self, eval_env=None, n_eval_episodes: int = 10):
        env = eval_env if eval_env is not None else self.env
        return self.agent.evaluate(env, n_eval_episodes=n_eval_episodes)

    @property
    def average_reward(self) -> float:
        return float(np.mean(self._cb.rewards)) if self._cb.rewards else 0.0

    @property
    def total_steps(self) -> int:
        return int(self.agent.num_timesteps)

    @property
    def total_episodes(self) -> int:
        return len(self._cb.rewards)

    def get_episode_infos(self) -> List[Dict[str, Any]]:
        return list(self._cb.infos)

    def summary(self) -> Dict[str, Any]:
        return {
            "avg_reward": self.average_reward,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def get_status(self) -> Dict[str, Any]:
        latest_reward = self._cb.rewards[-1] if self._cb.rewards else None
        latest_length = self._cb.lengths[-1] if self._cb.lengths else None
        latest_info = self._cb.infos[-1] if self._cb.infos else None
        return {
            "timesteps": self.total_steps,
            "episodes": self.total_episodes,
            "latest_reward": latest_reward,
            "latest_length": latest_length,
            "avg_reward_100": float(np.mean(self._cb.rewards[-100:])) if self._cb.rewards else 0.0,
            "latest_info": latest_info,
        }
