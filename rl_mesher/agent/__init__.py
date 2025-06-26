"""
Agent module for RL-MESH-GENERATION.

This module provides the SAC agent implementation and training utilities
for reinforcement learning-based mesh generation.

Components:
- SACAgent: Core Soft Actor-Critic algorithm implementation
- MeshSACTrainer: High-level training loop and evaluation utilities
"""

from .sac_agent import SACAgent
from .trainer import MeshSACTrainer

__all__ = ['SACAgent', 'MeshSACTrainer']