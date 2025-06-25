"""
RL-MESH-GENERATION: Reinforcement Learning for Automatic Quadrilateral Mesh Generation

This package implements a fully automatic quadrilateral mesh generation system
using Soft Actor-Critic (SAC) reinforcement learning algorithm.

Core Design Philosophy:
The system follows a clean separation between geometric expertise (Environment)
and decision-making intelligence (Agent), allowing for modular development
and easy extension to other RL algorithms or geometric domains.

Main Components:
- Environment: Handles all mesh generation logic and geometric computations
- Agent: Implements SAC algorithm for learning optimal meshing policies
- Networks: Neural network architectures for actor and critic
- Utils: Geometry computation and visualization utilities

Authors: Based on the research paper "Reinforcement learning for automatic
quadrilateral mesh generation: a soft actor-critic approach"
"""

__version__ = "1.0.0"
__author__ = "RL-Mesh-Generation Team"

# Core classes
from .environment import MeshEnv, MultiDomainMeshEnv
from .agent import SACAgent, MeshSACTrainer
from .networks import MeshActor, MeshCritic, Actor, Critic, DoubleCritic
from .replay_buffer import MeshReplayBuffer, ReplayBuffer, PrioritizedReplayBuffer

# Utility modules
from .utils import geometry, visualization

__all__ = [
    # Core environment
    "MeshEnv",
    "MultiDomainMeshEnv",

    # Agent and training
    "SACAgent",
    "MeshSACTrainer",

    # Neural networks
    "MeshActor",
    "MeshCritic",
    "Actor",
    "Critic",
    "DoubleCritic",

    # Experience replay
    "MeshReplayBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",

    # Utility modules
    "geometry",
    "visualization",
]

# Package information
__package_info__ = {
    "name": "RL-MESH-GENERATION",
    "description": "Reinforcement Learning for Automatic Quadrilateral Mesh Generation",
    "version": __version__,
    "author": __author__,
    "algorithm": "Soft Actor-Critic (SAC)",
    "domain": "Computational Geometry / Mesh Generation",
    "paper": "Reinforcement learning for automatic quadrilateral mesh generation: a soft actor-critic approach"
}


def get_package_info():
    """Get package information dictionary."""
    return __package_info__.copy()


def print_package_info():
    """Print package information in a formatted way."""
    info = get_package_info()
    print("=" * 60)
    print(f"📐 {info['name']}")
    print("=" * 60)
    print(f"Description: {info['description']}")
    print(f"Version: {info['version']}")
    print(f"Algorithm: {info['algorithm']}")
    print(f"Domain: {info['domain']}")
    print(f"Paper: {info['paper']}")
    print("=" * 60)


# Version compatibility check
def check_dependencies():
    """Check if all required dependencies are available."""
    import sys
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'shapely',
        'gymnasium', 'yaml', 'scipy'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Warning: Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True


# Initialize package
def _initialize_package():
    """Initialize package and perform basic checks."""
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("Warning: This package requires Python 3.8 or higher")

    # Check dependencies
    check_dependencies()


# Run initialization
_initialize_package()