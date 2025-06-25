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

# Import with error handling
_import_errors = []

try:
    from .environment import MeshEnv, MultiDomainMeshEnv
except ImportError as e:
    _import_errors.append(f"environment: {e}")
    MeshEnv = None
    MultiDomainMeshEnv = None

try:
    from .agent import SACAgent, MeshSACTrainer
except ImportError as e:
    _import_errors.append(f"agent: {e}")
    SACAgent = None
    MeshSACTrainer = None

try:
    from .networks import MeshActor, MeshCritic, Actor, Critic, DoubleCritic
except ImportError as e:
    _import_errors.append(f"networks: {e}")
    MeshActor = None
    MeshCritic = None
    Actor = None
    Critic = None
    DoubleCritic = None

try:
    from .replay_buffer import MeshReplayBuffer, ReplayBuffer, PrioritizedReplayBuffer
except ImportError as e:
    _import_errors.append(f"replay_buffer: {e}")
    MeshReplayBuffer = None
    ReplayBuffer = None
    PrioritizedReplayBuffer = None

# Utility modules - import with fallback
try:
    from . import utils
    from .utils import geometry, visualization
except ImportError as e:
    _import_errors.append(f"utils: {e}")
    utils = None
    geometry = None
    visualization = None

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

# Export successful imports only
__all__ = []

if MeshEnv is not None:
    __all__.extend(["MeshEnv", "MultiDomainMeshEnv"])

if SACAgent is not None:
    __all__.extend(["SACAgent", "MeshSACTrainer"])

if MeshActor is not None:
    __all__.extend(["MeshActor", "MeshCritic", "Actor", "Critic", "DoubleCritic"])

if MeshReplayBuffer is not None:
    __all__.extend(["MeshReplayBuffer", "ReplayBuffer", "PrioritizedReplayBuffer"])

if utils is not None:
    __all__.extend(["utils", "geometry", "visualization"])


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

    if _import_errors:
        print("⚠️  Import warnings:")
        for error in _import_errors:
            print(f"  - {error}")
        print("=" * 60)


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
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Warning: Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True


def _initialize_package():
    """Initialize package and perform basic checks."""
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("Warning: This package requires Python 3.8 or higher")

    # Only check dependencies if there were no import errors
    if not _import_errors:
        check_dependencies()


# Run initialization
_initialize_package()

# Print import status for debugging
if _import_errors and __name__ == "__main__":
    print("Import errors detected:")
    for error in _import_errors:
        print(f"  - {error}")
