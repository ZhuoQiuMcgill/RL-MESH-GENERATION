# Import agent-related classes
try:
    from .agent.sac_agent import SACAgent
except ImportError:
    # Skip if custom SAC is not implemented
    SACAgent = None

from .agent.sb3_sac_agent import SB3SACAgent

# Import network architectures
try:
    from .agent.network import Actor, Critic
except ImportError:
    # Skip if network architectures are not implemented
    Actor = None
    Critic = None


# Import trainer
from .training.sb3_sac_trainer import SB3SACTrainer

# Import environment
from .environment import MeshEnv

# Import configuration loading function
from .config import load_config

# Define module's public API
__all__ = [
    # Agents
    'SB3SACAgent',
    'SB3SACTrainer',

    # Environment
    'MeshEnv',

    # Configuration
    'load_config'
]

# Conditionally imported classes (add to __all__ if implemented)
if SACAgent is not None:
    __all__.append('SACAgent')

if Actor is not None and Critic is not None:
    __all__.extend(['Actor', 'Critic'])

# Version information
__version__ = '1.1.0'

# Module author information
__author__ = 'ZhuoQiuMcgill'

# Module description
__description__ = 'Mesh generation reinforcement learning module, providing SAC algorithm and training environment'
