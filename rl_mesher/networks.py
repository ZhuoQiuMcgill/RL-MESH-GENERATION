import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List


class Actor(nn.Module):
    """
    Actor network for SAC algorithm.
    Simplified to match original author's approach with flat observation input.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = [128, 128, 128],
                 log_std_min: float = -20, log_std_max: float = 2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build network layers - simplified architecture matching original
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.

        Args:
            state: Input state tensor (flat array)

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        x = self.network(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy distribution.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create normal distribution
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Calculate log probability with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action for evaluation (with option for deterministic action).

        Args:
            state: Input state tensor
            deterministic: Whether to use deterministic (mean) action

        Returns:
            Action tensor
        """
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.sample()
            action = torch.tanh(x_t)

        return action


class Critic(nn.Module):
    """
    Critic network for SAC algorithm.
    Q-function that takes state and action as input.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = [128, 128, 128]):
        super(Critic, self).__init__()

        # Build network layers
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state: Input state tensor (flat array)
            action: Input action tensor

        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=1)
        q_value = self.network(x)
        return q_value


class DoubleCritic(nn.Module):
    """
    Double critic network for SAC algorithm.
    Contains two critic networks to reduce overestimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = [128, 128, 128]):
        super(DoubleCritic, self).__init__()

        self.critic1 = Critic(state_dim, action_dim, hidden_layers)
        self.critic2 = Critic(state_dim, action_dim, hidden_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critic networks.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Tuple of (q1_value, q2_value)
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic only."""
        return self.critic1(state, action)


class MeshActor(nn.Module):
    """
    Mesh-specific actor network that handles flat state input.
    Simplified to match original author's approach.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_layers: List[int] = [128, 128, 128],
                 log_std_min: float = -20, log_std_max: float = 2):
        super(MeshActor, self).__init__()

        # Direct actor network without state encoder since we use flat input
        self.actor = Actor(state_dim, action_dim, hidden_layers, log_std_min, log_std_max)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mesh actor.

        Args:
            state: Flat state tensor

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        return self.actor(state)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        return self.actor.sample(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation."""
        return self.actor.get_action(state, deterministic)


class MeshCritic(nn.Module):
    """
    Mesh-specific critic network that handles flat state input.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_layers: List[int] = [128, 128, 128]):
        super(MeshCritic, self).__init__()

        # Direct double critic without state encoder
        self.critic = DoubleCritic(state_dim, action_dim, hidden_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mesh critic.

        Args:
            state: Flat state tensor
            action: Action tensor

        Returns:
            Tuple of (q1_value, q2_value)
        """
        return self.critic(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic only."""
        return self.critic.q1(state, action)


class SimpleMeshActor(nn.Module):
    """
    Simple actor network matching original author's architecture more closely.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 128, num_layers: int = 3):
        super(SimpleMeshActor, self).__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.shared_net = nn.Sequential(*layers)

        # Separate heads for mean and std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        features = self.shared_net(state)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.sample()
            return torch.tanh(x_t)


class SimpleMeshCritic(nn.Module):
    """
    Simple critic network matching original author's architecture.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 128, num_layers: int = 3):
        super(SimpleMeshCritic, self).__init__()

        input_dim = state_dim + action_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.q_net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q_net(x)


def create_target_network(network: nn.Module) -> nn.Module:
    """
    Create target network by copying weights from source network.

    Args:
        network: Source network

    Returns:
        Target network with same architecture and copied weights
    """
    target_network = type(network)(**network.__dict__)
    target_network.load_state_dict(network.state_dict())
    return target_network


def soft_update_target_network(target_network: nn.Module,
                               source_network: nn.Module,
                               tau: float):
    """
    Soft update target network parameters.

    Args:
        target_network: Target network to update
        source_network: Source network to copy from
        tau: Soft update coefficient
    """
    for target_param, source_param in zip(target_network.parameters(),
                                          source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def count_parameters(network: nn.Module) -> int:
    """Count total number of trainable parameters in network."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)
