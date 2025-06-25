import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List


class Actor(nn.Module):
    """
    Actor network for SAC algorithm.
    Outputs action distribution parameters (mean and std).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = [128, 128, 128],
                 log_std_min: float = -20, log_std_max: float = 2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build network layers
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
            state: Input state tensor

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
            state: Input state tensor
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


class StateEncoder(nn.Module):
    """
    State encoder to process raw state components into feature vector.
    """

    def __init__(self, n_neighbors: int = 2, n_fan_points: int = 3,
                 output_dim: int = 64):
        super(StateEncoder, self).__init__()

        self.n_neighbors = n_neighbors
        self.n_fan_points = n_fan_points

        # Calculate input dimensions
        # ref_vertex (2) + left_neighbors (n*2) + right_neighbors (n*2) +
        # fan_points (g*2) + area_ratio (1)
        input_dim = 2 + (n_neighbors * 2) + (n_neighbors * 2) + (n_fan_points * 2) + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state_dict: dict) -> torch.Tensor:
        """
        Encode state dictionary into feature vector.

        Args:
            state_dict: Dictionary containing state components

        Returns:
            Encoded state tensor
        """
        # Extract components
        ref_vertex = state_dict['ref_vertex']
        left_neighbors = state_dict['left_neighbors']
        right_neighbors = state_dict['right_neighbors']
        fan_points = state_dict['fan_points']
        area_ratio = state_dict['area_ratio']

        # Handle batch dimension - check if we're processing a batch
        if ref_vertex.dim() > 1:  # Batch processing
            batch_size = ref_vertex.shape[0]

            # Flatten each component properly for batch processing
            ref_vertex_flat = ref_vertex.view(batch_size, -1)
            left_neighbors_flat = left_neighbors.view(batch_size, -1)
            right_neighbors_flat = right_neighbors.view(batch_size, -1)
            fan_points_flat = fan_points.view(batch_size, -1)

            # Handle area_ratio for batch processing
            if area_ratio.dim() == 1:
                area_ratio_flat = area_ratio.unsqueeze(1)  # [batch_size, 1]
            else:
                area_ratio_flat = area_ratio.view(batch_size, -1)

        else:  # Single sample processing
            # Flatten components
            ref_vertex_flat = ref_vertex.flatten()
            left_neighbors_flat = left_neighbors.flatten()
            right_neighbors_flat = right_neighbors.flatten()
            fan_points_flat = fan_points.flatten()

            # Handle area_ratio for single sample
            if area_ratio.dim() == 0:
                area_ratio_flat = area_ratio.unsqueeze(0)
            else:
                area_ratio_flat = area_ratio.flatten()

            # Add batch dimension for consistency
            ref_vertex_flat = ref_vertex_flat.unsqueeze(0)
            left_neighbors_flat = left_neighbors_flat.unsqueeze(0)
            right_neighbors_flat = right_neighbors_flat.unsqueeze(0)
            fan_points_flat = fan_points_flat.unsqueeze(0)
            area_ratio_flat = area_ratio_flat.unsqueeze(0)

        # Concatenate all components
        state_vector = torch.cat([
            ref_vertex_flat,
            left_neighbors_flat,
            right_neighbors_flat,
            fan_points_flat,
            area_ratio_flat
        ], dim=1)

        # Encode
        encoded_state = self.encoder(state_vector)

        # Remove batch dimension if input was single sample
        if ref_vertex.dim() == 1:
            encoded_state = encoded_state.squeeze(0)

        return encoded_state


class ActionDecoder(nn.Module):
    """
    Action decoder to convert network output to mesh generation actions.
    """

    def __init__(self, input_dim: int):
        super(ActionDecoder, self).__init__()

        # Output: [type_prob, x_coord, y_coord]
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # type_prob, x, y
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """
        Decode action features to action vector.

        Args:
            action_features: Input action features

        Returns:
            Action vector [type_prob, x, y]
        """
        action = self.decoder(action_features)

        # Apply appropriate activations
        # type_prob: sigmoid to [0, 1]
        # x, y: tanh to [-1, 1] (will be scaled later)
        action[:, 0] = torch.sigmoid(action[:, 0])
        action[:, 1:] = torch.tanh(action[:, 1:])

        return action


class MeshActor(nn.Module):
    """
    Specialized actor network for mesh generation.
    Combines state encoding and action decoding.
    """

    def __init__(self, n_neighbors: int = 2, n_fan_points: int = 3,
                 hidden_layers: List[int] = [128, 128, 128],
                 log_std_min: float = -20, log_std_max: float = 2):
        super(MeshActor, self).__init__()

        self.state_encoder = StateEncoder(n_neighbors, n_fan_points, hidden_layers[0])

        # Actor network
        self.actor = Actor(hidden_layers[0], 3, hidden_layers[1:], log_std_min, log_std_max)

    def forward(self, state_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mesh actor.

        Args:
            state_dict: State dictionary

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        encoded_state = self.state_encoder(state_dict)
        return self.actor(encoded_state)

    def sample(self, state_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        encoded_state = self.state_encoder(state_dict)
        return self.actor.sample(encoded_state)

    def get_action(self, state_dict: dict, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation."""
        encoded_state = self.state_encoder(state_dict)
        return self.actor.get_action(encoded_state, deterministic)


class MeshCritic(nn.Module):
    """
    Specialized critic network for mesh generation.
    """

    def __init__(self, n_neighbors: int = 2, n_fan_points: int = 3,
                 hidden_layers: List[int] = [128, 128, 128]):
        super(MeshCritic, self).__init__()

        self.state_encoder = StateEncoder(n_neighbors, n_fan_points, hidden_layers[0])

        # Double critic
        self.critic = DoubleCritic(hidden_layers[0], 3, hidden_layers[1:])

    def forward(self, state_dict: dict, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mesh critic.

        Args:
            state_dict: State dictionary
            action: Action tensor

        Returns:
            Tuple of (q1_value, q2_value)
        """
        encoded_state = self.state_encoder(state_dict)
        return self.critic(encoded_state, action)

    def q1(self, state_dict: dict, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic only."""
        encoded_state = self.state_encoder(state_dict)
        return self.critic.q1(encoded_state, action)


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