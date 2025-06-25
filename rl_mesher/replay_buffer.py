import numpy as np
import torch
from typing import Dict, Tuple, Optional
import random
from collections import deque


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples.
    Optimized for SAC algorithm with state dictionaries.
    """

    def __init__(self, capacity: int, state_shape: Tuple, action_dim: int,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of state observations
            action_dim: Dimension of action space
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate memory for efficient storage
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float,
            next_state: torch.Tensor, done: bool):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.states[self.position] = state.to(self.device)
        self.actions[self.position] = action.to(self.device)
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state.to(self.device)
        self.dones[self.position] = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer. Size: {self.size}, Requested: {batch_size}")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        """Return current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size


class MeshReplayBuffer:
    """
    Specialized replay buffer for mesh generation with state dictionaries.
    """

    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")):
        """
        Initialize mesh replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def add(self, state_dict: Dict, action: np.ndarray, reward: float,
            next_state_dict: Dict, done: bool):
        """
        Add a transition to the buffer.

        Args:
            state_dict: Current state dictionary
            action: Action taken
            reward: Reward received
            next_state_dict: Next state dictionary
            done: Whether episode is done
        """
        # Convert numpy arrays to tensors and store
        state_tensor_dict = self._convert_state_dict_to_tensor(state_dict)
        next_state_tensor_dict = self._convert_state_dict_to_tensor(next_state_dict)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)

        transition = {
            'state': state_tensor_dict,
            'action': action_tensor,
            'reward': reward,
            'next_state': next_state_tensor_dict,
            'done': done
        }

        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batched transitions
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Size: {len(self.buffer)}, Requested: {batch_size}")

        # Sample random transitions
        transitions = random.sample(self.buffer, batch_size)

        # Batch the transitions
        batch = self._batch_transitions(transitions)

        return batch

    def _convert_state_dict_to_tensor(self, state_dict: Dict) -> Dict:
        """Convert state dictionary numpy arrays to tensors."""
        tensor_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Already a tensor, just move to correct device
                tensor_dict[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            else:
                # Scalar value
                tensor_dict[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        return tensor_dict

    def _batch_transitions(self, transitions: list) -> Dict:
        """Batch a list of transitions into a single batch dictionary."""
        batch = {
            'states': {},
            'actions': [],
            'rewards': [],
            'next_states': {},
            'dones': []
        }

        # Get all state keys from first transition
        state_keys = transitions[0]['state'].keys()

        # Initialize batched state dictionaries
        for key in state_keys:
            batch['states'][key] = []
            batch['next_states'][key] = []

        # Collect all components
        for transition in transitions:
            # Batch states
            for key in state_keys:
                batch['states'][key].append(transition['state'][key])
                batch['next_states'][key].append(transition['next_state'][key])

            # Batch other components
            batch['actions'].append(transition['action'])
            batch['rewards'].append(transition['reward'])
            batch['dones'].append(transition['done'])

        # Stack tensors
        for key in state_keys:
            batch['states'][key] = torch.stack(batch['states'][key])
            batch['next_states'][key] = torch.stack(batch['next_states'][key])

        batch['actions'] = torch.stack(batch['actions'])
        batch['rewards'] = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device).unsqueeze(1)
        batch['dones'] = torch.tensor(batch['dones'], dtype=torch.bool, device=self.device).unsqueeze(1)

        return batch

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()

    def save(self, filepath: str):
        """Save buffer to file."""
        torch.save(list(self.buffer), filepath)

    def load(self, filepath: str):
        """Load buffer from file."""
        self.buffer = deque(torch.load(filepath), maxlen=self.capacity)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer for improved sample efficiency.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent
            beta: Importance sampling correction exponent
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state_dict: Dict, action: np.ndarray, reward: float,
            next_state_dict: Dict, done: bool, priority: Optional[float] = None):
        """
        Add a transition with priority to the buffer.

        Args:
            state_dict: Current state dictionary
            action: Action taken
            reward: Reward received
            next_state_dict: Next state dictionary
            done: Whether episode is done
            priority: Transition priority (uses max priority if None)
        """
        # Convert state dictionaries to tensors
        state_tensor_dict = self._convert_state_dict_to_tensor(state_dict)
        next_state_tensor_dict = self._convert_state_dict_to_tensor(next_state_dict)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)

        transition = {
            'state': state_tensor_dict,
            'action': action_tensor,
            'reward': reward,
            'next_state': next_state_tensor_dict,
            'done': done
        }

        # Set priority
        if priority is None:
            priority = self.max_priority

        self.buffer.append(transition)
        self.priorities.append(priority)

        # Update max priority
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions using prioritized sampling.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (batch, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Size: {len(self.buffer)}, Requested: {batch_size}")

        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        batch = self._batch_transitions(transitions)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return batch, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _convert_state_dict_to_tensor(self, state_dict: Dict) -> Dict:
        """Convert state dictionary numpy arrays to tensors."""
        tensor_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Already a tensor, just move to correct device
                tensor_dict[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            else:
                # Scalar value
                tensor_dict[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        return tensor_dict

    def _batch_transitions(self, transitions: list) -> Dict:
        """Batch a list of transitions into a single batch dictionary."""
        batch = {
            'states': {},
            'actions': [],
            'rewards': [],
            'next_states': {},
            'dones': []
        }

        # Get all state keys from first transition
        state_keys = transitions[0]['state'].keys()

        # Initialize batched state dictionaries
        for key in state_keys:
            batch['states'][key] = []
            batch['next_states'][key] = []

        # Collect all components
        for transition in transitions:
            # Batch states
            for key in state_keys:
                batch['states'][key].append(transition['state'][key])
                batch['next_states'][key].append(transition['next_state'][key])

            # Batch other components
            batch['actions'].append(transition['action'])
            batch['rewards'].append(transition['reward'])
            batch['dones'].append(transition['done'])

        # Stack tensors
        for key in state_keys:
            batch['states'][key] = torch.stack(batch['states'][key])
            batch['next_states'][key] = torch.stack(batch['next_states'][key])

        batch['actions'] = torch.stack(batch['actions'])
        batch['rewards'] = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device).unsqueeze(1)
        batch['dones'] = torch.tensor(batch['dones'], dtype=torch.bool, device=self.device).unsqueeze(1)

        return batch

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)