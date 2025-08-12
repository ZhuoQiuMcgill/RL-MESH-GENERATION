"""
Checkpoint Manager

Responsible for managing the loading, validation, and information retrieval of training checkpoints.
"""

import os
import json
import torch
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class CheckpointManager:
    """Checkpoint Manager"""

    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Checkpoint directory, defaults to data/checkpoints under project root
        """
        self.checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), "data", "checkpoints")
        self.logger = logging.getLogger(__name__)

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints

        Returns:
            List[Dict[str, Any]]: List of checkpoint information
        """
        checkpoints = []

        if not os.path.exists(self.checkpoint_dir):
            return checkpoints

        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.pth'):
                    checkpoint_name = filename[:-4]  # Remove .pth extension
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)

                    # Get checkpoint information
                    checkpoint_info = self.get_checkpoint_info(checkpoint_name)
                    if checkpoint_info:
                        checkpoints.append(checkpoint_info)

        except Exception as e:
            self.logger.error(f"Error occurred while listing checkpoints: {e}")

        # Sort by modification time in descending order
        checkpoints.sort(key=lambda x: x.get('modified_time', 0), reverse=True)
        return checkpoints

    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed checkpoint information

        Args:
            checkpoint_name: Checkpoint name (without .pth extension)

        Returns:
            Optional[Dict[str, Any]]: Checkpoint information, returns None if checkpoint doesn't exist or is invalid
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            return None

        try:
            # Get file information
            file_stat = os.stat(checkpoint_path)
            file_size = file_stat.st_size
            modified_time = file_stat.st_mtime
            modified_datetime = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")

            # Try to load checkpoint to validate it and get training information
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            # Extract training information
            training_timesteps = checkpoint_data.get('training_timesteps', 0)
            learning_rate = checkpoint_data.get('learning_rate', 0.0)
            gamma = checkpoint_data.get('gamma', 0.0)
            tau = checkpoint_data.get('tau', 0.0)
            has_replay_buffer = checkpoint_data.get('has_replay_buffer', False)

            # Check for required network parameters
            has_actor = 'actor_state_dict' in checkpoint_data
            has_critic = 'critic_state_dict' in checkpoint_data
            has_critic_target = 'critic_target_state_dict' in checkpoint_data

            return {
                'name': checkpoint_name,
                'file_path': checkpoint_path,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'modified_time': modified_time,
                'modified_datetime': modified_datetime,
                'training_timesteps': training_timesteps,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'tau': tau,
                'has_replay_buffer': has_replay_buffer,
                'has_actor': has_actor,
                'has_critic': has_critic,
                'has_critic_target': has_critic_target,
                'is_valid': has_actor and has_critic,  # At least need actor and critic
                'checkpoint_path': checkpoint_path
            }

        except Exception as e:
            self.logger.error(f"Failed to get checkpoint info {checkpoint_name}: {e}")
            return {
                'name': checkpoint_name,
                'file_path': checkpoint_path,
                'file_size': file_stat.st_size if 'file_stat' in locals() else 0,
                'file_size_mb': 0,
                'modified_time': modified_time if 'modified_time' in locals() else 0,
                'modified_datetime': modified_datetime if 'modified_datetime' in locals() else "Unknown",
                'training_timesteps': 0,
                'learning_rate': 0.0,
                'gamma': 0.0,
                'tau': 0.0,
                'has_replay_buffer': False,
                'has_actor': False,
                'has_critic': False,
                'has_critic_target': False,
                'is_valid': False,
                'error': str(e),
                'checkpoint_path': checkpoint_path
            }

    def validate_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Validate if checkpoint is valid

        Args:
            checkpoint_name: Checkpoint name

        Returns:
            bool: Whether the checkpoint is valid
        """
        checkpoint_info = self.get_checkpoint_info(checkpoint_name)
        return checkpoint_info is not None and checkpoint_info.get('is_valid', False)

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data

        Args:
            checkpoint_name: Checkpoint name

        Returns:
            Optional[Dict[str, Any]]: Checkpoint data, returns None if loading fails
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return None

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"Successfully loaded checkpoint: {checkpoint_name}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_name}: {e}")
            return None

    def load_replay_buffer(self, checkpoint_name: str) -> Optional[Any]:
        """
        Load replay buffer corresponding to checkpoint

        Args:
            checkpoint_name: Checkpoint name

        Returns:
            Optional[Any]: Replay buffer, returns None if loading fails
        """
        # First check if checkpoint is marked as having replay buffer
        checkpoint_info = self.get_checkpoint_info(checkpoint_name)
        if not checkpoint_info or not checkpoint_info.get('has_replay_buffer', False):
            return None

        # Try to find replay buffer in original training directory
        # Assume checkpoint name contains training_id information
        history_dir = os.path.join(os.getcwd(), "data", "history")

        # Try multiple possible replay buffer file paths
        possible_paths = [
            os.path.join(self.checkpoint_dir, f"{checkpoint_name}_replay_buffer.pkl"),
            os.path.join(history_dir, checkpoint_name, "model", f"{checkpoint_name}_replay_buffer.pkl")
        ]

        for buffer_path in possible_paths:
            if os.path.exists(buffer_path):
                try:
                    with open(buffer_path, 'rb') as f:
                        replay_buffer = pickle.load(f)
                    self.logger.info(f"Successfully loaded replay buffer: {buffer_path}")
                    return replay_buffer
                except Exception as e:
                    self.logger.warning(f"Failed to load replay buffer {buffer_path}: {e}")

        self.logger.warning(f"Replay buffer not found for checkpoint {checkpoint_name}")
        return None

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete the specified checkpoint

        Args:
            checkpoint_name: Checkpoint name

        Returns:
            bool: Whether deletion was successful
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint to delete does not exist: {checkpoint_path}")
            return False

        try:
            os.remove(checkpoint_path)
            self.logger.info(f"Successfully deleted checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}")
            return False

    def copy_checkpoint_from_history(self, training_id: str, checkpoint_name: str = None) -> bool:
        """
        Copy checkpoint from history training directory to checkpoints directory

        Args:
            training_id: History training ID
            checkpoint_name: Target checkpoint name, uses training_id if not provided

        Returns:
            bool: Whether copy was successful
        """
        if checkpoint_name is None:
            checkpoint_name = training_id

        history_dir = os.path.join(os.getcwd(), "data", "history")
        source_path = os.path.join(history_dir, training_id, "model", f"{training_id}_checkpoint.pth")
        target_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(source_path):
            self.logger.error(f"Source checkpoint file does not exist: {source_path}")
            return False

        try:
            import shutil
            shutil.copy2(source_path, target_path)

            # Also copy replay buffer (if exists)
            source_buffer_path = os.path.join(history_dir, training_id, "model", f"{training_id}_replay_buffer.pkl")
            if os.path.exists(source_buffer_path):
                target_buffer_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_replay_buffer.pkl")
                shutil.copy2(source_buffer_path, target_buffer_path)
                self.logger.info(f"Also copied replay buffer: {target_buffer_path}")

            self.logger.info(f"Successfully copied checkpoint: {source_path} -> {target_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy checkpoint: {e}")
            return False


# Global checkpoint manager instance
_checkpoint_manager = None


def get_checkpoint_manager() -> CheckpointManager:
    """
    Get global checkpoint manager instance

    Returns:
        CheckpointManager: Checkpoint manager instance
    """
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
