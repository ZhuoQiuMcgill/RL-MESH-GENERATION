"""
Checkpoint管理器

负责管理训练检查点的加载、验证和信息获取功能。
"""

import os
import json
import torch
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: str = None):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点目录，默认为项目根目录下的data/checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), "data", "checkpoints")
        self.logger = logging.getLogger(__name__)

        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的检查点

        Returns:
            List[Dict[str, Any]]: 检查点信息列表
        """
        checkpoints = []

        if not os.path.exists(self.checkpoint_dir):
            return checkpoints

        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.pth'):
                    checkpoint_name = filename[:-4]  # 移除.pth扩展名
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)

                    # 获取检查点信息
                    checkpoint_info = self.get_checkpoint_info(checkpoint_name)
                    if checkpoint_info:
                        checkpoints.append(checkpoint_info)

        except Exception as e:
            self.logger.error(f"列出检查点时发生错误: {e}")

        # 按修改时间倒序排列
        checkpoints.sort(key=lambda x: x.get('modified_time', 0), reverse=True)
        return checkpoints

    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        获取检查点的详细信息

        Args:
            checkpoint_name: 检查点名称（不包含.pth扩展名）

        Returns:
            Optional[Dict[str, Any]]: 检查点信息，如果检查点不存在或无效则返回None
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            return None

        try:
            # 获取文件信息
            file_stat = os.stat(checkpoint_path)
            file_size = file_stat.st_size
            modified_time = file_stat.st_mtime
            modified_datetime = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")

            # 尝试加载检查点以验证其有效性并获取训练信息
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            # 提取训练信息
            training_timesteps = checkpoint_data.get('training_timesteps', 0)
            learning_rate = checkpoint_data.get('learning_rate', 0.0)
            gamma = checkpoint_data.get('gamma', 0.0)
            tau = checkpoint_data.get('tau', 0.0)
            has_replay_buffer = checkpoint_data.get('has_replay_buffer', False)

            # 检查是否有必要的网络参数
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
                'is_valid': has_actor and has_critic,  # 至少需要actor和critic
                'checkpoint_path': checkpoint_path
            }

        except Exception as e:
            self.logger.error(f"获取检查点信息失败 {checkpoint_name}: {e}")
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
        验证检查点是否有效

        Args:
            checkpoint_name: 检查点名称

        Returns:
            bool: 检查点是否有效
        """
        checkpoint_info = self.get_checkpoint_info(checkpoint_name)
        return checkpoint_info is not None and checkpoint_info.get('is_valid', False)

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        加载检查点数据

        Args:
            checkpoint_name: 检查点名称

        Returns:
            Optional[Dict[str, Any]]: 检查点数据，加载失败时返回None
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            self.logger.error(f"检查点文件不存在: {checkpoint_path}")
            return None

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"成功加载检查点: {checkpoint_name}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"加载检查点失败 {checkpoint_name}: {e}")
            return None

    def load_replay_buffer(self, checkpoint_name: str) -> Optional[Any]:
        """
        加载检查点对应的经验回放缓冲区

        Args:
            checkpoint_name: 检查点名称

        Returns:
            Optional[Any]: 经验回放缓冲区，加载失败时返回None
        """
        # 首先检查checkpoint中是否标记有replay buffer
        checkpoint_info = self.get_checkpoint_info(checkpoint_name)
        if not checkpoint_info or not checkpoint_info.get('has_replay_buffer', False):
            return None

        # 尝试在原始训练目录中查找replay buffer
        # 假设checkpoint名称包含training_id信息
        history_dir = os.path.join(os.getcwd(), "data", "history")

        # 尝试多种可能的replay buffer文件路径
        possible_paths = [
            os.path.join(self.checkpoint_dir, f"{checkpoint_name}_replay_buffer.pkl"),
            os.path.join(history_dir, checkpoint_name, "model", f"{checkpoint_name}_replay_buffer.pkl")
        ]

        for buffer_path in possible_paths:
            if os.path.exists(buffer_path):
                try:
                    with open(buffer_path, 'rb') as f:
                        replay_buffer = pickle.load(f)
                    self.logger.info(f"成功加载经验回放缓冲区: {buffer_path}")
                    return replay_buffer
                except Exception as e:
                    self.logger.warning(f"加载经验回放缓冲区失败 {buffer_path}: {e}")

        self.logger.warning(f"未找到检查点 {checkpoint_name} 对应的经验回放缓冲区")
        return None

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        删除指定的检查点

        Args:
            checkpoint_name: 检查点名称

        Returns:
            bool: 删除是否成功
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"要删除的检查点不存在: {checkpoint_path}")
            return False

        try:
            os.remove(checkpoint_path)
            self.logger.info(f"成功删除检查点: {checkpoint_name}")
            return True
        except Exception as e:
            self.logger.error(f"删除检查点失败 {checkpoint_name}: {e}")
            return False

    def copy_checkpoint_from_history(self, training_id: str, checkpoint_name: str = None) -> bool:
        """
        从历史训练目录复制检查点到checkpoints目录

        Args:
            training_id: 历史训练ID
            checkpoint_name: 目标检查点名称，如果不提供则使用training_id

        Returns:
            bool: 复制是否成功
        """
        if checkpoint_name is None:
            checkpoint_name = training_id

        history_dir = os.path.join(os.getcwd(), "data", "history")
        source_path = os.path.join(history_dir, training_id, "model", f"{training_id}_checkpoint.pth")
        target_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")

        if not os.path.exists(source_path):
            self.logger.error(f"源检查点文件不存在: {source_path}")
            return False

        try:
            import shutil
            shutil.copy2(source_path, target_path)

            # 同时复制replay buffer（如果存在）
            source_buffer_path = os.path.join(history_dir, training_id, "model", f"{training_id}_replay_buffer.pkl")
            if os.path.exists(source_buffer_path):
                target_buffer_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_replay_buffer.pkl")
                shutil.copy2(source_buffer_path, target_buffer_path)
                self.logger.info(f"同时复制了经验回放缓冲区: {target_buffer_path}")

            self.logger.info(f"成功复制检查点: {source_path} -> {target_path}")
            return True
        except Exception as e:
            self.logger.error(f"复制检查点失败: {e}")
            return False


# 全局检查点管理器实例
_checkpoint_manager = None


def get_checkpoint_manager() -> CheckpointManager:
    """
    获取全局检查点管理器实例

    Returns:
        CheckpointManager: 检查点管理器实例
    """
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
