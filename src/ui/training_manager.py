"""
训练管理器

负责管理强化学习训练会话，包括启动、停止、状态监控等功能。
提供与前端API的桥接功能。
支持从checkpoint继续训练。
"""

import os
import threading
import time
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from src.rl.config import load_config
from src.utils import create_default_importer

from src.utils.checkpoint_manager import get_checkpoint_manager
from src.rl.environment import MeshEnv
from src.rl.training.sb3_sac_trainer import SB3SACTrainer

BASE_DIR = os.getcwd()
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')


class TrainingManager:
    """
    训练管理器

    负责管理强化学习训练会话的生命周期，包括启动、停止、状态监控等。
    支持从checkpoint继续训练。
    """

    def __init__(self, config=None):
        """
        初始化训练管理器

        Args:
            config: 配置字典，如果为None则从config.yaml加载
        """
        self.config = config if config is not None else load_config()
        self.logger = logging.getLogger(__name__)

        # 训练状态
        self._is_running = False
        self._training_thread = None
        self._stop_event = threading.Event()

        # 训练实例
        self.trainer = None
        self.env = None
        self.mesh_importer = create_default_importer()
        self.checkpoint_manager = get_checkpoint_manager()

        # 当前训练会话信息
        self.current_training_config = None
        self.training_start_time = None
        self.training_id = None
        self.training_session_dir = None

        # checkpoint相关
        self.loaded_checkpoint_name = None
        self.checkpoint_data = None

        # 训练统计信息
        self.current_stats = {
            "episode": 0,
            "total_steps": 0,
            "episode_reward": 0.0,
            "average_reward": 0.0,
            "episode_length": 0,
            "boundary_vertices": 0,
            "buffer_size": 0,
            "training_id": None,
            "online_learning_mode": False,
            "recent_actor_loss": 0.0,
            "recent_critic_loss": 0.0,
            "current_alpha": 0.0,
            "avg_element_quality": 0.0,
            "mesh_data": {},
            "boundary_vertices_data": [],
            "reference_point_info": None,
            # 评估信息
            "eval_count": 0,
            "last_eval_reward": 0.0,
            "best_eval_reward": 0.0,
            "evaluation_frequency": 10,
            "n_eval_episodes": 10,
        }

        self.logger.info("训练管理器初始化完成")

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        启动训练会话

        Args:
            config: 训练配置
                - mesh_name: mesh文件名
                - subfolder: 子文件夹名称，默认为"mesh"
                - max_timesteps: 最大训练时间步数（前端参数，优先级最高）
                - max_steps: 每episode最大步数（前端参数，优先级最高）
                - description: 训练描述
                - checkpoint_name: checkpoint名称（可选，用于继续训练）

        Returns:
            Dict[str, Any]: 启动结果

        Raises:
            RuntimeError: 当训练已在运行时
            ValueError: 当配置无效时
        """
        if self._is_running:
            raise RuntimeError("Training already running")

        try:
            # 合并配置：前端参数优先，然后是config.yaml中的参数
            merged_config = self._merge_config(config)

            # 验证配置
            self._validate_config(merged_config)

            # 处理checkpoint - 修复：确保checkpoint_name正确传递
            checkpoint_name = merged_config.get("checkpoint_name")
            if checkpoint_name:
                self.logger.info(f"准备加载checkpoint: {checkpoint_name}")
                self._load_checkpoint(checkpoint_name)
            else:
                self.logger.info("未指定checkpoint，将从头开始训练")

            # 生成训练ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mesh_name = merged_config.get("mesh_name", "default")

            if checkpoint_name:
                self.training_id = f"continue_{checkpoint_name}_{timestamp}_{mesh_name}"
            else:
                self.training_id = f"sac_{timestamp}_{mesh_name}"

            # 创建训练会话目录
            self.training_session_dir = os.path.join(HISTORY_DIR, self.training_id)
            os.makedirs(os.path.join(self.training_session_dir, "plot"), exist_ok=True)
            os.makedirs(os.path.join(self.training_session_dir, "model"), exist_ok=True)
            os.makedirs(os.path.join(self.training_session_dir, "history"), exist_ok=True)

            # 更新统计信息中的training_id
            self.current_stats["training_id"] = self.training_id

            # 创建环境
            self._create_environment(merged_config)

            # 创建训练器
            self._create_trainer(merged_config)

            # 如果有checkpoint，加载到训练器中 - 修复：确保在训练器创建后立即应用
            if checkpoint_name and self.checkpoint_data:
                self.logger.info(f"应用checkpoint到训练器: {checkpoint_name}")
                self._apply_checkpoint_to_trainer()
            else:
                self.logger.info("无checkpoint数据需要应用")

            # 保存配置
            self.current_training_config = merged_config.copy()
            self.training_start_time = datetime.now()

            # 启动训练线程
            self._stop_event.clear()
            self._training_thread = threading.Thread(
                target=self._training_loop,
                args=(merged_config,),
                daemon=True
            )
            self._training_thread.start()

            self._is_running = True

            log_msg = f"训练已启动，ID: {self.training_id}"
            if checkpoint_name:
                log_msg += f"，使用checkpoint: {checkpoint_name}"
            self.logger.info(log_msg)

            return {
                "message": "training_started",
                "success": True,
                "config": merged_config,
                "training_id": self.training_id,
                "from_checkpoint": checkpoint_name is not None,
                "checkpoint_name": checkpoint_name
            }

        except Exception as e:
            self.logger.error(f"启动训练失败: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def stop_training(self) -> Dict[str, Any]:
        """
        停止当前训练会话

        Returns:
            Dict[str, Any]: 停止结果
        """
        if not self._is_running:
            return {
                "message": "stop_requested",
                "success": True
            }

        try:
            # 设置停止事件
            self._stop_event.set()

            # 等待训练线程结束
            if self._training_thread and self._training_thread.is_alive():
                self._training_thread.join(timeout=5.0)

            # 保存训练结果
            self._save_results()

            self._is_running = False

            self.logger.info("训练已停止")

            return {
                "message": "stop_requested",
                "success": True
            }

        except Exception as e:
            self.logger.error(f"停止训练失败: {e}")
            return {
                "message": f"stop_failed: {str(e)}",
                "success": False
            }

    def _sanitize_value(self, value):
        """
        清理数值，确保JSON序列化兼容
        """
        import math
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return 0.0
            return float(value)
        return value

    def _sanitize_dict(self, data):
        """
        递归清理字典中的所有数值
        """
        if isinstance(data, dict):
            return {k: self._sanitize_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_dict(item) for item in data]
        else:
            return self._sanitize_value(data)

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        if not self._is_running:
            return {
                "running": False,
                "status": "idle",
                "stats": None,
                "progress": None,
                "timestamp": time.time()
            }

        # 更新统计信息
        self._update_stats()

        # 构建进度信息
        progress_info = {
            "current_episode": self.current_stats.get("episode", 0),
            "total_steps": self.current_stats.get("total_steps", 0),
            "latest_reward": self.current_stats.get("episode_reward", 0.0),
            "average_reward": self.current_stats.get("average_reward", 0.0),
            "buffer_utilization": self.current_stats.get("buffer_size", 0)
        }

        result = {
            "running": True,
            "status": "running",
            "stats": self.current_stats.copy(),
            "progress": progress_info,
            "timestamp": time.time()
        }

        # 清理数据以确保JSON兼容性
        return self._sanitize_dict(result)

    def is_running(self) -> bool:
        """
        检查训练是否正在运行

        Returns:
            bool: 训练运行状态
        """
        return self._is_running

    def get_current_training_id(self) -> Optional[str]:
        """
        获取当前训练会话ID

        Returns:
            Optional[str]: 训练会话ID，如果没有运行的训练则返回None
        """
        return self.training_id if self._is_running else None

    def _load_checkpoint(self, checkpoint_name: str) -> None:
        """
        加载checkpoint数据

        Args:
            checkpoint_name: checkpoint名称

        Raises:
            ValueError: 当checkpoint无效时
        """
        # 验证checkpoint
        if not self.checkpoint_manager.validate_checkpoint(checkpoint_name):
            raise ValueError(f"Invalid checkpoint: {checkpoint_name}")

        # 加载checkpoint数据
        self.checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if not self.checkpoint_data:
            raise ValueError(f"Failed to load checkpoint: {checkpoint_name}")

        self.loaded_checkpoint_name = checkpoint_name
        self.logger.info(f"成功加载checkpoint: {checkpoint_name}")
        self.logger.info(f"Checkpoint包含的键: {list(self.checkpoint_data.keys())}")

    def _apply_checkpoint_to_trainer(self) -> None:
        """
        将checkpoint数据应用到训练器中 - 修复版本
        """
        if not self.checkpoint_data or not self.trainer:
            self.logger.warning("无checkpoint数据或训练器，跳过应用")
            return

        try:
            model = self.trainer.model
            self.logger.info("开始应用checkpoint到SB3模型")

            # 修复：使用正确的SB3模型结构来加载参数
            policy = model.policy

            # 加载Actor网络参数
            if 'actor_state_dict' in self.checkpoint_data:
                try:
                    policy.actor.load_state_dict(self.checkpoint_data['actor_state_dict'])
                    self.logger.info("✓ 成功加载Actor网络参数")
                except Exception as e:
                    self.logger.error(f"✗ 加载Actor网络参数失败: {e}")
                    raise

            # 加载Critic网络参数
            if 'critic_state_dict' in self.checkpoint_data:
                try:
                    policy.critic.load_state_dict(self.checkpoint_data['critic_state_dict'])
                    self.logger.info("✓ 成功加载Critic网络参数")
                except Exception as e:
                    self.logger.error(f"✗ 加载Critic网络参数失败: {e}")
                    raise

            # 加载Target Critic网络参数
            if 'critic_target_state_dict' in self.checkpoint_data:
                try:
                    policy.critic_target.load_state_dict(self.checkpoint_data['critic_target_state_dict'])
                    self.logger.info("✓ 成功加载Target Critic网络参数")
                except Exception as e:
                    self.logger.error(f"✗ 加载Target Critic网络参数失败: {e}")
                    raise

            # 加载温度参数α
            if 'log_ent_coef' in self.checkpoint_data:
                try:
                    import torch
                    if hasattr(policy, 'log_ent_coef'):
                        # 确保张量在正确的设备上
                        device = policy.log_ent_coef.device
                        loaded_coef = self.checkpoint_data['log_ent_coef'].to(device)
                        policy.log_ent_coef.data.copy_(loaded_coef)
                        self.logger.info("✓ 成功加载温度参数log_ent_coef")
                    else:
                        self.logger.warning("模型没有log_ent_coef属性")
                except Exception as e:
                    self.logger.error(f"✗ 加载温度参数失败: {e}")

            elif 'ent_coef' in self.checkpoint_data:
                try:
                    if hasattr(policy, 'ent_coef'):
                        policy.ent_coef = self.checkpoint_data['ent_coef']
                        self.logger.info("✓ 成功加载温度参数ent_coef")
                    else:
                        self.logger.warning("模型没有ent_coef属性")
                except Exception as e:
                    self.logger.error(f"✗ 加载温度参数失败: {e}")

            # 修复：正确加载优化器状态
            # SB3中优化器是在policy内部管理的，需要通过trainer的方法访问
            try:
                # 通过trainer获取优化器
                optimizers = self.trainer.get_policy_optimizers()
                if optimizers and len(optimizers) >= 2:
                    actor_optimizer = optimizers[0]  # 通常第一个是actor优化器
                    critic_optimizer = optimizers[1]  # 第二个是critic优化器

                    if 'actor_optimizer_state_dict' in self.checkpoint_data:
                        actor_optimizer.load_state_dict(self.checkpoint_data['actor_optimizer_state_dict'])
                        self.logger.info("✓ 成功加载Actor优化器状态")

                    if 'critic_optimizer_state_dict' in self.checkpoint_data:
                        critic_optimizer.load_state_dict(self.checkpoint_data['critic_optimizer_state_dict'])
                        self.logger.info("✓ 成功加载Critic优化器状态")
                else:
                    self.logger.warning("无法获取优化器，跳过优化器状态加载")
            except Exception as e:
                self.logger.warning(f"加载优化器状态时出现问题: {e}")

            # 加载温度参数优化器状态
            if 'ent_coef_optimizer_state_dict' in self.checkpoint_data:
                try:
                    if hasattr(policy, 'ent_coef_optimizer'):
                        policy.ent_coef_optimizer.load_state_dict(
                            self.checkpoint_data['ent_coef_optimizer_state_dict'])
                        self.logger.info("✓ 成功加载温度参数优化器状态")
                    else:
                        self.logger.warning("模型没有ent_coef_optimizer属性")
                except Exception as e:
                    self.logger.warning(f"加载温度参数优化器状态失败: {e}")

            # 尝试加载经验回放缓冲区 - 修复版本
            if self.checkpoint_data.get('has_replay_buffer', False):
                try:
                    replay_buffer = self.checkpoint_manager.load_replay_buffer(self.loaded_checkpoint_name)
                    if replay_buffer is not None and hasattr(model, 'replay_buffer'):
                        # Buffer兼容性检查和修复
                        success = self._apply_replay_buffer_safely(model, replay_buffer)
                        if success:
                            self.logger.info("✓ 成功加载并应用经验回放缓冲区")
                        else:
                            self.logger.warning("经验回放缓冲区不兼容，将使用空缓冲区开始训练")
                    else:
                        if replay_buffer is None:
                            self.logger.warning("经验回放缓冲区文件不存在或加载失败，将使用空缓冲区开始训练")
                        else:
                            self.logger.warning("模型不支持replay_buffer属性，跳过加载")
                except Exception as e:
                    self.logger.warning(f"加载经验回放缓冲区失败: {e}")
            else:
                self.logger.info("checkpoint未标记包含replay buffer，将使用空缓冲区开始训练")

            # 修复：设置正确的训练步数
            original_timesteps = self.checkpoint_data.get('training_timesteps', 0)
            if original_timesteps > 0:
                model.num_timesteps = original_timesteps
                self.logger.info(f"✓ 设置训练步数为: {original_timesteps}")

            # 验证加载是否成功 - 添加一些基本检查
            self._verify_checkpoint_loading()

            self.logger.info("🎉 Checkpoint应用完成！")

        except Exception as e:
            self.logger.error(f"应用checkpoint到训练器失败: {e}")
            self.logger.error(traceback.format_exc())
            raise ValueError(f"Failed to apply checkpoint to trainer: {e}")

    def _apply_replay_buffer_safely(self, model, replay_buffer) -> bool:
        """
        安全地应用replay buffer到模型，处理兼容性问题
        
        Args:
            model: SB3 SAC模型
            replay_buffer: 要加载的replay buffer
            
        Returns:
            bool: 是否成功应用
        """
        try:
            import torch
            
            # 1. 检查buffer类型兼容性
            if not hasattr(replay_buffer, 'buffer_size') or not hasattr(replay_buffer, 'pos'):
                self.logger.error("Replay buffer缺少必要属性，类型不兼容")
                return False
            
            # 2. 检查buffer大小兼容性
            current_buffer_size = model.replay_buffer.buffer_size if model.replay_buffer else 0
            loaded_buffer_size = replay_buffer.buffer_size
            
            if current_buffer_size != loaded_buffer_size:
                self.logger.warning(f"Buffer大小不匹配: 当前={current_buffer_size}, 加载={loaded_buffer_size}")
                # 如果大小不匹配，尝试调整
                if current_buffer_size > loaded_buffer_size:
                    self.logger.info("当前buffer更大，将加载的数据复制到新buffer中")
                    # 可以继续，数据会被复制到更大的buffer中
                else:
                    self.logger.error("当前buffer较小，无法容纳加载的数据")
                    return False
            
            # 3. 检查观测空间和动作空间兼容性
            if hasattr(replay_buffer, 'observations') and hasattr(model.replay_buffer, 'observations'):
                try:
                    loaded_obs_shape = replay_buffer.observations.shape
                    current_obs_shape = model.replay_buffer.observations.shape
                    
                    # 比较形状（除了第一维度buffer大小）
                    if loaded_obs_shape[1:] != current_obs_shape[1:]:
                        self.logger.error(f"观测空间不匹配: 加载={loaded_obs_shape}, 当前={current_obs_shape}")
                        return False
                except Exception as e:
                    self.logger.warning(f"检查观测空间时出错: {e}")
            
            # 4. 处理设备不匹配问题
            device = model.device
            if hasattr(replay_buffer, 'observations') and replay_buffer.observations is not None:
                try:
                    # 检查并转移到正确设备
                    if hasattr(replay_buffer.observations, 'device'):
                        if replay_buffer.observations.device != device:
                            self.logger.info(f"转移replay buffer到设备: {device}")
                            replay_buffer.observations = replay_buffer.observations.to(device)
                            
                    if hasattr(replay_buffer, 'actions') and replay_buffer.actions is not None:
                        if hasattr(replay_buffer.actions, 'device'):
                            replay_buffer.actions = replay_buffer.actions.to(device)
                            
                    if hasattr(replay_buffer, 'rewards') and replay_buffer.rewards is not None:
                        if hasattr(replay_buffer.rewards, 'device'):
                            replay_buffer.rewards = replay_buffer.rewards.to(device)
                            
                    if hasattr(replay_buffer, 'next_observations') and replay_buffer.next_observations is not None:
                        if hasattr(replay_buffer.next_observations, 'device'):
                            replay_buffer.next_observations = replay_buffer.next_observations.to(device)
                            
                    if hasattr(replay_buffer, 'dones') and replay_buffer.dones is not None:
                        if hasattr(replay_buffer.dones, 'device'):
                            replay_buffer.dones = replay_buffer.dones.to(device)
                            
                except Exception as e:
                    self.logger.warning(f"转移设备时出错: {e}")
            
            # 5. 应用replay buffer
            model.replay_buffer = replay_buffer
            
            # 6. 验证应用结果
            buffer_size = replay_buffer.size() if hasattr(replay_buffer, 'size') else 0
            buffer_pos = replay_buffer.pos if hasattr(replay_buffer, 'pos') else 0
            buffer_full = replay_buffer.full if hasattr(replay_buffer, 'full') else False
            
            self.logger.info(f"Replay buffer应用成功:")
            self.logger.info(f"  - 缓冲区大小: {buffer_size}")
            self.logger.info(f"  - 当前位置: {buffer_pos}")
            self.logger.info(f"  - 是否满: {buffer_full}")
            self.logger.info(f"  - 容量: {loaded_buffer_size}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"安全应用replay buffer失败: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _verify_checkpoint_loading(self) -> None:
        """
        验证checkpoint是否正确加载 - 新增方法
        """
        try:
            if self.trainer and self.trainer.model:
                model = self.trainer.model
                policy = model.policy

                # 检查网络参数是否存在
                actor_params = sum(p.numel() for p in policy.actor.parameters())
                critic_params = sum(p.numel() for p in policy.critic.parameters())

                self.logger.info(f"验证：Actor网络参数数量: {actor_params}")
                self.logger.info(f"验证：Critic网络参数数量: {critic_params}")
                self.logger.info(f"验证：当前训练步数: {model.num_timesteps}")

                if hasattr(policy, 'log_ent_coef'):
                    self.logger.info(f"验证：温度参数值: {policy.log_ent_coef.item()}")

        except Exception as e:
            self.logger.warning(f"验证checkpoint加载时出错: {e}")

    def _save_results(self) -> None:
        """保存训练结果"""
        if not self.training_session_dir or not self.trainer:
            return

        try:
            model_dir = os.path.join(self.training_session_dir, "model")

            # 保存SB3模型（zip格式，用于完整恢复）
            sb3_model_path = os.path.join(model_dir, f"{self.training_id}_sb3_model")
            self.trainer.save(sb3_model_path)
            self.logger.info(f"SB3模型已保存: {sb3_model_path}")

            # 保存PyTorch模型参数（用于迁移学习）
            import torch
            if hasattr(self.trainer, 'model') and self.trainer.model:
                model = self.trainer.model

                # 创建完整的checkpoint字典
                checkpoint = {
                    'training_timesteps': model.num_timesteps,
                    'learning_rate': model.learning_rate,
                    'gamma': model.gamma,
                    'tau': model.tau,
                }

                # 添加原始checkpoint信息（如果是继续训练）
                if self.loaded_checkpoint_name:
                    checkpoint['original_checkpoint'] = self.loaded_checkpoint_name
                    checkpoint['is_continued_training'] = True
                else:
                    checkpoint['is_continued_training'] = False

                # 保存策略网络的所有参数
                if hasattr(model.policy, 'actor'):
                    checkpoint['actor_state_dict'] = model.policy.actor.state_dict()

                if hasattr(model.policy, 'critic'):
                    checkpoint['critic_state_dict'] = model.policy.critic.state_dict()

                if hasattr(model.policy, 'critic_target'):
                    checkpoint['critic_target_state_dict'] = model.policy.critic_target.state_dict()

                # 保存优化器状态
                try:
                    optimizers = self.trainer.get_policy_optimizers()
                    if optimizers and len(optimizers) >= 2:
                        checkpoint['actor_optimizer_state_dict'] = optimizers[0].state_dict()
                        checkpoint['critic_optimizer_state_dict'] = optimizers[1].state_dict()
                        if len(optimizers) >= 3:
                            checkpoint['ent_coef_optimizer_state_dict'] = optimizers[2].state_dict()
                except Exception as e:
                    self.logger.warning(f"保存优化器状态失败: {e}")

                # 保存温度参数α（SAC特有）
                if hasattr(model.policy, 'log_ent_coef'):
                    checkpoint['log_ent_coef'] = model.policy.log_ent_coef.data.clone()
                    if hasattr(model.policy, 'ent_coef_optimizer'):
                        checkpoint['ent_coef_optimizer_state_dict'] = model.policy.ent_coef_optimizer.state_dict()
                elif hasattr(model.policy, 'ent_coef'):
                    checkpoint['ent_coef'] = model.policy.ent_coef

                # 保存经验回放缓冲区（如果需要真正的继续训练）
                if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
                    buffer_path = os.path.join(model_dir, f"{self.training_id}_replay_buffer.pkl")
                    import pickle
                    with open(buffer_path, 'wb') as f:
                        pickle.dump(model.replay_buffer, f)
                    self.logger.info(f"经验回放缓冲区已保存: {buffer_path}")
                    checkpoint['has_replay_buffer'] = True
                else:
                    checkpoint['has_replay_buffer'] = False

                # 保存完整的checkpoint
                checkpoint_path = os.path.join(model_dir, f"{self.training_id}_checkpoint.pth")
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"完整checkpoint已保存: {checkpoint_path}")

                # 单独保存各个组件（方便选择性加载）
                if 'actor_state_dict' in checkpoint:
                    actor_path = os.path.join(model_dir, f"{self.training_id}_actor.pth")
                    torch.save(checkpoint['actor_state_dict'], actor_path)
                    self.logger.info(f"Actor网络已保存: {actor_path}")

                if 'critic_state_dict' in checkpoint:
                    critic_path = os.path.join(model_dir, f"{self.training_id}_critic.pth")
                    torch.save(checkpoint['critic_state_dict'], critic_path)
                    self.logger.info(f"Critic网络已保存: {critic_path}")

                if 'critic_target_state_dict' in checkpoint:
                    critic_target_path = os.path.join(model_dir, f"{self.training_id}_critic_target.pth")
                    torch.save(checkpoint['critic_target_state_dict'], critic_target_path)
                    self.logger.info(f"Target Critic网络已保存: {critic_target_path}")

                # 保存模型配置信息
                config_path = os.path.join(model_dir, f"{self.training_id}_model_config.json")
                import json
                model_config = {
                    "observation_space": {
                        "shape": list(self.env.observation_space.shape),
                        "dtype": str(self.env.observation_space.dtype)
                    },
                    "action_space": {
                        "shape": list(self.env.action_space.shape),
                        "dtype": str(self.env.action_space.dtype),
                        "low": self.env.action_space.low.tolist(),
                        "high": self.env.action_space.high.tolist()
                    },
                    "policy_class": str(type(self.trainer.model.policy)),
                    "learning_rate": float(model.learning_rate),
                    "gamma": float(model.gamma),
                    "tau": float(model.tau),
                    "batch_size": getattr(model, 'batch_size', None),
                    "buffer_size": getattr(model, 'buffer_size', None),
                    "learning_starts": getattr(model, 'learning_starts', None),
                    "train_freq": getattr(model, 'train_freq', None),
                    "gradient_steps": getattr(model, 'gradient_steps', None),
                    "training_config": self.current_training_config,
                    "sb3_config": self.config.get("sb3_sac", {}),
                    "environment_config": self.config.get("environment", {}),
                    "from_checkpoint": self.loaded_checkpoint_name is not None,
                    "original_checkpoint": self.loaded_checkpoint_name
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, indent=2, default=str)
                self.logger.info(f"模型配置已保存: {config_path}")

            # 保存奖励图表
            plot_path = os.path.join(self.training_session_dir, "plot", f"{self.training_id}_rewards.png")
            self.trainer.plot_reward(plot_path)
            self.logger.info(f"奖励图表已保存: {plot_path}")
            
            # 保存训练指标图表（actor loss, critic loss, alpha）
            plot_dir = os.path.join(self.training_session_dir, "plot")
            try:
                saved_plots = self.trainer.plot_training_metrics(plot_dir)
                if saved_plots:
                    self.logger.info("训练指标图表已保存:")
                    for metric_name, plot_path in saved_plots.items():
                        self.logger.info(f"  {metric_name}: {plot_path}")
                else:
                    self.logger.info("没有训练指标数据可供绘图")
            except Exception as e:
                self.logger.warning(f"保存训练指标图表失败: {e}")
            
            # 保存动作分布图表
            try:
                action_dist_path = os.path.join(plot_dir, f"{self.training_id}_action_distribution.png")
                self.trainer.plot_action_distribution(action_dist_path)
                self.logger.info(f"动作分布图表已保存: {action_dist_path}")
            except Exception as e:
                self.logger.warning(f"保存动作分布图表失败: {e}")
            
            # 保存动作奖励分布图表
            try:
                action_reward_path = os.path.join(plot_dir, f"{self.training_id}_action_reward_distribution.png")
                self.trainer.plot_action_reward_distribution(action_reward_path)
                self.logger.info(f"动作奖励分布图表已保存: {action_reward_path}")
            except Exception as e:
                self.logger.warning(f"保存动作奖励分布图表失败: {e}")

            # 保存平均元素质量变化图表
            try:
                avg_quality_path = os.path.join(plot_dir, f"{self.training_id}_avg_element_quality.png")
                self.trainer.plot_avg_element_quality(avg_quality_path)
                self.logger.info(f"平均元素质量图表已保存: {avg_quality_path}")
            except Exception as e:
                self.logger.warning(f"保存平均元素质量图表失败: {e}")

            history_path = os.path.join(self.training_session_dir, "history")
            self.trainer.save_history(history_path)
            self.logger.info(f"训练历史已保存: {history_path}")

        except Exception as e:
            self.logger.error(f"保存训练结果失败: {e}")
            self.logger.error(traceback.format_exc())

    def _merge_config(self, frontend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置：前端参数优先，config.yaml作为默认值

        Args:
            frontend_config: 前端传入的配置

        Returns:
            Dict[str, Any]: 合并后的配置
        """
        # 从config.yaml获取默认配置
        training_config = self.config.get("training", {})
        environment_config = self.config.get("environment", {})

        # 构建合并后的配置
        merged_config = {}

        # mesh相关参数（前端必须提供）
        merged_config["mesh_name"] = frontend_config.get("mesh_name")
        merged_config["subfolder"] = frontend_config.get("subfolder", "mesh")

        # checkpoint相关参数（可选） - 修复：确保正确传递
        checkpoint_name = frontend_config.get("checkpoint_name")
        if checkpoint_name and checkpoint_name.strip():
            merged_config["checkpoint_name"] = checkpoint_name.strip()
            self.logger.info(f"配置中包含checkpoint: {checkpoint_name}")
        else:
            merged_config["checkpoint_name"] = None
            self.logger.info("配置中无checkpoint")

        # 训练参数：前端优先，然后使用config.yaml的默认值
        merged_config["max_timesteps"] = (
                frontend_config.get("max_timesteps") or
                training_config.get("max_timesteps", 1000000)
        )

        merged_config["max_steps"] = (
                frontend_config.get("max_steps") or
                environment_config.get("max_steps", 1000)
        )

        # 其他可选参数
        merged_config["description"] = frontend_config.get("description")

        # 添加其他配置项，这些不会被前端修改
        merged_config.update({
            "save_frequency": training_config.get("save_frequency", 10000),
            "log_frequency": training_config.get("log_frequency", 1000),
            "evaluation_frequency": training_config.get("evaluation_frequency", 5000),
            "history_save_frequency": training_config.get("history_save_frequency", 10000)
        })

        log_msg = f"配置合并完成: max_timesteps={merged_config['max_timesteps']}, max_steps={merged_config['max_steps']}"
        if merged_config.get("checkpoint_name"):
            log_msg += f", checkpoint={merged_config['checkpoint_name']}"
        self.logger.info(log_msg)

        return merged_config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证训练配置

        Args:
            config: 训练配置

        Raises:
            ValueError: 当配置无效时
        """
        # 检查必要的参数
        mesh_name = config.get("mesh_name")
        if not mesh_name:
            raise ValueError("mesh_name参数是必需的")

        # 验证mesh文件是否存在
        subfolder = config.get("subfolder", "mesh")
        try:
            mesh_info = self.mesh_importer.get_mesh_info(mesh_name, subfolder)
            if not mesh_info.get("exists", False):
                raise ValueError(f"Mesh文件不存在: {mesh_name}")
        except Exception as e:
            raise ValueError(f"无法验证mesh文件: {str(e)}")

        # 验证checkpoint（如果提供）
        checkpoint_name = config.get("checkpoint_name")
        if checkpoint_name:
            if not self.checkpoint_manager.validate_checkpoint(checkpoint_name):
                raise ValueError(f"Checkpoint无效或不存在: {checkpoint_name}")

        # 验证数值参数
        max_timesteps = config.get("max_timesteps")
        if max_timesteps is not None:
            if not isinstance(max_timesteps, int) or max_timesteps <= 0:
                raise ValueError("max_timesteps必须是正整数")

        max_steps = config.get("max_steps")
        if max_steps is not None:
            if not isinstance(max_steps, int) or max_steps <= 0:
                raise ValueError("max_steps必须是正整数")

    def _create_environment(self, config: Dict[str, Any]) -> None:
        """
        创建训练环境

        Args:
            config: 训练配置
        """
        mesh_name = config["mesh_name"]
        subfolder = config.get("subfolder", "mesh")

        # 加载边界
        boundary = self.mesh_importer.load_boundary_by_name(mesh_name, subfolder)

        # 获取环境参数
        max_steps = config.get("max_steps")

        # 创建环境，传入完整的config以确保所有环境参数都能被读取
        self.env = MeshEnv(
            initial_boundary=boundary,
            max_steps=max_steps,
            config=self.config
        )

        log_msg = f"环境已创建，使用mesh: {mesh_name}, max_steps: {max_steps}"
        if config.get("checkpoint_name"):
            log_msg += f", 将使用checkpoint: {config['checkpoint_name']}"
        self.logger.info(log_msg)

    def _create_trainer(self, config: Dict[str, Any]) -> None:
        """
        创建训练器

        Args:
            config: 训练配置
        """
        if self.env is None:
            raise RuntimeError("环境未创建")

        # 确定设备
        device = "cuda"

        # 获取日志配置
        training_config = self.config.get("training", {})
        enable_verbose_logging = training_config.get("enable_verbose_logging", False)

        # 创建SB3 SAC训练器，传入完整的config以确保所有训练参数都能被读取
        self.trainer = SB3SACTrainer(
            env=self.env,
            device=device,
            config=self.config,
            training_session_dir=self.training_session_dir,
            stop_event=self._stop_event,  # 传递停止事件
            enable_verbose_logging=enable_verbose_logging  # 传递日志配置
        )

        self.logger.info(f"训练器已创建，设备: {device}")

    def _training_loop(self, config: Dict[str, Any]) -> None:
        """
        训练主循环

        Args:
            config: 训练配置
        """
        try:
            max_timesteps = config.get("max_timesteps")

            log_msg = f"开始训练，最大时间步数: {max_timesteps}"
            if config.get("checkpoint_name"):
                log_msg += f"，继续训练自checkpoint: {config['checkpoint_name']}"
            self.logger.info(log_msg)

            # 检查训练开始前是否已经收到停止信号
            if self._stop_event.is_set():
                self.logger.info("训练开始前检测到停止信号，取消训练")
                return

            # 开始训练
            self.trainer.train(total_timesteps=max_timesteps)

            # 检查是否是正常完成还是被停止
            if self._stop_event.is_set():
                self.logger.info("训练被用户停止")
            else:
                self.logger.info("训练正常完成")

        except Exception as e:
            if self._stop_event.is_set():
                self.logger.info("训练在停止过程中发生异常，这可能是正常的")
            else:
                self.logger.error(f"训练过程中发生错误: {e}")
                self.logger.error(traceback.format_exc())
        finally:
            # 训练结束时保存结果
            if not self._stop_event.is_set():
                # 只有在正常结束时才保存结果
                self._save_results()
            else:
                self.logger.info("训练被停止，跳过结果保存")
            
            self._is_running = False
            self._stop_event.set()  # 确保停止事件被设置

    def _update_stats(self) -> None:
        """
        更新训练统计信息
        """
        if not self.trainer:
            return

        try:
            # 获取训练器状态
            trainer_status = self.trainer.get_status()

            # 更新基础统计信息
            self.current_stats.update({
                "episode": trainer_status.get("episodes", 0),
                "total_steps": trainer_status.get("timesteps", 0),
                "episode_reward": trainer_status.get("latest_reward", 0.0) or 0.0,
                "average_reward": trainer_status.get("avg_reward_100", 0.0) or 0.0,
                "episode_length": trainer_status.get("latest_length", 0) or 0,
                "training_id": self.training_id,
                "online_learning_mode": False,  # SB3默认使用经验回放
                # 更新训练指标
                "recent_actor_loss": trainer_status.get("recent_actor_loss", 0.0),
                "recent_critic_loss": trainer_status.get("recent_critic_loss", 0.0),
                "current_alpha": trainer_status.get("current_alpha", 0.0),
                "avg_element_quality": trainer_status.get("avg_element_quality", 0.0) or 0.0,
                # 更新评估信息
                "eval_count": trainer_status.get("eval_count", 0),
                "last_eval_reward": trainer_status.get("last_eval_reward", 0.0),
                "best_eval_reward": trainer_status.get("best_eval_reward", 0.0),
                "evaluation_frequency": trainer_status.get("evaluation_frequency", 10),
                "n_eval_episodes": trainer_status.get("n_eval_episodes", 10),
            })

            mesh_data = trainer_status.get("latest_mesh")
            boundary_vertices = trainer_status.get("latest_boundary")
            ref_info = trainer_status.get("latest_ref_point")

            self.current_stats["boundary_vertices"] = len(boundary_vertices) if boundary_vertices else 0
            self.current_stats["mesh_data"] = mesh_data if mesh_data else {}
            self.current_stats["boundary_vertices_data"] = boundary_vertices if boundary_vertices else []
            self.current_stats["reference_point_info"] = ref_info if ref_info else None
            
            # Update action attempted information for frontend visualization
            latest_action_attempted = trainer_status.get("latest_action_attempted")
            self.current_stats["latest_action_attempted"] = latest_action_attempted if latest_action_attempted else None

            # 获取缓冲区大小
            if hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'replay_buffer'):
                replay_buffer = self.trainer.model.replay_buffer
                if replay_buffer and hasattr(replay_buffer, 'size'):
                    self.current_stats["buffer_size"] = replay_buffer.size()
                else:
                    self.current_stats["buffer_size"] = 0
            else:
                self.current_stats["buffer_size"] = 0

        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取训练管理器健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "status": "healthy",
            "service": "training-api",
            "manager_running": self._is_running,
            "current_training_id": self.training_id,
            "from_checkpoint": self.loaded_checkpoint_name is not None,
            "checkpoint_name": self.loaded_checkpoint_name,
            "timestamp": time.time()
        }

    def cleanup(self) -> None:
        """
        清理资源
        """
        if self._is_running:
            self.stop_training()

        self.trainer = None
        self.env = None
        self.current_training_config = None
        self.training_start_time = None
        self.training_id = None
        self.training_session_dir = None
        self.loaded_checkpoint_name = None
        self.checkpoint_data = None

        self.logger.info("训练管理器资源已清理")


# 全局训练管理器实例
_training_manager = None


def get_training_manager(config=None) -> TrainingManager:
    """
    获取全局训练管理器实例

    Args:
        config: 配置字典

    Returns:
        TrainingManager: 训练管理器实例
    """
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager(config)
    return _training_manager


def reset_training_manager() -> None:
    """
    重置全局训练管理器实例
    """
    global _training_manager
    if _training_manager:
        _training_manager.cleanup()
    _training_manager = None
