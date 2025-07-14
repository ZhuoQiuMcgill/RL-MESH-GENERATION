"""
SB3 SAC训练器实现

基于BaseTrainer的SB3训练器，集成stable_baselines3的SAC算法
"""
import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from collections import deque

from .base_trainer import BaseTrainer
from ..environment import MeshEnv
from src.geometry import Boundary
from src.utils import MeshImporter

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    import torch.nn as th

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    SAC = None
    BaseCallback = None


class SB3TrainingCallback(BaseCallback):
    """
    SB3训练回调类

    负责收集SB3训练过程中的数据并与trainer交互
    """

    def __init__(self, trainer_instance, verbose=0):
        """
        初始化回调

        Args:
            trainer_instance: SB3SACTrainer实例
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0
        self.last_log_timestep = 0

    def _on_training_start(self) -> None:
        """训练开始时的初始化"""
        super()._on_training_start()
        total_timesteps = self.model.num_timesteps + self.model._total_timesteps
        print(f"SB3训练开始，目标timesteps: {total_timesteps}")

    def _on_step(self) -> bool:
        """每个训练步骤后调用"""
        # 检查是否有episode结束
        dones = self.locals.get('dones', [False])

        if any(dones):
            self._on_episode_end()

        # 定期输出训练日志
        current_timestep = self.model.num_timesteps
        if self._should_log_progress(current_timestep):
            self._log_training_progress(current_timestep)
            self.last_log_timestep = current_timestep

        # 检查停止信号
        if self.trainer.stop_event.is_set():
            print("收到停止信号，停止SB3训练")
            return False

        return True

    def _should_log_progress(self, current_timestep: int) -> bool:
        """判断是否应该输出日志"""
        return current_timestep - self.last_log_timestep >= self.trainer.log_frequency

    def _log_training_progress(self, current_timestep):
        """输出训练进度日志"""
        try:
            # 获取最近的奖励统计
            if self.trainer.recent_rewards:
                avg_reward = np.mean(list(self.trainer.recent_rewards))
                latest_reward = list(self.trainer.recent_rewards)[-1]
            else:
                avg_reward = 0.0
                latest_reward = 0.0

            training_id = self.trainer.history_manager.get_current_training_id()

            print(f"SB3 Timestep {current_timestep} Episode {self.episode_count} [{training_id}]: "
                  f"最新奖励={latest_reward:.3f}, 平均奖励={avg_reward:.3f}")

        except Exception as e:
            print(f"SB3训练日志输出错误: {e}")

    def _on_episode_end(self):
        """Episode结束时的处理"""
        self.episode_count += 1

        # 获取episode统计信息
        episode_reward = 0.0
        episode_length = 0

        try:
            # 尝试从环境中获取episode统计
            env = self.training_env

            # 多种方式尝试获取episode统计
            if hasattr(env, 'episode_reward'):
                episode_reward = float(env.episode_reward)
            elif hasattr(env, 'get_wrapper_attr'):
                try:
                    episode_reward = float(env.get_wrapper_attr('episode_reward'))
                except:
                    pass

            if hasattr(env, 'episode_length'):
                episode_length = int(env.episode_length)
            elif hasattr(env, 'get_wrapper_attr'):
                try:
                    episode_length = int(env.get_wrapper_attr('episode_length'))
                except:
                    pass

        except Exception as e:
            print(f"SB3获取episode统计失败: {e}")
            # 使用默认值
            episode_reward = 0.0
            episode_length = 0

        # 获取参考信息
        ref_info = None
        try:
            env = self.training_env
            if hasattr(env, 'get_last_reference_info'):
                ref_info = env.get_last_reference_info()
        except Exception as e:
            print(f"SB3获取参考信息失败: {e}")

        # 更新trainer统计
        self.trainer._update_training_stats(episode_reward, episode_length)

        # 创建episode数据
        episode_data = self.trainer._create_episode_data(
            episode=self.episode_count,
            episode_reward=episode_reward,
            episode_length=episode_length,
            ref_info=ref_info
        )

        # 触发回调
        self.trainer._trigger_episode_callbacks(episode_data)

        # 缓存到历史管理器
        self.trainer.history_manager.cache_episode_data(episode_data)


class SB3SACTrainer(BaseTrainer):
    """
    SB3 SAC训练器

    使用stable_baselines3的SAC实现进行训练
    """

    def __init__(self, boundary_source: Union[Boundary, str, Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化SB3 SAC训练器

        Args:
            boundary_source: 边界数据源
            config: 配置字典
            device: 训练设备
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 未安装。请运行: pip install stable-baselines3[extra]"
            )

        super().__init__(config)

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 初始化组件
        self.importer = MeshImporter(config=self.config)
        self.initial_boundary = self._create_boundary_from_source(boundary_source)

        # 初始化环境
        self._init_environments()

        # 初始化智能体
        self._initialize_agent()

        print("SB3 SAC训练器初始化完成")

    def _create_boundary_from_source(self, boundary_source: Union[Boundary, str, Dict[str, str], None]) -> Boundary:
        """根据源创建边界对象"""
        if boundary_source is None:
            print("使用默认示例边界（正方形）")
            default_vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
            return Boundary(default_vertices)
        elif isinstance(boundary_source, Boundary):
            print("使用提供的边界对象")
            return boundary_source
        elif isinstance(boundary_source, str):
            if boundary_source.endswith('.txt'):
                print(f"从文件加载边界: {boundary_source}")
                return self.importer.load_boundary_from_file(boundary_source)
            else:
                print(f"从mesh加载边界: {boundary_source}")
                return self.importer.load_boundary_from_mesh(boundary_source)
        elif isinstance(boundary_source, dict):
            source_type = boundary_source.get('type')
            if source_type == 'file':
                path = boundary_source.get('path')
                return self.importer.load_boundary_from_file(path)
            elif source_type == 'mesh':
                name = boundary_source.get('name')
                subfolder = boundary_source.get('subfolder', 'mesh')
                return self.importer.load_boundary_from_mesh(name, subfolder)
            else:
                raise ValueError(f"不支持的边界源类型: {source_type}")
        else:
            raise ValueError(f"不支持的边界源格式: {type(boundary_source)}")

    def _init_environments(self, max_steps=None):
        """初始化训练和评估环境"""
        self.env = MeshEnv(
            initial_boundary=self.initial_boundary,
            max_steps=max_steps,
            config=self.config
        )

        self.eval_env = MeshEnv(
            initial_boundary=self.initial_boundary,
            max_steps=max_steps,
            config=self.config
        )

        # 获取环境信息
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")

    def _initialize_agent(self):
        """初始化SB3 SAC智能体"""
        # 获取SB3配置
        sb3_config = self.config.get("sb3_sac", {})

        # 设置网络架构
        policy_kwargs = dict(
            activation_fn=th.ReLU,
            net_arch=sb3_config.get("net_arch", [128, 128, 128])
        )

        # 创建SAC模型
        self.agent = SAC(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=float(sb3_config.get("learning_rate", 3e-4)),
            buffer_size=int(sb3_config.get("buffer_size", 1000000)),
            learning_starts=int(sb3_config.get("learning_starts", 10000)),
            batch_size=int(sb3_config.get("batch_size", 100)),
            tau=float(sb3_config.get("tau", 0.005)),
            gamma=float(sb3_config.get("gamma", 0.99)),
            train_freq=int(sb3_config.get("train_freq", 1)),
            gradient_steps=int(sb3_config.get("gradient_steps", 1)),
            policy_kwargs=policy_kwargs,
            seed=sb3_config.get("seed", None),
            device=str(self.device),
            verbose=sb3_config.get("verbose", 0)
        )

    def _select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        # 确保state是正确的形状
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

        action, _ = self.agent.predict(state, deterministic=deterministic)
        return action

    def _train_step(self, **kwargs) -> Dict[str, Any]:
        """执行一步训练（SB3内部处理）"""
        # SB3内部处理训练，这里只返回空字典
        return {}

    def _save_model(self, path: str):
        """保存模型"""
        self.agent.save(path)

    def _load_model(self, path: str):
        """加载模型"""
        self.agent = SAC.load(path, env=self.env)

    def train(self, max_timesteps: int = 100000, max_steps: int = 1000,
              batch_size: int = 128, start_training_steps: int = 10000,
              description: str = None, mesh_name: str = None) -> Dict[str, Any]:
        """
        执行训练主循环

        Args:
            max_timesteps: 最大训练步数
            max_steps: 每episode最大步数（SB3中由环境控制）
            batch_size: 批次大小（SB3内部处理）
            start_training_steps: 开始训练的步数（SB3的learning_starts）
            description: 训练描述
            mesh_name: Mesh名称

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        print(f"开始SB3训练: 最大timesteps={max_timesteps}")

        start_time = time.time()

        # 创建回调函数
        callback = SB3TrainingCallback(trainer_instance=self)

        try:
            # 使用SB3进行训练，定期检查停止事件
            training_stopped_early = False
            remaining_timesteps = max_timesteps
            check_interval = 5000  # 每5000步检查一次停止事件

            while remaining_timesteps > 0 and not self.stop_event.is_set():
                # 计算当前批次的训练步数
                current_batch = min(check_interval, remaining_timesteps)

                # 执行训练
                self.agent.learn(
                    total_timesteps=current_batch,
                    callback=callback,
                    reset_num_timesteps=False
                )

                remaining_timesteps -= current_batch

                # 检查停止事件
                if self.stop_event.is_set():
                    print("收到停止信号，停止SB3训练")
                    training_stopped_early = True
                    break

            if remaining_timesteps <= 0:
                print("SB3训练完成所有timesteps")

        except KeyboardInterrupt:
            print("SB3训练被用户中断")
            training_stopped_early = True
        except Exception as e:
            print(f"SB3训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            training_stopped_early = True

        # 更新统计信息
        self.training_stats['training_time'] = time.time() - start_time
        self.training_stats['total_steps'] = self.agent.num_timesteps
        self.training_stats['episodes_completed'] = callback.episode_count

        # 强制保存剩余缓存数据
        self.history_manager.force_save_cache()

        # 结束训练会话
        self.history_manager.finish_training_session(
            final_stats=self.training_stats,
            stopped_early=training_stopped_early
        )

        # 保存最终模型
        self._save_final_model()

        if training_stopped_early:
            print(f"SB3训练被提前停止! 总计{self.training_stats['total_steps']}个timesteps, "
                  f"{callback.episode_count}个episodes")
        else:
            print(f"SB3训练完成! 总计{self.training_stats['total_steps']}个timesteps, "
                  f"{callback.episode_count}个episodes")

        return self.training_stats

    def _save_final_model(self):
        """保存最终模型到history目录"""
        if hasattr(self.history_manager, 'current_training_dir'):
            model_path = os.path.join(self.history_manager.current_training_dir, "final_model.zip")
            self._save_model(model_path)
            print(f"最终模型已保存到: {model_path}")

    def load_boundary(self, boundary_source: Union[Boundary, str, Dict[str, str]]):
        """
        加载新边界并重新初始化环境

        Args:
            boundary_source: 边界数据源
        """
        print("加载新边界并重新初始化环境...")

        # 创建新边界
        old_boundary_size = len(self.initial_boundary.get_vertices()) if hasattr(self, 'initial_boundary') else 0
        self.initial_boundary = self._create_boundary_from_source(boundary_source)
        new_boundary_size = len(self.initial_boundary.get_vertices())

        print(f"边界顶点数量: {old_boundary_size} -> {new_boundary_size}")

        # 重新初始化环境
        old_state_dim = self.state_dim if hasattr(self, 'state_dim') else 0
        self._init_environments()

        # 检查状态维度是否变化
        if hasattr(self, 'agent') and old_state_dim != 0 and old_state_dim != self.state_dim:
            print(f"状态维度已改变 ({old_state_dim} -> {self.state_dim})，需要重新训练智能体")
            self._initialize_agent()
        elif hasattr(self, 'agent'):
            print("状态维度未改变，保留已训练的智能体权重")
            # 更新agent的环境
            self.agent.set_env(self.env)

        print("边界加载完成")
