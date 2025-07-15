"""
自制SAC训练器实现

基于BaseTrainer的自制SAC训练器，提供完整的训练循环和状态管理
"""
import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Union

from .base_trainer import BaseTrainer
from ..agent.sac_agent import SACAgent
from ..environment import MeshEnv
from ..buffer_factory import create_replay_buffer, get_buffer_info
from src.geometry import Boundary
from src.utils import MeshImporter


class CustomSACTrainer(BaseTrainer):
    """
    自制SAC训练器

    实现了使用自制SAC算法的完整训练流程
    """

    def __init__(self, boundary_source: Union[Boundary, str, Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化自制SAC训练器

        Args:
            boundary_source: 边界数据源
            config: 配置字典
            device: 训练设备
        """
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

        # 初始化经验回放缓冲区
        self._init_replay_buffer()

        # 训练频率配置
        training_config = self.config.get("training", {})
        self.log_frequency = training_config.get("log_frequency", 1000)
        self.save_frequency = training_config.get("save_frequency", 10000)
        self.evaluation_frequency = training_config.get("evaluation_frequency", 5000)
        self.history_save_frequency = training_config.get("history_save_frequency", 10000)

        print("自制SAC训练器初始化完成")

    def _create_boundary_from_source(self, boundary_source: Union[Boundary, str, Dict[str, str], None]) -> Boundary:
        """
        根据源创建边界对象

        Args:
            boundary_source: 边界数据源

        Returns:
            Boundary: 创建的边界对象

        Raises:
            FileNotFoundError: 当指定的文件不存在时
            IOError: 当文件读取失败时
            ValueError: 当数据格式不正确时
        """
        if boundary_source is None:
            print("警告: boundary_source为None，使用默认示例边界（正方形）")
            print("这通常表示前端没有正确传递边界数据源参数")
            print("如果您期望使用特定的边界，请检查前端请求参数")
            default_vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
            return Boundary(default_vertices)
        elif isinstance(boundary_source, Boundary):
            print(f"使用提供的边界对象，包含{len(boundary_source.get_vertices())}个顶点")
            return boundary_source
        elif isinstance(boundary_source, str):
            if not boundary_source.strip():
                raise ValueError("boundary_source字符串不能为空")

            if boundary_source.endswith('.txt'):
                print(f"从文件加载边界: {boundary_source}")
                try:
                    return self.importer.load_boundary_from_file(boundary_source)
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"找不到边界文件: {boundary_source}\n"
                        f"请检查文件路径是否正确。\n"
                        f"原始错误: {e}"
                    )
                except Exception as e:
                    raise IOError(
                        f"加载边界文件失败: {boundary_source}\n"
                        f"原始错误: {e}\n"
                        f"请检查文件格式是否正确。"
                    )
            else:
                print(f"从mesh加载边界: {boundary_source}")
                try:
                    return self.importer.load_boundary_by_name(boundary_source)
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"找不到mesh文件: {boundary_source}\n"
                        f"请检查mesh名称是否正确，或确认文件存在于data/mesh/目录下。\n"
                        f"原始错误: {e}"
                    )
                except Exception as e:
                    raise IOError(
                        f"加载mesh边界失败: {boundary_source}\n"
                        f"原始错误: {e}\n"
                        f"请检查mesh文件格式是否正确。"
                    )
        elif isinstance(boundary_source, dict):
            source_type = boundary_source.get('type')
            if source_type is None:
                raise ValueError(
                    f"字典格式的boundary_source必须包含'type'字段。\n"
                    f"当前字典: {boundary_source}\n"
                    f"支持的类型: 'file' 或 'mesh'"
                )
            elif source_type == 'file':
                path = boundary_source.get('path')
                if path is None:
                    raise ValueError(
                        f"type为'file'时必须提供'path'字段。\n"
                        f"当前字典: {boundary_source}"
                    )
                print(f"从字典指定的文件加载边界: {path}")
                try:
                    return self.importer.load_boundary_from_file(path)
                except Exception as e:
                    raise IOError(
                        f"从字典指定的文件加载边界失败: {path}\n"
                        f"原始错误: {e}"
                    )
            elif source_type == 'mesh':
                name = boundary_source.get('name')
                if name is None:
                    raise ValueError(
                        f"type为'mesh'时必须提供'name'字段。\n"
                        f"当前字典: {boundary_source}"
                    )
                subfolder = boundary_source.get('subfolder', 'mesh')
                print(f"从字典指定的mesh加载边界: {name} (subfolder: {subfolder})")
                try:
                    return self.importer.load_boundary_by_name(name, subfolder)
                except Exception as e:
                    raise IOError(
                        f"从字典指定的mesh加载边界失败: {name}\n"
                        f"subfolder: {subfolder}\n"
                        f"原始错误: {e}"
                    )
            else:
                raise ValueError(
                    f"不支持的边界源类型: {source_type}\n"
                    f"支持的类型: 'file' 或 'mesh'\n"
                    f"当前字典: {boundary_source}"
                )
        else:
            raise ValueError(
                f"不支持的边界源格式: {type(boundary_source)}\n"
                f"传入的值: {boundary_source}\n"
                f"支持的格式:\n"
                f"1. Boundary对象\n"
                f"2. 字符串（文件路径或mesh名称）\n"
                f"3. 字典 {{'type': 'file', 'path': '...'}} 或 {{'type': 'mesh', 'name': '...'}}"
            )

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
        """初始化SAC智能体"""
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            device=self.device,
            config=self.config
        )

    def _init_replay_buffer(self):
        """初始化经验回放缓冲区"""
        self.replay_buffer = create_replay_buffer(config=self.config)
        buffer_info = get_buffer_info(self.replay_buffer)
        self.online_learning_mode = buffer_info.get("mode") == "online_learning"

        print(f"经验回放缓冲区类型: {buffer_info.get('type', 'unknown')}")
        if self.online_learning_mode:
            print("启用在线学习模式")

    def _select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        # 原始的SACAgent.select_action不接受deterministic参数
        # 在deterministic=True时，可以直接使用均值动作，但这需要修改原始Agent
        # 暂时忽略deterministic参数，因为原始agent的select_action已经是确定性的
        return self.agent.select_action(state)

    def _train_step(self, **kwargs) -> Dict[str, Any]:
        """执行一步训练"""
        if self.online_learning_mode:
            # 在线学习模式
            state = kwargs.get('state')
            action = kwargs.get('action')
            reward = kwargs.get('reward')
            next_state = kwargs.get('next_state')
            done = kwargs.get('done')

            loss_info = self.agent.train_online(state, action, reward, next_state, done)
        else:
            # 经验回放模式
            batch_size = kwargs.get('batch_size', 128)
            loss_info = self.agent.train(self.replay_buffer, batch_size)

        return loss_info if loss_info else {}

    def _should_log_progress(self, current_timestep: int, last_log_timestep: int) -> bool:
        """判断是否应该输出训练日志"""
        return current_timestep - last_log_timestep >= self.log_frequency

    def _should_save_history(self, current_timestep: int, last_save_timestep: int) -> bool:
        """判断是否应该保存历史数据"""
        return current_timestep - last_save_timestep >= self.history_save_frequency

    def _should_evaluate(self, current_timestep: int, last_eval_timestep: int) -> bool:
        """判断是否应该进行评估"""
        return current_timestep - last_eval_timestep >= self.evaluation_frequency

    def _log_training_progress(self, episode: int, episode_reward: float):
        """输出训练进度日志"""
        training_id = self.history_manager.get_current_training_id()
        avg_reward = self.training_stats['average_reward']
        total_steps = self.training_stats['total_steps']

        print(f"Timestep {total_steps} Episode {episode} [{training_id}]: "
              f"最新奖励={episode_reward:.3f}, 平均奖励={avg_reward:.3f}")

    def _save_model(self, path: str):
        """保存模型"""
        self.agent.save(path)

    def _load_model(self, path: str):
        """加载模型"""
        self.agent.load(path)

    def train(self, max_timesteps: int = 100000, max_steps_per_episode: int = 1000,
              batch_size: int = 128, start_training_steps: int = 5000,
              **kwargs) -> Dict[str, Any]:
        """
        执行训练主循环

        Args:
            max_timesteps: 最大训练步数
            max_steps_per_episode: 每episode最大步数
            batch_size: 批次大小
            start_training_steps: 开始训练的步数
            **kwargs: 其他训练参数

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        print(f"开始自制SAC训练: 最大timesteps={max_timesteps}")

        start_time = time.time()
        episode = 0

        # 统计变量
        last_save_timestep = 0
        last_log_timestep = 0
        last_eval_timestep = 0

        # 在线学习模式下立即开始训练
        if self.online_learning_mode:
            start_training_steps = 0
            print("使用在线学习模式，立即开始训练")

        # 主训练循环
        while self.training_stats['total_steps'] < max_timesteps:
            if self.stop_event.is_set():
                print("收到停止信号，结束训练")
                break

            episode_reward = 0
            episode_length = 0

            # 重置环境
            state, info = self.env.reset()

            # Episode循环
            for step in range(max_steps_per_episode):
                if self.stop_event.is_set():
                    break

                # 检查是否达到最大timesteps
                if self.training_stats['total_steps'] >= max_timesteps:
                    break

                # 选择动作
                if self.training_stats['total_steps'] < start_training_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self._select_action(state)

                # 执行动作
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                # 存储经验或在线训练
                if self.online_learning_mode:
                    # 在线学习模式：直接训练
                    if self.training_stats['total_steps'] >= start_training_steps:
                        loss_info = self._train_step(
                            state=state, action=action, reward=reward,
                            next_state=next_state, done=done or truncated
                        )

                        # 触发步骤回调
                        step_data = {
                            'timestep': self.training_stats['total_steps'],
                            'loss_info': loss_info
                        }
                        self._trigger_step_callbacks(step_data)
                else:
                    # 经验回放模式：存储经验
                    self.replay_buffer.add(state, action, reward, next_state, done or truncated)

                    # 训练智能体
                    if (len(self.replay_buffer) > batch_size and
                            self.training_stats['total_steps'] >= start_training_steps):
                        loss_info = self._train_step(batch_size=batch_size)

                        # 触发步骤回调
                        step_data = {
                            'timestep': self.training_stats['total_steps'],
                            'loss_info': loss_info
                        }
                        self._trigger_step_callbacks(step_data)

                # 更新状态和统计
                state = next_state
                self.training_stats['total_steps'] += 1

                # 定期日志输出
                if self._should_log_progress(self.training_stats['total_steps'], last_log_timestep):
                    self._log_training_progress(episode, episode_reward)
                    last_log_timestep = self.training_stats['total_steps']

                # 定期保存历史数据
                if self._should_save_history(self.training_stats['total_steps'], last_save_timestep):
                    self.history_manager.force_save_cache()
                    last_save_timestep = self.training_stats['total_steps']

                # Episode结束
                if done or truncated:
                    break

            # Episode结束处理
            self._update_training_stats(episode_reward, episode_length)

            # 获取参考信息
            ref_info = None
            if hasattr(self.env, 'get_last_reference_info'):
                ref_info = self.env.get_last_reference_info()

            # 创建episode数据并触发回调
            episode_data = self._create_episode_data(
                episode=episode,
                episode_reward=episode_reward,
                episode_length=episode_length,
                ref_info=ref_info
            )

            self._trigger_episode_callbacks(episode_data)

            # 缓存到历史管理器
            self.history_manager.cache_episode_data(episode_data)

            episode += 1

        # 训练结束处理
        self.training_stats['training_time'] = time.time() - start_time

        # 强制保存剩余缓存数据
        self.history_manager.force_save_cache()

        # 结束训练会话
        training_stopped_early = self.stop_event.is_set()
        self.history_manager.finish_training_session(
            final_stats=self.training_stats,
            stopped_early=training_stopped_early
        )

        # 保存最终模型
        self._save_final_model()

        mode_str = "在线学习" if self.online_learning_mode else "经验回放"
        if training_stopped_early:
            print(f"自制SAC训练被提前停止! 模式: {mode_str}, "
                  f"总计{self.training_stats['total_steps']}个timesteps, {episode}个episodes")
        else:
            print(f"自制SAC训练完成! 模式: {mode_str}, "
                  f"总计{self.training_stats['total_steps']}个timesteps, {episode}个episodes")

        return self.training_stats

    def _save_final_model(self):
        """保存最终模型到history目录"""
        if hasattr(self.history_manager, 'current_training_dir'):
            model_path = os.path.join(self.history_manager.current_training_dir, "final_model.pth")
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

        print("边界加载完成")