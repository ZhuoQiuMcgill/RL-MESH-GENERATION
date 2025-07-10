import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import deque
import json
import threading

from .agent.sac_agent import SACAgent
from .environment import MeshEnv
from .buffer_factory import create_replay_buffer, get_buffer_info
from .config import load_config
from .training_history_manager import TrainingHistoryManager  # 新增导入
from src.geometry import Boundary
from src.utils import MeshImporter


class MeshTrainer:
    """
    网格生成强化学习训练器

    该类封装了整个SAC训练循环，提供训练监控和结果回调功能。
    支持从文件、mesh名称或直接提供Boundary对象来初始化训练环境。
    现在集成了基于timestep间隔的训练历史管理功能，会定期保存episode数据到data/history目录。

    Attributes:
        config: 配置字典
        device: 训练设备(CPU/CUDA)
        env: 训练环境
        eval_env: 评估环境
        agent: SAC智能体
        replay_buffer: 经验回放缓冲区
        training_stats: 训练统计信息
        episode_callbacks: episode完成时的回调函数列表
        importer: 网格数据导入器
        history_manager: 训练历史管理器（使用timestep间隔保存）
    """

    def __init__(self,
                 boundary_source: Union[Boundary, str, Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化训练器

        Args:
            boundary_source: 边界数据源，支持以下格式：
                - Boundary对象：直接使用该边界对象
                - str：文件路径（.txt文件）或mesh名称
                - Dict：包含'type'和相关参数的字典
                  - {'type': 'file', 'path': 'path/to/file.txt'}
                  - {'type': 'mesh', 'name': 'mesh_name', 'subfolder': 'mesh'}
                - None：将使用默认的示例边界
            config: 配置字典，如果为None则从config.yaml加载
            device: 训练设备，如果为None则自动选择

        Raises:
            ValueError: 当boundary_source格式不正确时
            FileNotFoundError: 当指定的文件不存在时

        Note:
            所有模型权重和检查点将保存到data/history/{training_id}/目录下
            Episode历史数据按timestep间隔批量保存，减少I/O操作
        """
        # 加载配置
        self.config = config if config is not None else load_config()

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 初始化网格导入器
        self.importer = MeshImporter(config=self.config)

        # 初始化训练历史管理器（使用timestep间隔保存）
        self.history_manager = TrainingHistoryManager(config=self.config)

        # 验证数据目录结构
        if not self.importer.validate_data_structure():
            print("警告: 数据目录结构验证失败，某些功能可能无法正常工作")

        # 解析并创建边界对象
        self.initial_boundary = self._create_boundary_from_source(boundary_source)

        # 初始化环境
        self._init_environments()

        # 初始化智能体
        self._init_agent()

        # 初始化经验回放缓冲区
        self._init_replay_buffer()

        # 初始化训练统计
        self._init_training_stats()

        # 初始化episode回调系统
        self.episode_callbacks: List[Callable] = []

        # 获取history保存频率
        training_config = self.config.get("training", {})
        self.history_save_frequency = training_config.get("history_save_frequency", 10000)

        print("训练器初始化完成 - Episode历史数据将基于timestep间隔保存到data/history目录")
        print(f"History保存频率: 每{self.history_save_frequency}个timesteps")

    def _create_boundary_from_source(self, boundary_source: Union[Boundary, str, Dict[str, str], None]) -> Boundary:
        """
        根据不同的源类型创建边界对象

        Args:
            boundary_source: 边界数据源

        Returns:
            Boundary: 创建的边界对象

        Raises:
            ValueError: 当源格式不正确时
            FileNotFoundError: 当指定的文件不存在时
        """
        if boundary_source is None:
            # 使用默认的示例边界（简单正方形）
            print("使用默认示例边界（正方形）")
            default_vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
            return Boundary(default_vertices)

        elif isinstance(boundary_source, Boundary):
            # 直接使用提供的边界对象
            print("使用提供的边界对象")
            return boundary_source

        elif isinstance(boundary_source, str):
            # 字符串类型：可能是文件路径或mesh名称
            if boundary_source.endswith('.txt'):
                # 文件路径
                print(f"从文件加载边界: {boundary_source}")
                return self.importer.load_boundary_from_file(boundary_source)
            else:
                # mesh名称
                print(f"从mesh名称加载边界: {boundary_source}")
                return self.importer.load_boundary_by_name(boundary_source)

        elif isinstance(boundary_source, dict):
            # 字典格式：包含类型和参数
            source_type = boundary_source.get('type')

            if source_type == 'file':
                file_path = boundary_source.get('path')
                if not file_path:
                    raise ValueError("字典类型'file'需要提供'path'参数")
                print(f"从文件加载边界: {file_path}")
                return self.importer.load_boundary_from_file(file_path)

            elif source_type == 'mesh':
                mesh_name = boundary_source.get('name')
                subfolder = boundary_source.get('subfolder', 'mesh')
                if not mesh_name:
                    raise ValueError("字典类型'mesh'需要提供'name'参数")
                print(f"从mesh名称加载边界: {mesh_name} (子文件夹: {subfolder})")
                return self.importer.load_boundary_by_name(mesh_name, subfolder)

            else:
                raise ValueError(f"不支持的字典类型: {source_type}. 支持的类型: 'file', 'mesh'")

        else:
            raise ValueError(f"不支持的边界源类型: {type(boundary_source)}. "
                             f"支持的类型: Boundary, str, dict, None")

    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> 'MeshTrainer':
        """
        从文件创建训练器的便捷方法

        Args:
            file_path: txt文件路径
            **kwargs: 其他传递给__init__的参数

        Returns:
            MeshTrainer: 训练器实例

        Example:
            trainer = MeshTrainer.from_file("data/mesh/example.txt")
        """
        return cls(boundary_source=file_path, **kwargs)

    @classmethod
    def from_mesh_name(cls, mesh_name: str, subfolder: str = 'mesh', **kwargs) -> 'MeshTrainer':
        """
        从mesh名称创建训练器的便捷方法

        Args:
            mesh_name: mesh文件名（不含扩展名）
            subfolder: 子文件夹名称
            **kwargs: 其他传递给__init__的参数

        Returns:
            MeshTrainer: 训练器实例

        Example:
            trainer = MeshTrainer.from_mesh_name("1")
            trainer = MeshTrainer.from_mesh_name("complex_shape", "custom")
        """
        boundary_source = {'type': 'mesh', 'name': mesh_name, 'subfolder': subfolder}
        return cls(boundary_source=boundary_source, **kwargs)

    @classmethod
    def from_boundary(cls, boundary: Boundary, **kwargs) -> 'MeshTrainer':
        """
        从边界对象创建训练器的便捷方法

        Args:
            boundary: 边界对象
            **kwargs: 其他传递给__init__的参数

        Returns:
            MeshTrainer: 训练器实例

        Example:
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
            boundary = Boundary(vertices)
            trainer = MeshTrainer.from_boundary(boundary)
        """
        return cls(boundary_source=boundary, **kwargs)

    def list_available_meshes(self, subfolder: str = "mesh") -> List[str]:
        """
        列出可用的网格文件

        Args:
            subfolder: 子文件夹名称，默认为 "mesh"

        Returns:
            List[str]: 可用的网格文件名列表（不含扩展名）
        """
        return self.importer.list_available_meshes(subfolder)

    def get_mesh_info(self, mesh_name: str, subfolder: str = "mesh") -> dict:
        """
        获取网格文件的基本信息

        Args:
            mesh_name: 网格文件名（不含扩展名）
            subfolder: 子文件夹名称，默认为 "mesh"

        Returns:
            dict: 包含网格信息的字典
        """
        return self.importer.get_mesh_info(mesh_name, subfolder)

    def load_new_boundary(self, boundary_source: Union[Boundary, str, Dict[str, str]]) -> None:
        """
        加载新的边界并重新初始化环境

        Args:
            boundary_source: 新的边界数据源

        Note:
            这将重置当前的训练状态和环境，但保留已训练的智能体权重
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

        # 检查状态维度是否发生变化
        if hasattr(self, 'agent') and old_state_dim != 0 and old_state_dim != self.state_dim:
            print(f"状态维度已改变 ({old_state_dim} -> {self.state_dim})，需要重新训练智能体")
            self._init_agent()
        elif hasattr(self, 'agent'):
            print("状态维度未改变，保留已训练的智能体权重")

        # 重置训练统计
        self._init_training_stats()

        print("边界加载完成")

    def _init_environments(self, max_steps=None):
        """初始化训练和评估环境"""
        env_config = self.config.get("environment", {})

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

        # 获取状态和动作空间维度
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"边界顶点数量: {len(self.initial_boundary.get_vertices())}")

    def _init_agent(self):
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
        print(f"经验回放缓冲区类型: {buffer_info['type']}")

        # 检查是否为在线学习模式
        if buffer_info['type'] == 'off':
            print("启用在线学习模式 - 关闭经验回放缓冲区")
            self.online_learning_mode = True
        else:
            print(f"缓冲区容量: {self.replay_buffer.get_capacity()}")
            self.online_learning_mode = False

    def _init_training_stats(self):
        """初始化训练统计信息"""
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': [],
            'evaluation_rewards': [],
            'evaluation_episodes': [],
            'training_time': 0,
            'total_steps': 0,
            'episodes_completed': 0
        }

        # 用于记录最近的奖励（用于early stopping等）
        self.recent_rewards = deque(maxlen=100)

    def add_episode_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        添加episode完成时的回调函数

        Args:
            callback: 回调函数，接收episode数据字典作为参数
        """
        self.episode_callbacks.append(callback)

    def remove_episode_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        移除episode回调函数

        Args:
            callback: 要移除的回调函数
        """
        if callback in self.episode_callbacks:
            self.episode_callbacks.remove(callback)

    def _trigger_episode_callbacks(self, episode_data: Dict[str, Any]):
        """
        触发所有注册的episode回调函数

        Args:
            episode_data: episode完成后的数据
        """
        for callback in self.episode_callbacks:
            try:
                callback(episode_data)
            except Exception as e:
                print(f"回调函数执行错误: {e}")

    def _create_episode_data(self, episode: int, episode_reward: float,
                             episode_length: int, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建episode完成时的数据包

        Args:
            episode: episode编号
            episode_reward: episode总奖励
            episode_length: episode长度
            info: 环境返回的信息

        Returns:
            包含mesh数据和统计信息的字典
        """
        # 获取当前网格数据
        mesh_data = self.env.mesh.get_mesh() if hasattr(self.env, 'mesh') else {}
        boundary_vertices = self.env.boundary.get_vertices() if hasattr(self.env, 'boundary') else []

        # 获取参考点及其局部环境信息
        ref_point_info = None
        if hasattr(self.env, 'get_last_reference_info'):
            ref_point_info = self.env.get_last_reference_info()

        # 计算统计信息
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

        # 获取缓冲区大小（在线学习模式下为0）
        buffer_size = 0 if self.online_learning_mode else len(self.replay_buffer)

        episode_data = {
            'episode': episode,
            'episode_reward': float(episode_reward),
            'episode_length': episode_length,
            'average_reward': float(avg_reward),
            'total_steps': self.training_stats['total_steps'],
            'mesh_data': mesh_data,
            'boundary_vertices': boundary_vertices,
            'boundary_size': len(boundary_vertices),
            'buffer_size': buffer_size,
            'online_learning_mode': self.online_learning_mode,  # 新增字段
            'timestamp': time.time(),
            'episode_info': info,
            'reference_point_info': ref_point_info
        }

        # 添加最近的损失信息（如果有的话）
        if self.training_stats['actor_losses']:
            episode_data['recent_actor_loss'] = float(self.training_stats['actor_losses'][-1])
        if self.training_stats['critic_losses']:
            episode_data['recent_critic_loss'] = float(self.training_stats['critic_losses'][-1])
        if self.training_stats['alpha_values']:
            episode_data['current_alpha'] = float(self.training_stats['alpha_values'][-1])

        return episode_data

    def train(
            self,
            max_timesteps: int = None,
            max_steps: int = None,
            stop_event: Optional["threading.Event"] = None,
            description: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        执行训练过程 - 基于timestep控制，使用timestep间隔保存历史

        Args:
            max_timesteps: 最大训练步数，如果为None则从配置中读取
            max_steps: 每episode最大步数，如果为None则从配置中读取
            stop_event: 可选的threading.Event，用于在外部请求时提前停止训练
            description: 训练描述

        Returns:
            包含训练统计信息的字典
        """
        # 获取训练参数，确保类型正确
        training_config = self.config.get("training", {})

        # 主要参数：max_timesteps
        if max_timesteps is None:
            max_timesteps = int(training_config.get("max_timesteps", 1000000))

        # 环境参数
        if max_steps != self.env.max_steps:
            self._init_environments(max_steps=max_steps)

        # 从配置中读取其他训练参数，确保类型正确（基于timestep）
        save_frequency = int(training_config.get("save_frequency", 10000))
        log_frequency = int(training_config.get("log_frequency", 1000))
        evaluation_frequency = int(training_config.get("evaluation_frequency", 5000))

        # SAC训练参数，确保类型正确
        sac_config = self.config.get("sac_agent", {})
        start_training_steps = int(sac_config.get("start_training_steps", 1000))
        batch_size = int(sac_config.get("batch_size", 256))

        # 开始新的训练会话（历史管理）
        mesh_name = getattr(self, '_current_mesh_name', None)
        config_overrides = {
            'max_timesteps': max_timesteps,
            'max_steps': max_steps,
            'online_learning_mode': self.online_learning_mode,
            'batch_size': batch_size,
            'start_training_steps': start_training_steps,
            'history_save_frequency': self.history_save_frequency
        }

        training_id = self.history_manager.start_training_session(
            mesh_name=mesh_name,
            config_overrides=config_overrides,
            description=description
        )
        print(f"训练会话已开始，ID: {training_id}")
        print(f"训练数据将保存到: {self.history_manager.current_training_dir}")
        print(f"Episode历史将每{self.history_save_frequency}个timesteps批量保存一次")

        # 在线学习模式下，立即开始训练，不需要等待缓冲区填满
        if self.online_learning_mode:
            start_training_steps = 0
            print(f"开始在线学习训练: 最大timesteps={max_timesteps}")
        else:
            print(f"开始训练: 最大timesteps={max_timesteps}")

        start_time = time.time()
        training_stopped_early = False
        episode = 0

        # 初始化统计变量
        last_save_timestep = 0
        last_log_timestep = 0
        last_eval_timestep = 0

        # 主训练循环 - 基于timestep
        while self.training_stats['total_steps'] < max_timesteps:
            if stop_event is not None and stop_event.is_set():
                print("收到停止训练信号，提前结束训练")
                training_stopped_early = True
                break

            episode_reward = 0
            episode_length = 0

            # 重置环境
            state, info = self.env.reset()

            for step in range(max_steps):
                if stop_event is not None and stop_event.is_set():
                    training_stopped_early = True
                    break

                # 检查是否达到最大timesteps
                if self.training_stats['total_steps'] >= max_timesteps:
                    break

                # 选择动作
                if self.training_stats['total_steps'] < start_training_steps:
                    # 前期使用随机动作进行探索（仅在经验回放模式下）
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 根据模式选择不同的训练策略
                if self.online_learning_mode:
                    # 在线学习模式：使用当前转换立即训练
                    if self.training_stats['total_steps'] >= start_training_steps:
                        try:
                            train_info = self.agent.train_online(state, action, reward, next_state, done)

                            # 记录训练损失
                            self.training_stats['actor_losses'].append(train_info['actor_loss'])
                            self.training_stats['critic_losses'].append(train_info['critic_loss'])
                            self.training_stats['alpha_values'].append(train_info['alpha'])

                        except Exception as e:
                            print(f"在线训练时发生错误: {e}")
                else:
                    # 经验回放模式：存储经验到缓冲区
                    self.replay_buffer.add(state, action, reward, next_state, done)

                    # 训练智能体
                    if (len(self.replay_buffer) > batch_size and
                            self.training_stats['total_steps'] >= start_training_steps):
                        try:
                            train_info = self.agent.train(self.replay_buffer, batch_size)

                            # 记录训练损失
                            self.training_stats['actor_losses'].append(train_info['actor_loss'])
                            self.training_stats['critic_losses'].append(train_info['critic_loss'])
                            self.training_stats['alpha_values'].append(train_info['alpha'])

                        except Exception as e:
                            print(f"训练时发生错误: {e}")

                # 更新统计
                episode_reward += reward
                episode_length += 1
                self.training_stats['total_steps'] += 1

                # 基于timestep的日志输出
                if (self.training_stats['total_steps'] - last_log_timestep) >= log_frequency:
                    self._log_training_progress_timestep(episode, episode_reward, episode_length)
                    last_log_timestep = self.training_stats['total_steps']

                # 基于timestep的评估
                if (self.training_stats['total_steps'] - last_eval_timestep) >= evaluation_frequency and \
                        self.training_stats['total_steps'] > 0:
                    eval_reward = self._evaluate_agent()
                    self.training_stats['evaluation_rewards'].append(eval_reward)
                    self.training_stats['evaluation_episodes'].append(episode)
                    print(f"Timestep {self.training_stats['total_steps']}: 评估奖励 = {eval_reward:.3f}")
                    last_eval_timestep = self.training_stats['total_steps']

                # 基于timestep的模型保存
                if (self.training_stats['total_steps'] - last_save_timestep) >= save_frequency and self.training_stats[
                    'total_steps'] > 0:
                    self._save_checkpoint_to_history_timestep(self.training_stats['total_steps'])
                    last_save_timestep = self.training_stats['total_steps']

                state = next_state

                if done or (stop_event is not None and stop_event.is_set()):
                    if stop_event is not None and stop_event.is_set():
                        training_stopped_early = True
                    break

            # 记录episode统计
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['episodes_completed'] += 1
            self.recent_rewards.append(episode_reward)

            # 创建并触发episode回调
            episode_data = self._create_episode_data(episode, episode_reward, episode_length, info)
            self._trigger_episode_callbacks(episode_data)

            # 缓存episode历史数据（使用timestep间隔保存机制）
            self.history_manager.cache_episode_data(episode_data)

            # 如果提前停止，跳出训练循环
            if training_stopped_early:
                break

            episode += 1

        # 训练结束
        self.training_stats['training_time'] = time.time() - start_time

        # 强制保存剩余的缓存数据
        self.history_manager.force_save_cache()

        # 结束训练会话并生成图表（历史管理）
        self.history_manager.finish_training_session(
            final_stats=self.training_stats,
            stopped_early=training_stopped_early
        )

        # 保存最终模型到history目录
        self._save_final_model_to_history()

        if training_stopped_early:
            print("训练被外部停止")
        else:
            mode_str = "在线学习" if self.online_learning_mode else "经验回放"
            print(
                f"训练完成! 模式: {mode_str}, 总计{self.training_stats['total_steps']}个timesteps, {episode}个episodes")

        print(f"训练数据已保存到: {self.history_manager.get_training_plots_path()}")

        # 显示缓存统计信息
        cache_status = self.history_manager.get_cache_status()
        print(f"History保存统计: 最后保存在timestep {cache_status['last_save_timestep']}, "
              f"共保存了{episode}个episodes的历史数据")

        return self.training_stats

    def _save_checkpoint_to_history_timestep(self, timestep: int):
        """保存训练检查点到history目录（基于timestep）"""
        if not hasattr(self.history_manager, 'current_training_dir') or not self.history_manager.current_training_dir:
            return

        checkpoint_dir = self.history_manager.get_training_checkpoints_path()
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_timestep_{timestep}")

        # 保存智能体模型
        self.agent.save(checkpoint_path)

        # 保存训练统计信息
        stats_path = os.path.join(checkpoint_dir, f"stats_timestep_{timestep}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型以便JSON序列化
            serializable_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (np.float32, np.float64)):
                        serializable_stats[key] = [float(v) for v in value]
                    else:
                        serializable_stats[key] = value
                else:
                    serializable_stats[key] = value

            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        print(f"检查点已保存到history (timestep {timestep}): {checkpoint_path}")

    def _save_checkpoint_to_history(self, episode: int):
        """保存训练检查点到history目录（兼容旧版本，基于episode）"""
        if not hasattr(self.history_manager, 'current_training_dir') or not self.history_manager.current_training_dir:
            return

        checkpoint_dir = self.history_manager.get_training_checkpoints_path()
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}")

        # 保存智能体模型
        self.agent.save(checkpoint_path)

        # 保存训练统计信息
        stats_path = os.path.join(checkpoint_dir, f"stats_episode_{episode}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型以便JSON序列化
            serializable_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (np.float32, np.float64)):
                        serializable_stats[key] = [float(v) for v in value]
                    else:
                        serializable_stats[key] = value
                else:
                    serializable_stats[key] = value

            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        print(f"检查点已保存到history: {checkpoint_path}")

    def _save_final_model_to_history(self):
        """保存最终模型到history目录"""
        if not hasattr(self.history_manager, 'current_training_dir') or not self.history_manager.current_training_dir:
            return

        models_dir = self.history_manager.get_training_models_path()
        os.makedirs(models_dir, exist_ok=True)

        # 保存最终模型
        final_model_path = os.path.join(models_dir, "final_model")
        self.agent.save(final_model_path)

        # 保存完整统计信息
        stats_path = os.path.join(models_dir, "final_training_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            serializable_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (np.float32, np.float64)):
                        serializable_stats[key] = [float(v) for v in value]
                    else:
                        serializable_stats[key] = value
                else:
                    serializable_stats[key] = value

            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        print(f"最终模型已保存到history: {final_model_path}")

    def _log_training_progress_timestep(self, episode, episode_reward, episode_length):
        """
        记录训练进度（基于timestep）

        Args:
            episode: 当前episode编号
            episode_reward: episode总奖励
            episode_length: episode长度
        """
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

        # 获取最近的损失信息
        recent_actor_loss = self.training_stats['actor_losses'][-1] if self.training_stats['actor_losses'] else 0
        recent_critic_loss = self.training_stats['critic_losses'][-1] if self.training_stats['critic_losses'] else 0
        current_alpha = self.training_stats['alpha_values'][-1] if self.training_stats['alpha_values'] else 0

        # 根据模式显示不同的信息
        if self.online_learning_mode:
            buffer_info = "在线学习模式"
        else:
            buffer_info = f"缓冲区: {len(self.replay_buffer)}/{self.replay_buffer.get_capacity()}"

        # 显示当前训练ID和缓存状态
        training_id = self.history_manager.get_current_training_id()
        training_info = f"[{training_id}]" if training_id else ""

        # 获取缓存状态
        cache_status = self.history_manager.get_cache_status()
        cache_info = f"缓存: {cache_status['cached_episodes']}个episodes"

        print(
            f"Timestep {self.training_stats['total_steps']} Episode {episode} {training_info}: 奖励={episode_reward:.3f}, 平均奖励={avg_reward:.3f}, "
            f"步数={episode_length}, {buffer_info}, {cache_info}")
        print(f"    Actor损失={recent_actor_loss:.6f}, Critic损失={recent_critic_loss:.6f}, "
              f"Alpha={current_alpha:.6f}")

    def _log_training_progress(self, episode, episode_reward, episode_length):
        """
        记录训练进度（兼容旧版本，基于episode）

        Args:
            episode: 当前episode编号
            episode_reward: episode总奖励
            episode_length: episode长度
        """
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

        # 获取最近的损失信息
        recent_actor_loss = self.training_stats['actor_losses'][-1] if self.training_stats['actor_losses'] else 0
        recent_critic_loss = self.training_stats['critic_losses'][-1] if self.training_stats['critic_losses'] else 0
        current_alpha = self.training_stats['alpha_values'][-1] if self.training_stats['alpha_values'] else 0

        # 根据模式显示不同的信息
        if self.online_learning_mode:
            buffer_info = "在线学习模式"
        else:
            buffer_info = f"缓冲区: {len(self.replay_buffer)}/{self.replay_buffer.get_capacity()}"

        # 显示当前训练ID
        training_id = self.history_manager.get_current_training_id()
        training_info = f"[{training_id}]" if training_id else ""

        print(f"Episode {episode} {training_info}: 奖励={episode_reward:.3f}, 平均奖励={avg_reward:.3f}, "
              f"步数={episode_length}, {buffer_info}")
        print(f"    Actor损失={recent_actor_loss:.6f}, Critic损失={recent_critic_loss:.6f}, "
              f"Alpha={current_alpha:.6f}, 总步数={self.training_stats['total_steps']}")

    def _evaluate_agent(self, num_eval_episodes: int = 5) -> float:
        """
        评估智能体性能

        Args:
            num_eval_episodes: 评估episode数量

        Returns:
            平均评估奖励
        """
        eval_rewards = []

        for _ in range(num_eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                # 评估时使用确定性动作
                action = self.agent.select_action(state)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            eval_rewards.append(episode_reward)

        return np.mean(eval_rewards)

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载训练检查点

        Args:
            checkpoint_path: 检查点路径
        """
        # 加载智能体模型
        self.agent.load(checkpoint_path)

        # TODO: 加载训练统计信息和缓冲区状态
        print(f"检查点已加载: {checkpoint_path}")
        print("注意: 训练统计信息和缓冲区状态需要手动实现加载")

    def test_agent(self,
                   num_test_episodes: int = 10) -> Dict[str, Any]:
        """
        测试训练好的智能体

        Args:
            num_test_episodes: 测试episode数量

        Returns:
            测试结果统计
        """
        print(f"开始测试智能体，共{num_test_episodes}个episodes...")

        test_rewards = []
        test_lengths = []

        for episode in range(num_test_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)

                episode_reward += reward
                episode_length += 1

                state = next_state
                done = terminated or truncated

            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)

            print(f"测试Episode {episode + 1}/{num_test_episodes}: "
                  f"奖励={episode_reward:.3f}, 长度={episode_length}")

        # 计算统计信息
        test_stats = {
            'num_episodes': num_test_episodes,
            'mean_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'mean_length': np.mean(test_lengths),
            'std_length': np.std(test_lengths),
            'min_reward': np.min(test_rewards),
            'max_reward': np.max(test_rewards),
            'rewards': test_rewards,
            'lengths': test_lengths
        }

        print(f"\n测试完成!")
        print(f"平均奖励: {test_stats['mean_reward']:.3f} ± {test_stats['std_reward']:.3f}")
        print(f"平均长度: {test_stats['mean_length']:.1f} ± {test_stats['std_length']:.1f}")

        # 保存测试结果到history目录（如果有活动训练会话）
        if self.history_manager.current_training_dir:
            test_path = os.path.join(self.history_manager.current_training_dir, "test_results.json")
        else:
            # 如果没有活动训练会话，保存到data/history根目录
            test_path = os.path.join(self.history_manager.history_root, "test_results.json")

        with open(test_path, 'w', encoding='utf-8') as f:
            # 处理numpy类型以便JSON序列化
            serializable_stats = {}
            for key, value in test_stats.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (np.float32, np.float64)):
                        serializable_stats[key] = [float(v) for v in value]
                    else:
                        serializable_stats[key] = value
                else:
                    serializable_stats[key] = value

            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        print(f"测试结果已保存: {test_path}")

        return test_stats

    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要信息

        Returns:
            训练摘要字典
        """
        if not self.training_stats['episode_rewards']:
            return {"status": "未开始训练"}

        # 获取缓存状态
        cache_status = self.history_manager.get_cache_status()

        summary = {
            "总训练时间": f"{self.training_stats['training_time']:.2f}秒",
            "完成的episodes": self.training_stats['episodes_completed'],
            "总训练步数": self.training_stats['total_steps'],
            "最终奖励": self.training_stats['episode_rewards'][-1],
            "平均奖励": np.mean(self.training_stats['episode_rewards']),
            "最佳奖励": np.max(self.training_stats['episode_rewards']),
            "缓冲区使用情况": get_buffer_info(self.replay_buffer),
            "边界顶点数量": len(self.initial_boundary.get_vertices()),
            "状态维度": self.state_dim,
            "动作维度": self.action_dim,
            "当前训练ID": self.history_manager.get_current_training_id(),
            "数据保存位置": self.history_manager.current_training_dir,
            "History保存频率": f"每{self.history_save_frequency}个timesteps",
            "缓存状态": cache_status
        }

        if self.training_stats['evaluation_rewards']:
            summary["最佳评估奖励"] = np.max(self.training_stats['evaluation_rewards'])

        return summary

    # 新增方法：历史管理相关
    def get_training_history(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取训练历史信息

        Args:
            training_id: 训练ID，如果为None则返回当前训练信息

        Returns:
            Dict[str, Any]: 训练历史信息
        """
        return self.history_manager.get_training_history(training_id)

    def list_all_training_history(self) -> List[Dict[str, Any]]:
        """
        列出所有历史训练记录

        Returns:
            List[Dict[str, Any]]: 所有训练记录的列表
        """
        return self.history_manager.list_all_trainings()

    def export_training_summary(self, training_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """
        导出指定训练的摘要报告

        Args:
            training_id: 训练ID
            export_path: 导出路径

        Returns:
            Optional[str]: 导出的文件路径，失败时返回None
        """
        return self.history_manager.export_training_summary(training_id, export_path)

    def set_current_mesh_name(self, mesh_name: str):
        """
        设置当前使用的mesh名称（用于历史记录）

        Args:
            mesh_name: mesh名称
        """
        self._current_mesh_name = mesh_name

    def set_history_save_frequency(self, frequency: int):
        """
        设置history保存频率

        Args:
            frequency: 新的保存频率（timesteps）
        """
        self.history_save_frequency = frequency
        self.history_manager.set_save_frequency(frequency)

    def get_history_save_frequency(self) -> int:
        """
        获取当前的history保存频率

        Returns:
            int: 保存频率（timesteps）
        """
        return self.history_save_frequency

    def force_save_history_cache(self) -> bool:
        """
        强制保存当前缓存的历史数据

        Returns:
            bool: 保存是否成功
        """
        return self.history_manager.force_save_cache()
