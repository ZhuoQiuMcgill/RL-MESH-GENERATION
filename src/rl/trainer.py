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
from src.geometry import Boundary
from src.utils import MeshImporter


class MeshTrainer:
    """
    网格生成强化学习训练器

    该类封装了整个SAC训练循环，提供训练监控和结果回调功能。
    支持从文件、mesh名称或直接提供Boundary对象来初始化训练环境。

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
            所有路径配置都从config.yaml的paths部分读取
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

        # 验证数据目录结构
        if not self.importer.validate_data_structure():
            print("警告: 数据目录结构验证失败，某些功能可能无法正常工作")

        # 从配置中获取保存目录
        training_config = self.config.get("training", {})
        paths_config = self.config.get("paths", {})

        # 优先使用training配置中的save_dir，然后是paths配置中的results_dir，最后是默认值
        save_dir_name = training_config.get("save_dir") or paths_config.get("results_dir", "results")

        # 转换为绝对路径
        if os.path.isabs(save_dir_name):
            self.save_dir = save_dir_name
        else:
            self.save_dir = os.path.join(os.getcwd(), save_dir_name)

        os.makedirs(self.save_dir, exist_ok=True)

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

        print("训练器初始化完成")

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

    def _init_environments(self):
        """初始化训练和评估环境"""
        env_config = self.config.get("environment", {})

        # 训练环境
        self.env = MeshEnv(
            initial_boundary=self.initial_boundary,
            config=self.config
        )

        # 评估环境（使用相同的边界）
        self.eval_env = MeshEnv(
            initial_boundary=self.initial_boundary,
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
        print(f"缓冲区容量: {self.replay_buffer.get_capacity()}")

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

        # 计算统计信息
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

        episode_data = {
            'episode': episode,
            'episode_reward': float(episode_reward),
            'episode_length': episode_length,
            'average_reward': float(avg_reward),
            'total_steps': self.training_stats['total_steps'],
            'mesh_data': mesh_data,
            'boundary_vertices': boundary_vertices,
            'boundary_size': len(boundary_vertices),
            'buffer_size': len(self.replay_buffer),
            'timestamp': time.time(),
            'episode_info': info
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
            max_episodes: int = None,
            max_steps: int = None,
            stop_event: Optional["threading.Event"] = None,
    ) -> Dict[str, List[float]]:
        """
        执行训练过程

        Args:
            max_episodes: 最大episode数，如果为None则从配置中读取
            max_steps: 最大训练步数，如果为None则从配置中读取
            stop_event: 可选的threading.Event，用于在外部请求时提前停止训练

        Returns:
            包含训练统计信息的字典
        """
        # 获取训练参数，确保类型正确
        training_config = self.config.get("training", {})
        if max_episodes is None:
            max_episodes = int(training_config.get("max_episodes", 1000))
        if max_steps is None:
            max_steps = int(training_config.get("max_steps_per_episode", 1000))

        # 从配置中读取其他训练参数，确保类型正确
        save_frequency = int(training_config.get("save_frequency", 100))
        log_frequency = int(training_config.get("log_frequency", 10))
        evaluation_frequency = int(training_config.get("evaluation_frequency", 50))

        # SAC训练参数，确保类型正确
        sac_config = self.config.get("sac_agent", {})
        start_training_steps = int(sac_config.get("start_training_steps", 1000))
        batch_size = int(sac_config.get("batch_size", 256))

        print(f"开始训练: 最大episodes={max_episodes}, 每episode最大步数={max_steps}")
        start_time = time.time()

        for episode in range(max_episodes):
            if stop_event is not None and stop_event.is_set():
                print("收到停止训练信号，提前结束训练")
                break
            episode_reward = 0
            episode_length = 0

            # 重置环境
            state, info = self.env.reset()

            for step in range(max_steps):
                if stop_event is not None and stop_event.is_set():
                    break
                # 选择动作
                if self.training_stats['total_steps'] < start_training_steps:
                    # 前期使用随机动作进行探索
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 存储经验
                self.replay_buffer.add(state, action, reward, next_state, done)

                # 更新统计
                episode_reward += reward
                episode_length += 1
                self.training_stats['total_steps'] += 1

                # 训练智能体
                if (len(self.replay_buffer) > batch_size and
                        self.training_stats['total_steps'] >= start_training_steps):
                    train_info = self.agent.train(self.replay_buffer, batch_size)

                    # 记录训练损失
                    self.training_stats['actor_losses'].append(train_info['actor_loss'])
                    self.training_stats['critic_losses'].append(train_info['critic_loss'])
                    self.training_stats['alpha_values'].append(train_info['alpha'])

                state = next_state

                if done or (stop_event is not None and stop_event.is_set()):
                    break

            # 记录episode统计
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['episodes_completed'] += 1
            self.recent_rewards.append(episode_reward)

            # 创建并触发episode回调
            episode_data = self._create_episode_data(episode, episode_reward, episode_length, info)
            self._trigger_episode_callbacks(episode_data)

            # 日志输出
            if episode % log_frequency == 0:
                self._log_training_progress(episode, episode_reward, episode_length)

            # 评估模型
            if episode % evaluation_frequency == 0 and episode > 0:
                eval_reward = self._evaluate_agent()
                self.training_stats['evaluation_rewards'].append(eval_reward)
                self.training_stats['evaluation_episodes'].append(episode)
                print(f"Episode {episode}: 评估奖励 = {eval_reward:.3f}")

            # 保存模型
            if episode % save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)

        # 训练结束
        self.training_stats['training_time'] = time.time() - start_time
        if stop_event is not None and stop_event.is_set():
            print("训练被外部停止")
        else:
            print(
                f"训练完成! 总耗时: {self.training_stats['training_time']:.2f}秒"
            )

        # 保存最终模型和统计信息
        self._save_final_results()

        return self.training_stats

    def _log_training_progress(self, episode: int, reward: float, length: int):
        """记录训练进度"""
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        buffer_size = len(self.replay_buffer)

        print(f"Episode {episode:4d} | "
              f"奖励: {reward:8.3f} | "
              f"长度: {length:3d} | "
              f"平均奖励: {avg_reward:8.3f} | "
              f"缓冲区: {buffer_size:6d} | "
              f"总步数: {self.training_stats['total_steps']:6d}")

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

    def _save_checkpoint(self, episode: int):
        """保存训练检查点"""
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
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

        print(f"检查点已保存: {checkpoint_path}")

    def _save_final_results(self):
        """保存最终训练结果"""
        # 保存最终模型
        final_model_path = os.path.join(self.save_dir, "final_model")
        self.agent.save(final_model_path)

        # 保存完整统计信息
        stats_path = os.path.join(self.save_dir, "training_stats.json")
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

        print(f"最终结果已保存到: {self.save_dir}")

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

        # 保存测试结果
        test_path = os.path.join(self.save_dir, "test_results.json")
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
            "动作维度": self.action_dim
        }

        if self.training_stats['evaluation_rewards']:
            summary["最佳评估奖励"] = np.max(self.training_stats['evaluation_rewards'])

        return summary

