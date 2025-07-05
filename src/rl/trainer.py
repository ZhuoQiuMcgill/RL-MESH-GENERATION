import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import json

from .agent.sac_agent import SACAgent
from .environment import MeshEnv
from .buffer_factory import create_replay_buffer, get_buffer_info
from .config import load_config
from src.geometry import Boundary


class MeshTrainer:
    """
    网格生成强化学习训练器

    该类封装了整个SAC训练循环，提供训练监控和结果回调功能。
    主要职责是管理训练过程，不涉及复杂的数据预处理和UI交互。

    Attributes:
        config: 配置字典
        device: 训练设备(CPU/CUDA)
        env: 训练环境
        eval_env: 评估环境
        agent: SAC智能体
        replay_buffer: 经验回放缓冲区
        training_stats: 训练统计信息
        episode_callbacks: episode完成时的回调函数列表
    """

    def __init__(self,
                 initial_boundary: Boundary,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化训练器

        Args:
            initial_boundary: 初始边界对象
            config: 配置字典，如果为None则从config.yaml加载
            device: 训练设备，如果为None则自动选择

        Note:
            TODO: 数据导入和边界创建功能将在后续实现
            目前需要外部提供已构建好的Boundary对象
        """
        # 加载配置
        self.config = config if config is not None else load_config()

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 从配置中获取保存目录
        training_config = self.config.get("training", {})
        self.save_dir = training_config.get("save_dir", "results")
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化环境
        self.initial_boundary = initial_boundary
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

    def train(self,
              max_episodes: int = None,
              max_steps: int = None) -> Dict[str, List[float]]:
        """
        执行训练过程

        Args:
            max_episodes: 最大episode数，如果为None则从配置中读取
            max_steps: 最大训练步数，如果为None则从配置中读取

        Returns:
            包含训练统计信息的字典
        """
        # 获取训练参数
        training_config = self.config.get("training", {})
        if max_episodes is None:
            max_episodes = training_config.get("max_episodes", 1000)
        if max_steps is None:
            max_steps = training_config.get("max_steps_per_episode", 1000)

        # 从配置中读取其他训练参数
        save_frequency = training_config.get("save_frequency", 100)
        log_frequency = training_config.get("log_frequency", 10)
        evaluation_frequency = training_config.get("evaluation_frequency", 50)

        # SAC训练参数
        sac_config = self.config.get("sac_agent", {})
        start_training_steps = sac_config.get("start_training_steps", 1000)
        batch_size = sac_config.get("batch_size", 256)

        print(f"开始训练: 最大episodes={max_episodes}, 每episode最大步数={max_steps}")
        start_time = time.time()

        for episode in range(max_episodes):
            episode_reward = 0
            episode_length = 0

            # 重置环境
            state, info = self.env.reset()

            for step in range(max_steps):
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

                if done:
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
        print(f"训练完成! 总耗时: {self.training_stats['training_time']:.2f}秒")

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
        print("TODO: 需要实现完整的检查点恢复功能")

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
            "缓冲区使用情况": get_buffer_info(self.replay_buffer)
        }

        if self.training_stats['evaluation_rewards']:
            summary["最佳评估奖励"] = np.max(self.training_stats['evaluation_rewards'])

        return summary
