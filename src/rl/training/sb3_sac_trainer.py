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
    from stable_baselines3.common.monitor import Monitor
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

    def __init__(self, trainer_instance, verbose: int = 0):
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

    # ──────────────────────────────────────────────────────────────────────────────
    # 生命周期钩子
    # ──────────────────────────────────────────────────────────────────────────────
    def _on_training_start(self) -> None:
        """训练开始时触发"""
        super()._on_training_start()
        total_ts = self.model.num_timesteps + self.model._total_timesteps
        print(f"SB3训练开始，目标timesteps: {total_ts}")

    def _on_step(self) -> bool:
        """
        每个环境步都会进来一次。
        1. 遍历 self.locals['infos'] 找到刚结束的 episode；
        2. 从 training_env 抓取 reference / mesh 相关信息；
        3. 生成 episode_data 并触发回调；
        4. 打日志、响应停止信号。
        """
        # 同步全局步数
        self.trainer.training_stats["total_steps"] = self.num_timesteps

        # ── 1) 处理结束的 episode ─────────────────────────────────────────────
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            if "episode" not in info:
                continue

            self.episode_count += 1
            ep_r = float(info["episode"]["r"])
            ep_l = int(info["episode"]["l"])

            # 取 reference_point_info（兼容 DummyVecEnv / 单环境）
            ref_info = None
            try:
                if hasattr(self.training_env, "get_attr"):
                    funcs = self.training_env.get_attr(
                        "get_last_reference_info", indices=[env_idx]
                    )
                    if funcs and callable(funcs[0]):
                        ref_info = funcs[0]()
                elif hasattr(self.training_env, "get_last_reference_info"):
                    ref_info = self.training_env.get_last_reference_info()
            except Exception as e:
                print(f"获取 reference_point_info 失败: {e}")

            # 更新聚合统计
            self.trainer._update_training_stats(ep_r, ep_l)

            # 生成 episode_data（会自动加 mesh_data）
            episode_data = self.trainer._create_sb3_episode_data(
                episode=self.episode_count,
                episode_reward=ep_r,
                episode_length=ep_l,
                ref_info=ref_info,
            )

            # 回调 & 缓存
            self.trainer._trigger_episode_callbacks(episode_data)
            if hasattr(self.trainer, "history_manager"):
                self.trainer.history_manager.cache_episode_data(episode_data)

        # ── 2) 定期打印进度 ────────────────────────────────────────────────
        if self._should_log_progress(self.num_timesteps):
            self._log_training_progress(self.num_timesteps)
            self.last_log_timestep = self.num_timesteps

        # ── 3) 停止信号 ───────────────────────────────────────────────────
        if self.trainer.stop_event.is_set():
            print("收到停止信号，停止SB3训练")
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────────────
    def _should_log_progress(self, current_ts: int) -> bool:
        """是否到达日志输出间隔"""
        return current_ts - self.last_log_timestep >= self.trainer.log_frequency

    def _log_training_progress(self, current_ts: int) -> None:
        """输出当前训练进度"""
        tid = self.trainer.history_manager.get_current_training_id()
        avg_r = self.trainer.training_stats.get("average_reward", 0.0)
        last_r = self.trainer.training_stats.get("latest_reward", 0.0)
        print(f"SB3 Timestep {current_ts} [{tid}]: Episode {self.episode_count}, "
              f"最新奖励={last_r:.3f}, 平均奖励={avg_r:.3f}")

    def _on_episode_end(self):
        """兼容旧代码：逻辑已移至 _on_step，这里保留空实现避免外部调用失败"""
        pass


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

        # 训练频率配置
        training_config = self.config.get("training", {})
        self.log_frequency = training_config.get("log_frequency", 1000)
        self.save_frequency = training_config.get("save_frequency", 10000)
        self.evaluation_frequency = training_config.get("evaluation_frequency", 5000)

        print("SB3 SAC训练器初始化完成")

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

    def _create_sb3_episode_data(self, episode: int, episode_reward: float,
                                 episode_length: int, ref_info: Optional[Dict] = None) -> Dict[str, Any]:
        """创建SB3特定的episode数据字典"""
        episode_data = {
            'episode': episode,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_steps': self.training_stats['total_steps'],
            'average_reward': self.training_stats['average_reward'],
            'timestamp': time.time(),
            'training_id': self.history_manager.get_current_training_id() if hasattr(self, 'history_manager') else None
        }

        # 添加参考点信息
        if ref_info:
            episode_data['reference_point_info'] = ref_info

        # 添加边界信息
        if hasattr(self, 'initial_boundary'):
            episode_data['boundary_vertices'] = len(self.initial_boundary.get_vertices())
            episode_data['boundary_vertices_data'] = self.initial_boundary.get_vertices()

        # 添加mesh数据（如果可用）
        if hasattr(self, 'env') and hasattr(self.env, 'get_mesh_data'):
            try:
                episode_data['mesh_data'] = self.env.get_mesh_data()
            except Exception as e:
                print(f"获取mesh数据失败: {e}")

        # 添加buffer统计信息
        if hasattr(self, 'agent') and hasattr(self.agent, 'model'):
            try:
                buffer_size = self.agent.model.replay_buffer.size() if hasattr(self.agent.model.replay_buffer,
                                                                               'size') else 0
                episode_data['buffer_size'] = buffer_size
            except:
                episode_data['buffer_size'] = 0

        return episode_data

    from stable_baselines3.common.monitor import Monitor

    def _init_environments(self, max_steps: Optional[int] = None):
        """
        初始化训练与评估环境。
        * 使用 Monitor 包装 MeshEnv，确保每个 episode 结束时 info["episode"]
          自动包含 {"r": episode_return, "l": episode_length}。
        * max_steps 可通过参数显式传入；若为 None，则尝试从配置读取。
        """

        # 若调用方未给 max_steps，则尝试配置文件
        if max_steps is None:
            max_steps = self.config.get("environment", {}).get("max_steps", None)

        def _make_env() -> Monitor:
            return Monitor(
                MeshEnv(
                    initial_boundary=self.initial_boundary,
                    max_steps=max_steps,
                    config=self.config
                )
            )

        # 训练、评估环境各一份
        self.env = _make_env()
        self.eval_env = _make_env()

        # 基本空间信息
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}, "
              f"max_steps: {max_steps}")

    def _initialize_agent(self):
        """初始化SB3 SAC智能体"""
        # 获取SB3配置
        sb3_config = self.config.get("sb3_sac", {})

        # 设置网络架构
        policy_kwargs = dict(
            activation_fn=th.ReLU,
            net_arch=sb3_config.get("net_arch", [128, 128, 128])
        )

        # 创建SAC智能体
        self.agent = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=sb3_config.get("learning_rate", 0.0003),
            buffer_size=sb3_config.get("buffer_size", 1000000),
            learning_starts=sb3_config.get("learning_starts", 10000),
            batch_size=sb3_config.get("batch_size", 100),
            tau=sb3_config.get("tau", 0.005),
            gamma=sb3_config.get("gamma", 0.99),
            train_freq=sb3_config.get("train_freq", 1),
            gradient_steps=sb3_config.get("gradient_steps", 1),
            policy_kwargs=policy_kwargs,
            verbose=sb3_config.get("verbose", 0),
            seed=sb3_config.get("seed", None),
            device=self.device
        )

    def _save_model(self, path: str):
        """保存SB3模型"""
        self.agent.save(path)

    def _load_model(self, path: str):
        """加载SB3模型"""
        self.agent.load(path, env=self.env)

    def train(self, max_timesteps: int = 100000, **kwargs) -> Dict[str, Any]:
        """
        执行SB3训练主循环

        Args:
            max_timesteps: 最大训练步数
            **kwargs: 其他训练参数

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        print(f"开始SB3 SAC训练: 最大timesteps={max_timesteps}")

        start_time = time.time()

        # 创建训练回调
        callback = SB3TrainingCallback(self)

        try:
            # 开始训练
            self.agent.learn(
                total_timesteps=max_timesteps,
                callback=callback,
                progress_bar=False
            )

        except KeyboardInterrupt:
            print("训练被用户中断")
        except Exception as e:
            if self.stop_event.is_set():
                print("训练被停止信号中断")
            else:
                print(f"训练过程中发生错误: {e}")
                raise

        # 训练结束处理
        self.training_stats['training_time'] = time.time() - start_time

        # 强制保存剩余缓存数据
        self.history_manager.force_save_cache()

        # 保存最终模型（在结束训练会话之前）
        self._save_final_model()

        # 结束训练会话
        training_stopped_early = self.stop_event.is_set()
        self.history_manager.finish_training_session(
            final_stats=self.training_stats,
            stopped_early=training_stopped_early
        )

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
