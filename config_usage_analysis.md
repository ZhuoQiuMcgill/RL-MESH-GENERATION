# Config.yaml 配置键使用情况分析报告

本报告分析了 `config/config.yaml` 中每个配置键的使用情况，标识了哪些配置被使用，哪些配置可能未被使用。

## 分析方法
1. 通过 `src/rl/config.py` 中的 `load_config()` 函数加载配置
2. 搜索项目中使用 `load_config` 的文件
3. 分析每个配置键在代码中的使用情况

## 配置使用情况

### 1. paths 配置组 ✅ 已使用

| 配置键 | 状态 | 使用位置 | 说明 |
|--------|------|----------|------|
| `data_root` | ✅ 使用 | `src/utils/importer.py:31` | 数据根目录路径 |
| `mesh_dir` | ✅ 使用 | `src/utils/importer.py:71,82,85` | 网格数据目录 |
| `custom_dir` | ✅ 使用 | `src/utils/importer.py:82,85` | 自定义数据目录 |
| `examples_dir` | ✅ 使用 | `src/utils/importer.py:82,85` | 示例数据目录 |
| `results_dir` | ❓ 未直接使用 | - | 结果目录配置 |
| `logs_dir` | ❓ 未直接使用 | - | 日志目录配置 |
| `models_dir` | ❓ 未直接使用 | - | 模型目录配置 |
| `cache_dir` | ❓ 未直接使用 | - | 缓存目录配置 |
| `temp_dir` | ❓ 未直接使用 | - | 临时目录配置 |
| `tools_dir` | ❓ 未直接使用 | - | 工具目录配置 |
| `config_dir` | ❓ 未直接使用 | - | 配置目录 |
| `default_meshes` | ✅ 使用 | `src/utils/importer.py:365` | 默认网格列表 |

### 2. sb3_sac 配置组 ✅ 广泛使用

| 配置键 | 状态 | 使用位置 | 说明 |
|--------|------|----------|------|
| `learning_rate` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:54,67,84,114` | 学习率参数 |
| `buffer_size` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:55,68,88` | 经验回放缓冲区大小 |
| `learning_starts` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:56,69,85` | 开始训练前的探索步数 |
| `batch_size` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:57,70,86` | 训练批次大小 |
| `tau` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:58,71,89,116` | 软更新系数 |
| `gamma` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:59,72,90,115` | 折扣因子 |
| `train_freq` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:60,73,91` | 训练频率 |
| `gradient_steps` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:61,74,92` | 梯度更新步数 |
| `net_arch` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:47,50,75` | 神经网络架构 |
| `seed` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:63,76,82` | 随机种子 |
| `verbose` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:62,93` (注释掉) | 详细输出级别 |
| `ent_coef` | ✅ 使用 | `src/rl/agent/sb3_sac_agent.py:64,94` (注释掉) | 熵系数 |

### 3. environment 配置组 ✅ 广泛使用

| 配置键 | 状态 | 使用位置 | 说明 |
|--------|------|----------|------|
| `n` | ✅ 使用 | `src/rl/environment.py:22` | 邻居顶点数量 |
| `g` | ✅ 使用 | `src/rl/environment.py:23` | 扇形区域观察点数量 |
| `alpha` | ✅ 使用 | `src/rl/environment.py:24` | 动作空间半径因子 |
| `beta` | ✅ 使用 | `src/rl/environment.py:25` | 状态观察半径因子 |
| `max_steps` | ✅ 使用 | `src/rl/environment.py:26` | 每episode最大步数 |
| `upsilon` | ✅ 使用 | `src/rl/environment.py:27` | 密度奖励参数 |
| `kappa` | ✅ 使用 | `src/rl/environment.py:28` | 密度奖励参数 |
| `M_angle` | ✅ 使用 | `src/rl/environment.py:29,128` | 最大内角阈值 |
| `actions.enabled` | ✅ 使用 | `src/rl/environment.py:32` | 启用的动作类型列表 |
| `actions.auto_remap` | ✅ 使用 | `src/rl/environment.py:33` | 动作类型映射配置 |
| `actions.descriptions` | ❓ 未直接使用 | - | 动作类型描述 (仅用于文档) |

### 4. training 配置组 ✅ 广泛使用

| 配置键 | 状态 | 使用位置 | 说明 |
|--------|------|----------|------|
| `max_timesteps` | ✅ 使用 | `src/ui/training_manager.py:808,941,954` | 最大训练步数 |
| `max_steps_per_episode` | ❓ 未直接使用 | - | 每episode最大步数 (与environment.max_steps重复) |
| `evaluation_frequency` | ✅ 使用 | `src/rl/training/sb3_sac_trainer.py:566` | 评估频率 |
| `n_eval_episodes` | ✅ 使用 | `src/rl/training/sb3_sac_trainer.py:567` | 评估episode数量 |
| `require_completed_for_save` | ✅ 使用 | `src/rl/training/sb3_sac_trainer.py:568` | 保存条件配置 |
| `enable_verbose_logging` | ✅ 使用 | `src/ui/training_manager.py:919` | 详细日志配置 |

## 配置加载分析

### 使用 `load_config()` 的文件：
1. `src/rl/__init__.py` - 导入配置加载函数
2. `src/rl/agent/sb3_sac_agent.py` - SAC智能体配置
3. `src/ui/training_manager.py` - 训练管理器配置
4. `src/utils/importer.py` - 数据导入器路径配置
5. `src/rl/environment.py` - 环境配置
6. `src/rl/training/sb3_sac_trainer.py` - 训练器配置

## 总结

### ✅ 被广泛使用的配置组：
- **sb3_sac**: 所有SAC算法参数都被使用
- **environment**: 所有环境参数都被使用
- **training**: 大部分训练参数都被使用

### ❓ 可能未使用的配置：
- **paths 组中的部分配置**: `results_dir`, `logs_dir`, `models_dir`, `cache_dir`, `temp_dir`, `tools_dir`, `config_dir`
- **actions.descriptions**: 仅用于文档说明
- **training.max_steps_per_episode**: 与 environment.max_steps 功能重复

### 🔄 重复配置：
- `training.max_steps_per_episode` 与 `environment.max_steps` 功能相同

### 建议：
1. 可以考虑删除未使用的 paths 配置项，或在代码中实际使用它们
2. 统一 max_steps 配置，避免重复
3. 对于仅用于文档的配置项，可以考虑移到单独的文档配置文件中

## 配置加载机制

配置通过 `src/rl/config.py` 中的 `load_config()` 函数加载：
- 默认加载 `config/config.yaml` 文件
- 使用缓存机制避免重复加载
- 支持传入自定义路径

所有需要配置的模块都通过调用此函数获取配置，确保了配置的统一性和一致性。
