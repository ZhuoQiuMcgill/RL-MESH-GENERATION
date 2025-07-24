# Training Callback 数据收集分析报告

## 1. 当前 _EpisodeCallback 收集的数据

### 1.1 从环境 info 字典收集的核心数据
```python
detail = info.get("detail", {})
```

#### Episode 级别数据（每个episode结束时收集）：
- **`r`** (reward): 该episode的总奖励值
- **`l`** (length): 该episode的步数长度  
- **`mesh_data`**: 网格邻接关系数据，格式为 `{vertex_str: [adjacent_vertices]}`
- **`boundary_vertices_data`**: 边界顶点坐标列表 `[(x, y), ...]`
- **`last_ref_point`**: 最后一个参考点的信息
- **`is_completed`**: 布尔值，表示episode是否成功完成
- **`episode_number`**: 当前episode编号（callback添加）

### 1.2 Callback 内部维护的统计数据
- **`_current_episode`**: 当前episode计数
- **`_current_timesteps`**: 累积时间步数
- **`_best_reward`**: 最佳奖励值
- **`_best_episode`**: 最佳episode编号
- **`details`**: 完整的episode详情列表
- **`data`**: 按数据类型组织的数据字典

### 1.3 聚合数据结构
```python
self.data = {
    "r": [],                      # 奖励值列表
    "l": [],                      # episode长度列表
    "mesh_data": [],              # 网格数据列表
    "boundary_vertices_data": [], # 边界顶点数据列表
    "last_ref_point": [],         # 参考点信息列表
    "is_completed": []            # 完成状态列表
}
```

## 2. SB3 BaseCallback 继承的数据和方法

### 2.1 BaseCallback 提供的属性
- **`self.model`**: 训练模型实例 (SAC/PPO/DQN等)
- **`self.training_env`**: 训练环境实例
- **`self.logger`**: SB3的日志记录器
- **`self.num_timesteps`**: 当前总时间步数
- **`self.n_calls`**: 回调函数被调用的次数

### 2.2 在 _on_step() 中可访问的 self.locals 变量
- **`rewards`**: 当前步的奖励值列表 `[reward1, reward2, ...]`
- **`dones`**: 环境结束状态列表 `[True/False, ...]`
- **`infos`**: 环境返回的信息字典列表 `[{info1}, {info2}, ...]`
- **`observations`**: 当前观察值
- **`actions`**: 执行的动作
- **`new_obs`**: 执行动作后的新观察值
- **`log_probs`**: 动作的对数概率（对于策略梯度方法）
- **`values`**: 状态值函数输出（对于Actor-Critic方法）
- **`episode_rewards`**: 累积episode奖励
- **`episode_lengths`**: episode长度统计

### 2.3 BaseCallback 的生命周期方法
- **`on_training_start()`**: 训练开始时调用
- **`on_step()`**: 每步后调用（主要数据收集点）
- **`on_rollout_start()`**: 回合开始时调用
- **`on_rollout_end()`**: 回合结束时调用
- **`on_training_end()`**: 训练结束时调用

## 3. 环境 Info 字典的详细内容

### 3.1 每步返回的 info 数据
```python
info = {
    "action_valid": bool,           # 动作是否有效
    "action_name": str,             # 动作类型名称
    "boundary_vertices": int,       # 边界顶点数量
    "element_generated": bool,      # 是否生成了新元素
    "term_reason": str,             # 终止原因
    "trunc_reason": str             # 截断原因
}
```

### 3.2 Episode 结束时的额外 info 数据
```python
info["episode"] = {
    "r": float,                     # episode总奖励
    "l": int                        # episode长度
}

info["detail"] = {
    "r": float,                     # episode总奖励
    "l": int,                       # episode长度
    "mesh_data": dict,              # 网格邻接关系
    "boundary_vertices_data": list, # 边界顶点坐标
    "last_ref_point": dict,         # 最后参考点信息
    "is_completed": bool            # 是否完成
}
```

### 3.3 网格数据 (mesh_data) 结构
```python
# 格式: {vertex_string: [adjacent_vertex_list]}
mesh_data = {
    "(x1,y1)": ["(x2,y2)", "(x3,y3)", ...],
    "(x2,y2)": ["(x1,y1)", "(x4,y4)", ...],
    ...
}
```

### 3.4 参考点信息 (last_ref_point) 结构
包含最后一个参考点的局部环境信息，具体结构需要查看 `get_last_reference_info()` 方法。

## 4. 未被充分利用的 SB3 数据

### 4.1 模型内部状态
- **学习率调度**: `self.model.lr_schedule`
- **策略网络参数**: `self.model.policy.parameters()`
- **优化器状态**: `self.model.policy.optimizer.state_dict()`
- **损失函数值**: 需要从训练日志中提取

### 4.2 环境交互数据
- **动作分布**: `log_probs` 可以分析动作选择的确定性
- **状态值估计**: `values` 可以分析值函数的准确性
- **策略熵**: 可以分析探索程度

### 4.3 训练过程数据
- **梯度信息**: 参数梯度的范数和分布
- **网络激活**: 中间层的激活模式
- **批次统计**: 每个训练批次的统计信息

## 5. 建议的数据收集增强

### 5.1 新增训练过程监控
```python
# 在callback中添加
- model_loss_values: 模型损失值
- policy_entropy: 策略熵值
- value_function_error: 值函数误差
- gradient_norms: 梯度范数
- learning_rates: 当前学习率
```

### 5.2 新增环境交互分析
```python
# 从self.locals获取
- action_distributions: 动作分布统计
- state_values: 状态值估计
- exploration_metrics: 探索度量
- step_rewards: 每步奖励（不仅是episode总奖励）
```

### 5.3 新增性能度量
```python
# 训练效率度量
- training_speed: 每秒处理的步数
- memory_usage: 内存使用情况  
- computation_time: 计算时间分布
- convergence_metrics: 收敛指标
```

## 6. 总结

当前的 `_EpisodeCallback` 主要关注episode级别的结果数据收集，对于深入的训练过程分析和调试还有很大的提升空间。SB3的BaseCallback提供了丰富的训练过程数据接口，可以用来实现更全面的训练监控和分析系统。