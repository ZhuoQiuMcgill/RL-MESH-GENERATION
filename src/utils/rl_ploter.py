import os
import logging
from typing import List, Dict
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 允许重复加载 Intel OpenMP 库，避免 libiomp5md.dll 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 屏蔽 matplotlib.font_manager 的 DEBUG 日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def plot_reward_change(episode_rewards: List[float],
                       episode_lengths: List[int],
                       save_path: str) -> str:
    """
    绘制训练过程中各 episode reward 随累计 timesteps 的变化曲线，
    并把图例和统计信息放到图表外部。

    Args:
        episode_rewards: 各 episode 的 reward 列表，长度为 N。
        episode_lengths: 各 episode 对应的步数列表，长度也应为 N。
        save_path: 图表保存路径。
    Returns:
        str: 最终保存的文件路径。
    """
    # 校验 & 截断
    rewards = np.array(episode_rewards, dtype=float)
    lengths = np.array(episode_lengths, dtype=float)
    if rewards.ndim != 1 or lengths.ndim != 1:
        raise ValueError("episode_rewards 和 episode_lengths 必须都是一维列表")
    if rewards.shape[0] != lengths.shape[0]:
        logger.warning(f"Rewards 长度 {len(rewards)} 与 Lengths 长度 {len(lengths)} 不一致，截断到最小长度")
        m = min(len(rewards), len(lengths))
        rewards = rewards[:m]
        lengths = lengths[:m]

    # 计算横轴（累计 timesteps）
    x_data = np.cumsum(lengths)
    if x_data.shape[0] != rewards.shape[0]:
        raise ValueError("x_data 与 rewards 长度不匹配")

    # 配色
    bg_color = '#1a1a1a'
    primary_color = '#ff6b35'
    secondary_color = '#ffa726'
    accent_color = '#ff8f65'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    card_bg = '#2a2a2a'

    # 创建图和轴，预留右侧空间
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    fig.subplots_adjust(right=0.75)  # 留 25% 空间给外部图例和文本

    # 主曲线（减小线条粗细）
    ax.plot(x_data, rewards,
            color=primary_color, linewidth=1.5,
            alpha=0.9, label='Episode Reward', zorder=3)

    # 移动平均（减小线条粗细）
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(x_data[window - 1:], ma,
                color=secondary_color, linewidth=2,
                alpha=0.95, label=f'Moving Avg ({window})', zorder=4)

    # 100窗口移动平均线（新增）
    if len(rewards) > 100:
        ma_100 = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        ax.plot(x_data[99:], ma_100,
                color='#4fc3f7', linewidth=1.8,
                alpha=0.9, label='Moving Avg (100)', zorder=4)

    # 少量点时加散点
    if len(x_data) <= 100:
        ax.scatter(x_data, rewards,
                   color=accent_color, s=30,
                   alpha=0.7, edgecolors=bg_color,
                   linewidth=1, zorder=5)
    # 填充
    if len(rewards) > 5:
        ax.fill_between(x_data, rewards,
                        alpha=0.15, color=primary_color,
                        zorder=1)

    # 坐标与标题
    ax.set_xlabel('Timesteps', fontsize=13, color=text_color, fontweight='500')
    ax.set_ylabel('Reward', fontsize=13, color=text_color, fontweight='500')
    ax.set_title('Training Progress by Timesteps',
                 fontsize=16, color=text_color,
                 fontweight='600', pad=20)

    # 网格 & 边框
    ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(colors=text_color, labelsize=11)
    ax.tick_params(axis='x', length=6, width=1.5, color=grid_color)
    ax.tick_params(axis='y', length=6, width=1.5, color=grid_color)

    # 图例：放在右侧外部（修复位置对齐）
    legend = ax.legend(loc='upper left',
                       bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0, fancybox=True, fontsize=11)
    legend.get_frame().set_facecolor(card_bg)
    legend.get_frame().set_edgecolor(grid_color)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1)
    for txt in legend.get_texts():
        txt.set_color(text_color)

    # 统计信息：放在右侧外部（修复位置对齐）
    total_ts = int(lengths.sum())
    stats = (
        f"Episodes:     {len(rewards):,}\n"
        f"Timesteps:    {total_ts:,}\n"
        f"Avg Length:   {lengths.mean():.1f}\n"
        f"Max Reward:   {rewards.max():.3f}\n"
        f"Min Reward:   {rewards.min():.3f}\n"
        f"Final Reward: {rewards[-1]:.3f}\n"
        f"Mean Reward:  {rewards.mean():.3f}"
    )
    fig.text(1.02, 0.55, stats,
             fontsize=10, color=text_color,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor=card_bg,
                       edgecolor=primary_color,
                       alpha=0.95,
                       linewidth=1.5),
             transform=ax.transAxes)

    # 保存 & 关闭
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300,
                facecolor=bg_color, edgecolor='none')
    plt.close(fig)

    return save_path


def plot_training_metrics(actor_losses, critic_losses, alphas, timesteps, save_dir: str):
    """
    绘制训练过程中的actor loss, critic loss和alpha变化曲线
    
    Args:
        actor_losses: actor loss数据列表
        critic_losses: critic loss数据列表  
        alphas: alpha数据列表
        timesteps: 对应的时间步数列表
        save_dir: 保存目录路径
        
    Returns:
        dict: 包含生成的图片路径
    """
    import os
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 配色方案
    bg_color = '#1a1a1a'
    primary_color = '#ff6b35'
    secondary_color = '#ffa726'  
    accent_color = '#4fc3f7'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    plt.style.use('default')
    saved_plots = {}
    
    # 绘制Actor Loss
    if actor_losses and len(actor_losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # 确保数据长度一致
        plot_timesteps = timesteps[:len(actor_losses)]
        
        ax.plot(plot_timesteps, actor_losses, 
                color=primary_color, linewidth=1.5, alpha=0.9, 
                label='Actor Loss', zorder=3)
        
        # 添加移动平均
        if len(actor_losses) > 10:
            window = min(50, len(actor_losses) // 5)
            ma = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
            ma_timesteps = plot_timesteps[window-1:len(ma)+window-1]
            ax.plot(ma_timesteps, ma,
                    color=secondary_color, linewidth=2,
                    alpha=0.95, label=f'Moving Avg ({window})', zorder=4)
        
        ax.set_xlabel('Timesteps', fontsize=12, color=text_color, fontweight='500')
        ax.set_ylabel('Actor Loss', fontsize=12, color=text_color, fontweight='500')
        ax.set_title('Actor Loss During Training', 
                     fontsize=14, color=text_color, fontweight='600', pad=15)
        
        # 设置网格和样式
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # 图例
        legend = ax.legend(loc='upper right', fontsize=10)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor(grid_color)
        legend.get_frame().set_alpha(0.95)
        for txt in legend.get_texts():
            txt.set_color(text_color)
        
        plt.tight_layout()
        actor_loss_path = os.path.join(save_dir, 'actor_loss.png')
        plt.savefig(actor_loss_path, dpi=300, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
        saved_plots['actor_loss'] = actor_loss_path
        logger.info(f"Actor loss图表已保存: {actor_loss_path}")
    
    # 绘制Critic Loss
    if critic_losses and len(critic_losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        plot_timesteps = timesteps[:len(critic_losses)]
        
        # 智能Y轴缩放：处理初期极高值问题
        critic_losses_array = np.array(critic_losses)
        
        # 计算百分位数来确定合理的Y轴范围
        q25, q75 = np.percentile(critic_losses_array, [25, 75])
        iqr = q75 - q25
        
        # 如果有超过30%的数据点在后70%的时间范围内，使用后70%数据来设定Y轴
        if len(critic_losses) > 100:
            later_portion = critic_losses_array[int(len(critic_losses) * 0.3):]
            later_q95 = np.percentile(later_portion, 95)
            later_q5 = np.percentile(later_portion, 5)
            
            # 如果初期的极值远大于后期数据，使用后期数据的范围
            if np.max(critic_losses_array[:int(len(critic_losses) * 0.3)]) > later_q95 * 3:
                y_max = later_q95 * 1.1
                y_min = max(0, later_q5 * 0.9)
            else:
                # 使用全部数据的95%分位数
                y_max = np.percentile(critic_losses_array, 95)
                y_min = max(0, np.percentile(critic_losses_array, 5))
        else:
            # 数据量少时使用全部数据
            y_max = np.percentile(critic_losses_array, 95)
            y_min = max(0, np.percentile(critic_losses_array, 5))
        
        ax.plot(plot_timesteps, critic_losses,
                color=accent_color, linewidth=1.5, alpha=0.9,
                label='Critic Loss', zorder=3)
        
        # 添加移动平均
        if len(critic_losses) > 10:
            window = min(50, len(critic_losses) // 5)
            ma = np.convolve(critic_losses, np.ones(window) / window, mode='valid')
            ma_timesteps = plot_timesteps[window-1:len(ma)+window-1]
            ax.plot(ma_timesteps, ma,
                    color=secondary_color, linewidth=2,
                    alpha=0.95, label=f'Moving Avg ({window})', zorder=4)
        
        # 设置智能Y轴范围
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Timesteps', fontsize=12, color=text_color, fontweight='500')
        ax.set_ylabel('Critic Loss', fontsize=12, color=text_color, fontweight='500')
        ax.set_title('Critic Loss During Training',
                     fontsize=14, color=text_color, fontweight='600', pad=15)
        
        # 设置网格和样式
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # 图例
        legend = ax.legend(loc='upper right', fontsize=10)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor(grid_color)
        legend.get_frame().set_alpha(0.95)
        for txt in legend.get_texts():
            txt.set_color(text_color)
        
        plt.tight_layout()
        critic_loss_path = os.path.join(save_dir, 'critic_loss.png')
        plt.savefig(critic_loss_path, dpi=300, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
        saved_plots['critic_loss'] = critic_loss_path
        logger.info(f"Critic loss图表已保存: {critic_loss_path}")
    
    # 绘制Alpha (Entropy Coefficient)
    if alphas and len(alphas) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        plot_timesteps = timesteps[:len(alphas)]
        
        ax.plot(plot_timesteps, alphas,
                color='#ff8f65', linewidth=1.5, alpha=0.9,
                label='Alpha (Entropy Coefficient)', zorder=3)
        
        # 添加移动平均
        if len(alphas) > 10:
            window = min(50, len(alphas) // 5)
            ma = np.convolve(alphas, np.ones(window) / window, mode='valid')
            ma_timesteps = plot_timesteps[window-1:len(ma)+window-1]
            ax.plot(ma_timesteps, ma,
                    color=secondary_color, linewidth=2,
                    alpha=0.95, label=f'Moving Avg ({window})', zorder=4)
        
        ax.set_xlabel('Timesteps', fontsize=12, color=text_color, fontweight='500')
        ax.set_ylabel('Alpha Value', fontsize=12, color=text_color, fontweight='500')
        ax.set_title('Alpha (Entropy Coefficient) During Training',
                     fontsize=14, color=text_color, fontweight='600', pad=15)
        
        # 设置网格和样式
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # 图例
        legend = ax.legend(loc='upper right', fontsize=10)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor(grid_color)  
        legend.get_frame().set_alpha(0.95)
        for txt in legend.get_texts():
            txt.set_color(text_color)
        
        plt.tight_layout()
        alpha_path = os.path.join(save_dir, 'alpha.png')
        plt.savefig(alpha_path, dpi=300, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
        saved_plots['alpha'] = alpha_path
        logger.info(f"Alpha图表已保存: {alpha_path}")
    
    return saved_plots


def plot_action_distribution(action_counts_list: List[Dict], save_path: str) -> str:
    """
    绘制动作分布图，显示每种动作类型的valid/invalid统计
    
    Args:
        action_counts_list: 包含多个episode的action count数据的列表
                           每个元素是形如 {"type1": {"valid": 10, "invalid": 2}, ...} 的字典
        save_path: 图表保存路径
        
    Returns:
        str: 最终保存的文件路径
    """
    # 合并所有episodes的action count数据
    combined_counts = {}
    for episode_counts in action_counts_list:
        if not episode_counts:
            continue
        for action_name, counts in episode_counts.items():
            if action_name not in combined_counts:
                combined_counts[action_name] = {"valid": 0, "invalid": 0}
            combined_counts[action_name]["valid"] += counts.get("valid", 0)
            combined_counts[action_name]["invalid"] += counts.get("invalid", 0)
    
    if not combined_counts:
        logger.warning("没有动作统计数据可用于绘制分布图")
        return save_path
    
    # 配色方案
    bg_color = '#1a1a1a'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    card_bg = '#2a2a2a'
    
    # 准备数据
    action_names = list(combined_counts.keys())
    valid_counts = [combined_counts[name]["valid"] for name in action_names]
    invalid_counts = [combined_counts[name]["invalid"] for name in action_names]
    total_counts = [v + i for v, i in zip(valid_counts, invalid_counts)]
    
    # 创建统一的颜色调色板 - 每个动作类型使用不同的基色
    action_base_colors = {
        'type0_left': '#2196F3',    # 蓝色
        'type0_right': '#4CAF50',   # 绿色
        'type1': '#FF9800',         # 橙色
        'type2': '#9C27B0',         # 紫色
        'type3': '#F44336',         # 红色
        'type4': '#00BCD4',         # 青色
        'type5': '#795548',         # 棕色
        'type6': '#607D8B'          # 蓝灰色
    }
    
    # 为未定义的动作类型生成颜色
    import matplotlib.cm as cm
    base_cmap = cm.get_cmap('tab10')
    for i, name in enumerate(action_names):
        if name not in action_base_colors:
            action_base_colors[name] = base_cmap(i % 10)
    
    # 计算总体统计
    total_actions = sum(total_counts)
    total_valid = sum(valid_counts)
    success_rate = (total_valid / total_actions * 100) if total_actions > 0 else 0
    
    # 创建图表，在标题中包含统计信息
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), facecolor=bg_color)
    main_title = f'Action Distribution Analysis | Total Actions: {total_actions:,} ({success_rate:.1f}% Valid)'
    fig.suptitle(main_title, fontsize=14, color=text_color, fontweight='600', y=0.98)
    
    # 左图：堆叠柱状图显示valid/invalid分布
    ax1.set_facecolor(bg_color)
    x_pos = np.arange(len(action_names))
    
    # 使用浅色和深色版本表示valid/invalid
    def lighten_color(color, amount=0.3):
        """将颜色变浅"""
        import matplotlib.colors as mc
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = np.array(mc.to_rgb(c))
        return tuple(c + (1 - c) * amount)
    
    def darken_color(color, amount=0.3):
        """将颜色变深"""
        import matplotlib.colors as mc
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = np.array(mc.to_rgb(c))
        return tuple(c * (1 - amount))
    
    # 为每个动作类型创建颜色
    valid_colors = [lighten_color(action_base_colors[name], 0.2) for name in action_names]
    invalid_colors = [darken_color(action_base_colors[name], 0.3) for name in action_names]
    
    bars1 = ax1.bar(x_pos, valid_counts, color=valid_colors, alpha=0.9, label='Valid Actions')
    bars2 = ax1.bar(x_pos, invalid_counts, bottom=valid_counts, color=invalid_colors, alpha=0.9, label='Invalid Actions')
    
    # 在柱状图上添加数值标签（仅显示段值，不显示总计）
    for i, (valid, invalid, total) in enumerate(zip(valid_counts, invalid_counts, total_counts)):
        if total > 0:
            # Valid count label
            if valid > 0:
                ax1.text(i, valid/2, str(valid), ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
            # Invalid count label
            if invalid > 0:
                ax1.text(i, valid + invalid/2, str(invalid), ha='center', va='center',
                        color='white', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Action Types', fontsize=12, color=text_color, fontweight='500')
    ax1.set_ylabel('Action Count', fontsize=12, color=text_color, fontweight='500')
    ax1.set_title('Valid vs Invalid Actions by Type', fontsize=12, color=text_color, fontweight='600', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(action_names, rotation=45, ha='right', color=text_color)
    ax1.tick_params(colors=text_color, labelsize=10)
    
    # 设置网格和样式
    ax1.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0, axis='y')
    ax1.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)
    
    # 图例
    legend1 = ax1.legend(loc='upper right', fontsize=10)
    legend1.get_frame().set_facecolor(card_bg)
    legend1.get_frame().set_edgecolor(grid_color)
    legend1.get_frame().set_alpha(0.95)
    for txt in legend1.get_texts():
        txt.set_color(text_color)
    
    # 右图：饼图显示总体动作分布
    ax2.set_facecolor(bg_color)
    
    # 只显示总次数大于0的动作，使用统一的基色
    pie_names = []
    pie_counts = []
    pie_colors = []
    for name, count in zip(action_names, total_counts):
        if count > 0:
            pie_names.append(name)
            pie_counts.append(count)
            pie_colors.append(action_base_colors[name])
    
    if pie_counts:
        wedges, texts, autotexts = ax2.pie(pie_counts, labels=pie_names, colors=pie_colors,
                                          autopct='%1.1f%%', startangle=90, textprops={'color': text_color})
        
        # 设置百分比文字样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    ax2.set_title('Total Action Distribution', fontsize=12, color=text_color, fontweight='600', pad=20)
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Action distribution图表已保存: {save_path}")
    return save_path


def plot_action_reward_distribution(action_counts_list: List[Dict], save_path: str) -> str:
    """
    绘制动作奖励分布图，显示每种动作类型的奖励分布情况
    
    Args:
        action_counts_list: 包含多个episode的action count数据的列表
                           每个元素是形如 {"type1": {"valid": 10, "invalid": 2, "rewards": [...]}, ...} 的字典
        save_path: 图表保存路径
        
    Returns:
        str: 最终保存的文件路径
    """
    # 合并所有episodes的reward数据
    combined_rewards = {}
    for episode_counts in action_counts_list:
        if not episode_counts:
            continue
        for action_name, counts in episode_counts.items():
            if action_name not in combined_rewards:
                combined_rewards[action_name] = []
            # 获取奖励数据，确保是列表
            rewards = counts.get("rewards", [])
            if isinstance(rewards, list):
                combined_rewards[action_name].extend(rewards)
    
    if not combined_rewards or not any(combined_rewards.values()):
        logger.warning("没有奖励数据可用于绘制分布图")
        return save_path
    
    # 配色方案
    bg_color = '#1a1a1a'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    # 动作类型颜色 - 与action distribution图保持一致
    action_colors = {
        'type0_left': '#2196F3',    # 蓝色
        'type0_right': '#4CAF50',   # 绿色
        'type1': '#FF9800',         # 橙色
    }
    
    # 过滤出有数据的动作类型
    valid_actions = {name: rewards for name, rewards in combined_rewards.items() if rewards}
    
    if not valid_actions:
        logger.warning("没有有效的奖励数据")
        return save_path
    
    # 创建图表
    plt.style.use('default')
    num_actions = len(valid_actions)
    
    if num_actions == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7), facecolor=bg_color)
        axes = [axes]
    elif num_actions == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=bg_color)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=bg_color)
    
    # 计算总体统计
    total_rewards = []
    for rewards in valid_actions.values():
        total_rewards.extend(rewards)
    
    overall_mean = np.mean(total_rewards) if total_rewards else 0
    overall_std = np.std(total_rewards) if total_rewards else 0
    
    # 清晰的标题层次结构
    main_title = 'Reward Distribution by Action Type'
    subtitle = f'Total Samples: {len(total_rewards):,} | Overall Mean: {overall_mean:.3f} ± {overall_std:.3f}'
    
    fig.suptitle(main_title, fontsize=16, color=text_color, fontweight='600', y=0.95)
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12, color=text_color, 
             fontweight='400', transform=fig.transFigure)
    
    # 预先计算所有直方图以确定统一的Y轴范围
    all_hist_data = []
    for action_name, rewards in valid_actions.items():
        if rewards:
            n_bins = min(30, max(10, len(rewards) // 10))
            counts, _ = np.histogram(rewards, bins=n_bins)
            all_hist_data.extend(counts)
    
    # 设置统一的Y轴最大值
    max_frequency = max(all_hist_data) if all_hist_data else 100
    y_max = max_frequency * 1.1  # 添加10%的余量
    
    # 为每个动作类型绘制分布图
    for idx, (action_name, rewards) in enumerate(valid_actions.items()):
        ax = axes[idx] if num_actions > 1 else axes[0]
        ax.set_facecolor(bg_color)
        
        if not rewards:
            continue
            
        # 获取颜色
        color = action_colors.get(action_name, '#666666')
        
        # 绘制直方图
        n_bins = min(30, max(10, len(rewards) // 10))  # 自适应bin数量
        counts, bins, patches = ax.hist(rewards, bins=n_bins, color=color, alpha=0.7, 
                                       edgecolor='white', linewidth=0.8)
        
        # 绘制密度曲线
        try:
            from scipy import stats
            if len(rewards) > 1:
                # 计算核密度估计
                kde = stats.gaussian_kde(rewards)
                x_range = np.linspace(min(rewards), max(rewards), 100)
                density = kde(x_range)
                # 缩放密度以匹配直方图
                density_scaled = density * len(rewards) * (bins[1] - bins[0])
                ax.plot(x_range, density_scaled, color=color, linewidth=2, alpha=0.9)
        except ImportError:
            # 如果没有scipy，跳过密度曲线
            pass
        
        # 添加统计线
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        
        # 均值线
        ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                  label=f'Mean: {mean_reward:.3f}')
        # 中位数线  
        ax.axvline(median_reward, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Median: {median_reward:.3f}')
        
        # 设置标题和标签
        std_reward = np.std(rewards)
        ax.set_title(f'{action_name}\n({len(rewards)} samples, σ={std_reward:.3f})', 
                    fontsize=12, color=text_color, fontweight='600', pad=20)
        ax.set_xlabel('Reward Value', fontsize=10, color=text_color)
        ax.set_ylabel('Frequency', fontsize=10, color=text_color)
        
        # 设置统一的Y轴范围
        ax.set_ylim(0, y_max)
        
        # 设置网格和样式
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=9)
        
        # 图例
        legend = ax.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor(grid_color)
        legend.get_frame().set_alpha(0.9)
        for txt in legend.get_texts():
            txt.set_color(text_color)
    
    # 隐藏多余的子图
    if num_actions < len(axes):
        for idx in range(num_actions, len(axes)):
            axes[idx].set_visible(False)
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.subplots_adjust(top=0.80, bottom=0.12, left=0.08, right=0.95, wspace=0.25)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Action reward distribution图表已保存: {save_path}")
    return save_path


def plot_avg_element_quality(avg_qualities: List[float],
                            episode_lengths: List[int],
                            save_path: str) -> str:
    """
    绘制训练过程中平均元素质量随累计时间步的变化曲线
    
    Args:
        avg_qualities: 各episode的平均元素质量列表
        episode_lengths: 各episode对应的步数列表
        save_path: 图表保存路径
        
    Returns:
        str: 最终保存的文件路径
    """
    # 数据校验和预处理
    qualities = np.array(avg_qualities, dtype=float)
    lengths = np.array(episode_lengths, dtype=float)
    
    if qualities.ndim != 1 or lengths.ndim != 1:
        raise ValueError("avg_qualities 和 episode_lengths 必须都是一维列表")
    
    if qualities.shape[0] != lengths.shape[0]:
        logger.warning(f"Qualities 长度 {len(qualities)} 与 Lengths 长度 {len(lengths)} 不一致，截断到最小长度")
        m = min(len(qualities), lengths)
        qualities = qualities[:m]
        lengths = lengths[:m]
    
    if len(qualities) == 0:
        logger.warning("没有平均元素质量数据可用于绘制")
        return save_path
        
    # 计算累计时间步
    cumulative_steps = np.cumsum(lengths)
    
    # 配色方案
    bg_color = '#1a1a1a'
    primary_color = '#4CAF50'  # 绿色主色调
    secondary_color = '#81C784'  # 浅绿色
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # 绘制主曲线
    ax.plot(cumulative_steps, qualities, color=primary_color, linewidth=2.5, 
            alpha=0.9, label='Avg Element Quality')
    
    # 添加数据点
    ax.scatter(cumulative_steps, qualities, color=secondary_color, s=25, 
              alpha=0.7, zorder=5)
    
    # 计算并绘制移动平均线（平滑曲线）
    if len(qualities) >= 10:
        window_size = max(5, len(qualities) // 20)  # 窗口大小为数据点数的5%，最小为5
        moving_avg = np.convolve(qualities, np.ones(window_size)/window_size, mode='valid')
        moving_steps = cumulative_steps[window_size-1:]
        
        ax.plot(moving_steps, moving_avg, color='#FFA726', linewidth=3, 
                alpha=0.8, label=f'Moving Average ({window_size} episodes)')
    
    # 添加统计信息
    mean_quality = np.mean(qualities)
    max_quality = np.max(qualities)
    min_quality = np.min(qualities)
    final_quality = qualities[-1]
    
    # 绘制统计线
    ax.axhline(mean_quality, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean: {mean_quality:.4f}')
    ax.axhline(max_quality, color='orange', linestyle=':', linewidth=2, 
               alpha=0.6, label=f'Max: {max_quality:.4f}')
    
    # 设置标题和标签
    title = f'Average Element Quality Over Training\n'
    title += f'Final: {final_quality:.4f} | Range: [{min_quality:.4f}, {max_quality:.4f}]'
    ax.set_title(title, fontsize=16, color=text_color, fontweight='600', pad=20)
    
    ax.set_xlabel('Cumulative Training Steps', fontsize=12, color=text_color)
    ax.set_ylabel('Average Element Quality', fontsize=12, color=text_color)
    
    # 设置网格和样式
    ax.grid(True, alpha=0.3, color=grid_color, linewidth=1)
    ax.set_axisbelow(True)
    
    # 移除顶部和右侧边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    
    # 设置刻度颜色
    ax.tick_params(colors=text_color, labelsize=10)
    
    # 图例
    legend = ax.legend(loc='best', fontsize=11, framealpha=0.9)
    legend.get_frame().set_facecolor('#2a2a2a')
    legend.get_frame().set_edgecolor(grid_color)
    for txt in legend.get_texts():
        txt.set_color(text_color)
    
    # 添加统计文本框
    stats_text = f'Episodes: {len(qualities)}\n'
    stats_text += f'Total Steps: {int(cumulative_steps[-1])}\n'
    stats_text += f'Std Dev: {np.std(qualities):.4f}\n'
    stats_text += f'Improvement: {final_quality - qualities[0]:.4f}'
    
    # 在右下角添加统计信息
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, color=text_color, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', 
                     edgecolor=grid_color, alpha=0.9))
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Average element quality图表已保存: {save_path}")
    return save_path
