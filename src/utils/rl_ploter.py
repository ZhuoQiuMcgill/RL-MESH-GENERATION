import os
import logging
from typing import List
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
