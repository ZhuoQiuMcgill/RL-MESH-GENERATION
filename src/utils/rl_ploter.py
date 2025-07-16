"""
强化学习可视化模块
"""

import os
import logging
import numpy as np
from typing import List

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 禁用matplotlib的DEBUG日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import matplotlib

# 使用Agg后端避免GUI相关问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_reward_change(episode_rewards: List[float], save_path: str) -> str:
    """
    绘制episode奖励变化图 - 真正的Claude风格设计

    Args:
        episode_rewards: episode奖励列表
        save_path: 保存路径

    Returns:
        str: 保存的文件路径
    """
    if not episode_rewards or len(episode_rewards) < 2:
        raise ValueError("需要至少2个episode的奖励数据")

    episodes = list(range(1, len(episode_rewards) + 1))

    # 真正的Claude风格配色 - 橙色/暖色调
    bg_color = '#1a1a1a'  # 深色背景
    primary_color = '#ff6b35'  # Claude橙色
    secondary_color = '#ffa726'  # 浅橙色
    accent_color = '#ff8f65'  # 强调色
    text_color = '#e8e8e8'  # 浅色文字
    grid_color = '#333333'  # 深色网格
    card_bg = '#2a2a2a'  # 卡片背景

    # 创建图形，使用深色现代样式
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # 绘制主要的奖励曲线 - 使用Claude橙色
    ax.plot(episodes, episode_rewards, color=primary_color, linewidth=2.5,
            alpha=0.9, label='Episode Rewards', zorder=3)

    # 添加移动平均线 - 使用浅橙色
    if len(episode_rewards) > 10:
        window = min(20, len(episode_rewards) // 5)
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], moving_avg, color=secondary_color,
                linewidth=3, alpha=0.95, label=f'Moving Average ({window})', zorder=4)

    # 添加数据点 - 使用强调色
    if len(episodes) <= 100:  # 只在数据点不太多时显示
        ax.scatter(episodes, episode_rewards, color=accent_color, s=30,
                   alpha=0.7, zorder=5, edgecolors=bg_color, linewidth=1)

    # 添加填充区域显示趋势 - 使用渐变橙色
    if len(episode_rewards) > 5:
        ax.fill_between(episodes, episode_rewards, alpha=0.15, color=primary_color, zorder=1)

    # 设置坐标轴样式
    ax.set_xlabel('Episode', fontsize=13, color=text_color, fontweight='500')
    ax.set_ylabel('Reward', fontsize=13, color=text_color, fontweight='500')
    ax.set_title('Training Progress', fontsize=16, color=text_color,
                 fontweight='600', pad=20)

    # 美化网格 - 深色主题
    ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    # 设置坐标轴颜色和样式 - 深色主题
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 设置刻度样式
    ax.tick_params(colors=text_color, labelsize=11)
    ax.tick_params(axis='x', length=6, width=1.5, color=grid_color)
    ax.tick_params(axis='y', length=6, width=1.5, color=grid_color)

    # 添加图例，使用深色主题样式
    legend = ax.legend(loc='upper left', frameon=True, shadow=False,
                       fancybox=True, fontsize=11)
    legend.get_frame().set_facecolor(card_bg)
    legend.get_frame().set_edgecolor(grid_color)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1)
    for text in legend.get_texts():
        text.set_color(text_color)

    # 添加统计信息框 - 深色卡片样式
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    final_reward = episode_rewards[-1]
    mean_reward = np.mean(episode_rewards)

    stats_text = f'Episodes: {len(episode_rewards):,}\nMax: {max_reward:.3f}\nMin: {min_reward:.3f}\nFinal: {final_reward:.3f}\nMean: {mean_reward:.3f}'

    # 创建深色主题的文本框
    props = dict(boxstyle='round,pad=0.6', facecolor=card_bg,
                 edgecolor=primary_color, alpha=0.95, linewidth=1.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, color=text_color,
            bbox=props, fontfamily='monospace')

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存高质量图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor=bg_color, edgecolor='none')
    plt.close()

    return save_path
