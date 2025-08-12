import os
import logging
from typing import List, Dict
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow reloading Intel OpenMP library to avoid libiomp5md.dll conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Suppress matplotlib.font_manager DEBUG logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def plot_reward_change(episode_rewards: List[float],
                       episode_lengths: List[int],
                       save_path: str) -> str:
    """
    Plot the curve of episode rewards vs cumulative timesteps during training,
    with legends and statistics placed outside the chart.

    Args:
        episode_rewards: List of rewards for each episode, length N.
        episode_lengths: List of steps for each episode, length should also be N.
        save_path: Chart save path.
    Returns:
        str: Final saved file path.
    """
    # Validation & truncation
    rewards = np.array(episode_rewards, dtype=float)
    lengths = np.array(episode_lengths, dtype=float)
    if rewards.ndim != 1 or lengths.ndim != 1:
        raise ValueError("episode_rewards and episode_lengths must both be one-dimensional lists")
    if rewards.shape[0] != lengths.shape[0]:
        logger.warning(f"Rewards length {len(rewards)} and Lengths length {len(lengths)} do not match, truncating to minimum length")
        m = min(len(rewards), len(lengths))
        rewards = rewards[:m]
        lengths = lengths[:m]

    # Calculate x-axis (cumulative timesteps)
    x_data = np.cumsum(lengths)
    if x_data.shape[0] != rewards.shape[0]:
        raise ValueError("x_data and rewards length mismatch")

    # Color scheme
    bg_color = '#1a1a1a'
    primary_color = '#ff6b35'
    secondary_color = '#ffa726'
    accent_color = '#ff8f65'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    card_bg = '#2a2a2a'

    # Create figure and axes, reserve right side space
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    fig.subplots_adjust(right=0.75)  # Reserve 25% space for external legends and text

    # Main curve (reduce line thickness)
    ax.plot(x_data, rewards,
            color=primary_color, linewidth=1.5,
            alpha=0.9, label='Episode Reward', zorder=3)

    # Moving average (reduce line thickness)
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(x_data[window - 1:], ma,
                color=secondary_color, linewidth=2,
                alpha=0.95, label=f'Moving Avg ({window})', zorder=4)

    # 100-window moving average line (newly added)
    if len(rewards) > 100:
        ma_100 = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        ax.plot(x_data[99:], ma_100,
                color='#4fc3f7', linewidth=1.8,
                alpha=0.9, label='Moving Avg (100)', zorder=4)

    # Add scatter points when few data points
    if len(x_data) <= 100:
        ax.scatter(x_data, rewards,
                   color=accent_color, s=30,
                   alpha=0.7, edgecolors=bg_color,
                   linewidth=1, zorder=5)
    # Fill area
    if len(rewards) > 5:
        ax.fill_between(x_data, rewards,
                        alpha=0.15, color=primary_color,
                        zorder=1)

    # Axes and title
    ax.set_xlabel('Timesteps', fontsize=13, color=text_color, fontweight='500')
    ax.set_ylabel('Reward', fontsize=13, color=text_color, fontweight='500')
    ax.set_title('Training Progress by Timesteps',
                 fontsize=16, color=text_color,
                 fontweight='600', pad=20)

    # Grid & borders
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

    # Legend: place on the right side externally (fix position alignment)
    legend = ax.legend(loc='upper left',
                       bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0, fancybox=True, fontsize=11)
    legend.get_frame().set_facecolor(card_bg)
    legend.get_frame().set_edgecolor(grid_color)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1)
    for txt in legend.get_texts():
        txt.set_color(text_color)

    # Statistics: place on the right side externally (fix position alignment)
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

    # Save & close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300,
                facecolor=bg_color, edgecolor='none')
    plt.close(fig)

    return save_path


def plot_training_metrics(actor_losses, critic_losses, alphas, timesteps, save_dir: str):
    """
    Plot training curves for actor loss, critic loss, and alpha changes
    
    Args:
        actor_losses: List of actor loss data
        critic_losses: List of critic loss data  
        alphas: List of alpha data
        timesteps: Corresponding timestep list
        save_dir: Save directory path
        
    Returns:
        dict: Dictionary containing generated plot paths
    """
    import os
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme
    bg_color = '#1a1a1a'
    primary_color = '#ff6b35'
    secondary_color = '#ffa726'  
    accent_color = '#4fc3f7'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    plt.style.use('default')
    saved_plots = {}
    
    # Plot Actor Loss
    if actor_losses and len(actor_losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Ensure data length consistency
        plot_timesteps = timesteps[:len(actor_losses)]
        
        ax.plot(plot_timesteps, actor_losses, 
                color=primary_color, linewidth=1.5, alpha=0.9, 
                label='Actor Loss', zorder=3)
        
        # Add moving average
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
        
        # Set grid and styling
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # Legend
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
        logger.info(f"Actor loss chart saved: {actor_loss_path}")
    
    # Plot Critic Loss
    if critic_losses and len(critic_losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        plot_timesteps = timesteps[:len(critic_losses)]
        
        # Smart Y-axis scaling: handle initial extreme values
        critic_losses_array = np.array(critic_losses)
        
        # Calculate percentiles to determine reasonable Y-axis range
        q25, q75 = np.percentile(critic_losses_array, [25, 75])
        iqr = q75 - q25
        
        # If more than 30% of data points are in the later 70% time range, use later 70% data to set Y-axis
        if len(critic_losses) > 100:
            later_portion = critic_losses_array[int(len(critic_losses) * 0.3):]
            later_q95 = np.percentile(later_portion, 95)
            later_q5 = np.percentile(later_portion, 5)
            
            # If initial extreme values are much larger than later data, use later data range
            if np.max(critic_losses_array[:int(len(critic_losses) * 0.3)]) > later_q95 * 3:
                y_max = later_q95 * 1.1
                y_min = max(0, later_q5 * 0.9)
            else:
                # Use 95th percentile of all data
                y_max = np.percentile(critic_losses_array, 95)
                y_min = max(0, np.percentile(critic_losses_array, 5))
        else:
            # Use all data when data volume is small
            y_max = np.percentile(critic_losses_array, 95)
            y_min = max(0, np.percentile(critic_losses_array, 5))
        
        ax.plot(plot_timesteps, critic_losses,
                color=accent_color, linewidth=1.5, alpha=0.9,
                label='Critic Loss', zorder=3)
        
        # Add moving average
        if len(critic_losses) > 10:
            window = min(50, len(critic_losses) // 5)
            ma = np.convolve(critic_losses, np.ones(window) / window, mode='valid')
            ma_timesteps = plot_timesteps[window-1:len(ma)+window-1]
            ax.plot(ma_timesteps, ma,
                    color=secondary_color, linewidth=2,
                    alpha=0.95, label=f'Moving Avg ({window})', zorder=4)
        
        # Set smart Y-axis range
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Timesteps', fontsize=12, color=text_color, fontweight='500')
        ax.set_ylabel('Critic Loss', fontsize=12, color=text_color, fontweight='500')
        ax.set_title('Critic Loss During Training',
                     fontsize=14, color=text_color, fontweight='600', pad=15)
        
        # Set grid and styling
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # Legend
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
        logger.info(f"Critic loss chart saved: {critic_loss_path}")
    
    # Plot Alpha (Entropy Coefficient)
    if alphas and len(alphas) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        plot_timesteps = timesteps[:len(alphas)]
        
        ax.plot(plot_timesteps, alphas,
                color='#ff8f65', linewidth=1.5, alpha=0.9,
                label='Alpha (Entropy Coefficient)', zorder=3)
        
        # Add moving average
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
        
        # Set grid and styling
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # Legend
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
        logger.info(f"Alpha chart saved: {alpha_path}")
    
    return saved_plots


def plot_action_distribution(action_counts_list: List[Dict], save_path: str) -> str:
    """
    Plot action distribution chart showing valid/invalid statistics for each action type
    
    Args:
        action_counts_list: List containing action count data from multiple episodes
                           Each element is a dict like {"type1": {"valid": 10, "invalid": 2}, ...}
        save_path: Chart save path
        
    Returns:
        str: Final saved file path
    """
    # Merge action count data from all episodes
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
        logger.warning("No action statistics data available for plotting distribution chart")
        return save_path
    
    # Color scheme
    bg_color = '#1a1a1a'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    card_bg = '#2a2a2a'
    
    # Prepare data
    action_names = list(combined_counts.keys())
    valid_counts = [combined_counts[name]["valid"] for name in action_names]
    invalid_counts = [combined_counts[name]["invalid"] for name in action_names]
    total_counts = [v + i for v, i in zip(valid_counts, invalid_counts)]
    
    # Create unified color palette - each action type uses different base colors
    action_base_colors = {
        'type0_left': '#2196F3',    # Blue
        'type0_right': '#4CAF50',   # Green
        'type1': '#FF9800',         # Orange
        'type2': '#9C27B0',         # Purple
        'type3': '#F44336',         # Red
        'type4': '#00BCD4',         # Cyan
        'type5': '#795548',         # Brown
        'type6': '#607D8B'          # Blue-gray
    }
    
    # Generate colors for undefined action types
    import matplotlib.cm as cm
    base_cmap = cm.get_cmap('tab10')
    for i, name in enumerate(action_names):
        if name not in action_base_colors:
            action_base_colors[name] = base_cmap(i % 10)
    
    # Calculate overall statistics
    total_actions = sum(total_counts)
    total_valid = sum(valid_counts)
    success_rate = (total_valid / total_actions * 100) if total_actions > 0 else 0
    
    # Create chart with statistics in title
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), facecolor=bg_color)
    main_title = f'Action Distribution Analysis | Total Actions: {total_actions:,} ({success_rate:.1f}% Valid)'
    fig.suptitle(main_title, fontsize=14, color=text_color, fontweight='600', y=0.98)
    
    # Left chart: stacked bar chart showing valid/invalid distribution
    ax1.set_facecolor(bg_color)
    x_pos = np.arange(len(action_names))
    
    # Use light and dark versions to represent valid/invalid
    def lighten_color(color, amount=0.3):
        """Lighten color"""
        import matplotlib.colors as mc
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = np.array(mc.to_rgb(c))
        return tuple(c + (1 - c) * amount)
    
    def darken_color(color, amount=0.3):
        """Darken color"""
        import matplotlib.colors as mc
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = np.array(mc.to_rgb(c))
        return tuple(c * (1 - amount))
    
    # Create colors for each action type
    valid_colors = [lighten_color(action_base_colors[name], 0.2) for name in action_names]
    invalid_colors = [darken_color(action_base_colors[name], 0.3) for name in action_names]
    
    bars1 = ax1.bar(x_pos, valid_counts, color=valid_colors, alpha=0.9, label='Valid Actions')
    bars2 = ax1.bar(x_pos, invalid_counts, bottom=valid_counts, color=invalid_colors, alpha=0.9, label='Invalid Actions')
    
    # Add value labels on bar chart (only show segment values, not totals)
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
    
    # Set grid and styling
    ax1.grid(True, alpha=0.2, color=grid_color, linewidth=1, zorder=0, axis='y')
    ax1.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)
    
    # Legend
    legend1 = ax1.legend(loc='upper right', fontsize=10)
    legend1.get_frame().set_facecolor(card_bg)
    legend1.get_frame().set_edgecolor(grid_color)
    legend1.get_frame().set_alpha(0.95)
    for txt in legend1.get_texts():
        txt.set_color(text_color)
    
    # Right chart: pie chart showing overall action distribution
    ax2.set_facecolor(bg_color)
    
    # Only show actions with total count > 0, using unified base colors
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
        
        # Set percentage text styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    ax2.set_title('Total Action Distribution', fontsize=12, color=text_color, fontweight='600', pad=20)
    
    # Save chart
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Action distribution chart saved: {save_path}")
    return save_path


def plot_action_reward_distribution(action_counts_list: List[Dict], save_path: str) -> str:
    """
    Plot action reward distribution chart showing reward distribution for each action type
    
    Args:
        action_counts_list: List containing action count data from multiple episodes
                           Each element is a dict like {"type1": {"valid": 10, "invalid": 2, "rewards": [...]}, ...}
        save_path: Chart save path
        
    Returns:
        str: Final saved file path
    """
    # Merge reward data from all episodes
    combined_rewards = {}
    for episode_counts in action_counts_list:
        if not episode_counts:
            continue
        for action_name, counts in episode_counts.items():
            if action_name not in combined_rewards:
                combined_rewards[action_name] = []
            # Get reward data, ensure it's a list
            rewards = counts.get("rewards", [])
            if isinstance(rewards, list):
                combined_rewards[action_name].extend(rewards)
    
    if not combined_rewards or not any(combined_rewards.values()):
        logger.warning("No reward data available for plotting distribution chart")
        return save_path
    
    # Color scheme
    bg_color = '#1a1a1a'
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    # Action type colors - consistent with action distribution chart
    action_colors = {
        'type0_left': '#2196F3',    # Blue
        'type0_right': '#4CAF50',   # Green
        'type1': '#FF9800',         # Orange
    }
    
    # Filter out action types with data
    valid_actions = {name: rewards for name, rewards in combined_rewards.items() if rewards}
    
    if not valid_actions:
        logger.warning("No valid reward data")
        return save_path
    
    # Create chart
    plt.style.use('default')
    num_actions = len(valid_actions)
    
    if num_actions == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7), facecolor=bg_color)
        axes = [axes]
    elif num_actions == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=bg_color)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=bg_color)
    
    # Calculate overall statistics
    total_rewards = []
    for rewards in valid_actions.values():
        total_rewards.extend(rewards)
    
    overall_mean = np.mean(total_rewards) if total_rewards else 0
    overall_std = np.std(total_rewards) if total_rewards else 0
    
    # Clear title hierarchy
    main_title = 'Reward Distribution by Action Type'
    subtitle = f'Total Samples: {len(total_rewards):,} | Overall Mean: {overall_mean:.3f} ± {overall_std:.3f}'
    
    fig.suptitle(main_title, fontsize=16, color=text_color, fontweight='600', y=0.95)
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12, color=text_color, 
             fontweight='400', transform=fig.transFigure)
    
    # Pre-calculate all histograms to determine uniform Y-axis range
    all_hist_data = []
    for action_name, rewards in valid_actions.items():
        if rewards:
            n_bins = min(30, max(10, len(rewards) // 10))
            counts, _ = np.histogram(rewards, bins=n_bins)
            all_hist_data.extend(counts)
    
    # Set uniform Y-axis maximum value
    max_frequency = max(all_hist_data) if all_hist_data else 100
    y_max = max_frequency * 1.1  # Add 10% margin
    
    # Plot distribution chart for each action type
    for idx, (action_name, rewards) in enumerate(valid_actions.items()):
        ax = axes[idx] if num_actions > 1 else axes[0]
        ax.set_facecolor(bg_color)
        
        if not rewards:
            continue
            
        # Get color
        color = action_colors.get(action_name, '#666666')
        
        # Plot histogram
        n_bins = min(30, max(10, len(rewards) // 10))  # Adaptive bin count
        counts, bins, patches = ax.hist(rewards, bins=n_bins, color=color, alpha=0.7, 
                                       edgecolor='white', linewidth=0.8)
        
        # Plot density curve
        try:
            from scipy import stats
            if len(rewards) > 1:
                # Calculate kernel density estimation
                kde = stats.gaussian_kde(rewards)
                x_range = np.linspace(min(rewards), max(rewards), 100)
                density = kde(x_range)
                # Scale density to match histogram
                density_scaled = density * len(rewards) * (bins[1] - bins[0])
                ax.plot(x_range, density_scaled, color=color, linewidth=2, alpha=0.9)
        except ImportError:
            # Skip density curve if scipy is not available
            pass
        
        # Add statistical lines
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        
        # Mean line
        ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                  label=f'Mean: {mean_reward:.3f}')
        # Median line  
        ax.axvline(median_reward, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Median: {median_reward:.3f}')
        
        # Set title and labels
        std_reward = np.std(rewards)
        ax.set_title(f'{action_name}\n({len(rewards)} samples, σ={std_reward:.3f})', 
                    fontsize=12, color=text_color, fontweight='600', pad=20)
        ax.set_xlabel('Reward Value', fontsize=10, color=text_color)
        ax.set_ylabel('Frequency', fontsize=10, color=text_color)
        
        # Set uniform Y-axis range
        ax.set_ylim(0, y_max)
        
        # Set grid and styling
        ax.grid(True, alpha=0.2, color=grid_color, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(colors=text_color, labelsize=9)
        
        # Legend
        legend = ax.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor(grid_color)
        legend.get_frame().set_alpha(0.9)
        for txt in legend.get_texts():
            txt.set_color(text_color)
    
    # Hide excess subplots
    if num_actions < len(axes):
        for idx in range(num_actions, len(axes)):
            axes[idx].set_visible(False)
    
    # Save chart
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.subplots_adjust(top=0.80, bottom=0.12, left=0.08, right=0.95, wspace=0.25)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Action reward distribution chart saved: {save_path}")
    return save_path


def plot_avg_element_quality(avg_qualities: List[float],
                            episode_lengths: List[int],
                            save_path: str) -> str:
    """
    Plot the curve of average element quality vs cumulative timesteps during training
    
    Args:
        avg_qualities: List of average element quality for each episode
        episode_lengths: List of steps for each episode
        save_path: Chart save path
        
    Returns:
        str: Final saved file path
    """
    # Data validation and preprocessing
    qualities = np.array(avg_qualities, dtype=float)
    lengths = np.array(episode_lengths, dtype=float)
    
    if qualities.ndim != 1 or lengths.ndim != 1:
        raise ValueError("avg_qualities and episode_lengths must both be one-dimensional lists")
    
    if qualities.shape[0] != lengths.shape[0]:
        logger.warning(f"Qualities length {len(qualities)} and Lengths length {len(lengths)} do not match, truncating to minimum length")
        m = min(len(qualities), lengths)
        qualities = qualities[:m]
        lengths = lengths[:m]
    
    if len(qualities) == 0:
        logger.warning("No average element quality data available for plotting")
        return save_path
        
    # Calculate cumulative timesteps
    cumulative_steps = np.cumsum(lengths)
    
    # Color scheme
    bg_color = '#1a1a1a'
    primary_color = '#4CAF50'  # Green primary color
    secondary_color = '#81C784'  # Light green
    text_color = '#e8e8e8'
    grid_color = '#333333'
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot main curve
    ax.plot(cumulative_steps, qualities, color=primary_color, linewidth=2.5, 
            alpha=0.9, label='Avg Element Quality')
    
    # Add data points
    ax.scatter(cumulative_steps, qualities, color=secondary_color, s=25, 
              alpha=0.7, zorder=5)
    
    # Calculate and plot moving average line (smooth curve)
    if len(qualities) >= 10:
        window_size = max(5, len(qualities) // 20)  # Window size is 5% of data points, minimum 5
        moving_avg = np.convolve(qualities, np.ones(window_size)/window_size, mode='valid')
        moving_steps = cumulative_steps[window_size-1:]
        
        ax.plot(moving_steps, moving_avg, color='#FFA726', linewidth=3, 
                alpha=0.8, label=f'Moving Average ({window_size} episodes)')
    
    # Add statistical information
    mean_quality = np.mean(qualities)
    max_quality = np.max(qualities)
    min_quality = np.min(qualities)
    final_quality = qualities[-1]
    
    # Plot statistical lines
    ax.axhline(mean_quality, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean: {mean_quality:.4f}')
    ax.axhline(max_quality, color='orange', linestyle=':', linewidth=2, 
               alpha=0.6, label=f'Max: {max_quality:.4f}')
    
    # Set title and labels
    title = f'Average Element Quality Over Training\n'
    title += f'Final: {final_quality:.4f} | Range: [{min_quality:.4f}, {max_quality:.4f}]'
    ax.set_title(title, fontsize=16, color=text_color, fontweight='600', pad=20)
    
    ax.set_xlabel('Cumulative Training Steps', fontsize=12, color=text_color)
    ax.set_ylabel('Average Element Quality', fontsize=12, color=text_color)
    
    # Set grid and styling
    ax.grid(True, alpha=0.3, color=grid_color, linewidth=1)
    ax.set_axisbelow(True)
    
    # Remove top and right borders
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    
    # Set tick colors
    ax.tick_params(colors=text_color, labelsize=10)
    
    # Legend
    legend = ax.legend(loc='best', fontsize=11, framealpha=0.9)
    legend.get_frame().set_facecolor('#2a2a2a')
    legend.get_frame().set_edgecolor(grid_color)
    for txt in legend.get_texts():
        txt.set_color(text_color)
    
    # Add statistics text box
    stats_text = f'Episodes: {len(qualities)}\n'
    stats_text += f'Total Steps: {int(cumulative_steps[-1])}\n'
    stats_text += f'Std Dev: {np.std(qualities):.4f}\n'
    stats_text += f'Improvement: {final_quality - qualities[0]:.4f}'
    
    # Add statistics information in bottom right corner
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, color=text_color, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', 
                     edgecolor=grid_color, alpha=0.9))
    
    # Save chart
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Average element quality chart saved: {save_path}")
    return save_path
