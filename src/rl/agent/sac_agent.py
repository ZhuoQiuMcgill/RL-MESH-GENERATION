import torch
import torch.nn.functional as F
from copy import deepcopy
from .network import Actor, Critic
from ..config import load_config


class SACAgent:
    """
    实现了Soft Actor-Critic (SAC)算法的智能体。
    支持普通经验回放、优先级经验回放(PER)和在线学习模式。
    现在支持4种动作类型：ActionType0Left, ActionType0Right, ActionType1, ActionType2
    """

    def __init__(self, state_dim, action_dim, max_action, device, config=None):
        self.device = device

        cfg = load_config() if config is None else config
        sac_cfg = cfg.get("sac_agent", {})

        # 确保所有数值参数都转换为正确的类型
        hidden_dim = int(sac_cfg.get("hidden_dim", 128))
        self.gamma = float(sac_cfg.get("gamma", 0.99))
        self.tau = float(sac_cfg.get("tau", 0.005))

        # 初始化 Actor、Critic 以及对应的目标网络。
        # Critic_target 是 Critic 的深拷贝，其参数会以软更新方式延迟更新。
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim, max_action=max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        # 初始化优化器，学习率等超参数参考论文表1。
        # 确保学习率被正确转换为浮点数
        actor_lr = float(sac_cfg.get("actor_lr", 3e-4))
        critic_lr = float(sac_cfg.get("critic_lr", 3e-4))
        alpha_lr = float(sac_cfg.get("alpha_lr", 3e-4))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=sac_cfg.get("weight_decay", 1e-4)
        )

        # 初始化自动熵调优相关的参数。alpha（温度参数）会在训练过程中自动学习。
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()

        self.max_action = max_action

    def select_action(self, state):
        """
        在评估模式下，根据当前状态选择确定性动作
        输出的动作范围为[-1, 1]，环境会将其解码为4种动作类型之一
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # 在评估时不使用随机性，直接用均值
        _, _, action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        执行一次参数更新，对应论文 Algorithm 1 的主要循环
        支持普通经验回放、优先级经验回放(PER)和在线学习模式

        Args:
            replay_buffer: 经验回放缓冲区，可以是ReplayBuffer、PrioritizedReplayBuffer或NoReplayBuffer
            batch_size: 批次大小

        Returns:
            dict: 包含损失和指标的字典

        Raises:
            RuntimeError: 当在线学习模式下调用此方法时
        """
        # 检查是否为在线学习模式
        if hasattr(replay_buffer, 'get_statistics') and replay_buffer.get_statistics().get("mode") == "online_learning":
            raise RuntimeError("在线学习模式下请使用 train_online() 方法")

        # 检测是否为优先级经验回放
        is_prioritized = hasattr(replay_buffer, 'update_priorities')

        if is_prioritized:
            # 优先级经验回放
            state, action, reward, next_state, done, weights, indices = replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            # 普通经验回放
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size, 1).to(self.device)  # 均匀权重
            indices = None

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # 获取当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # 计算TD误差（用于PER）
        td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy().flatten()

        # 计算Critic损失
        critic_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none')).mean() + \
                      (weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        new_action, log_prob, _ = self.actor(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新Alpha（温度参数）
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 更新优先级经验回放的优先级
        if is_prioritized and indices is not None:
            replay_buffer.update_priorities(indices, td_errors)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "mean_td_error": float(td_errors.mean()),
            "max_td_error": float(td_errors.max()),
        }

    def train_online(self, state, action, reward, next_state, done):
        """
        在线学习模式：使用单个样本进行训练

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束episode

        Returns:
            dict: 包含损失和指标的字典
        """
        # 转换为torch张量并添加batch维度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # 获取当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # 计算TD误差
        td_error = torch.abs(current_q1 - target_q).item()

        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        new_action, log_prob, _ = self.actor(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新Alpha（温度参数）
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "td_error": float(td_error),
        }

    def save(self, filename):
        """保存模型权重"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """加载模型权重"""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
