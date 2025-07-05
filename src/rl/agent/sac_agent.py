import torch
import torch.nn.functional as F
from copy import deepcopy
from .network import Actor, Critic
from ..config import load_config


class SACAgent:
    """
    实现了Soft Actor-Critic (SAC)算法的智能体。
    """

    def __init__(self, state_dim, action_dim, max_action, device, config=None):
        self.device = device

        cfg = load_config() if config is None else config
        sac_cfg = cfg.get("sac_agent", {})

        hidden_dim = sac_cfg.get("hidden_dim", 128)
        self.gamma = sac_cfg.get("gamma", 0.99)
        self.tau = sac_cfg.get("tau", 0.005)

        # 初始化 Actor、Critic 以及对应的目标网络。
        # Critic_target 是 Critic 的深拷贝，其参数会以软更新方式延迟更新。
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim, max_action=max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        # 初始化优化器，学习率等超参数参考论文表1。
        actor_lr = sac_cfg.get("actor_lr", 3e-4)
        critic_lr = sac_cfg.get("critic_lr", 3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 初始化自动熵调优相关的参数。alpha（温度参数）会在训练过程中自动学习。
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_lr = sac_cfg.get("alpha_lr", 3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()

        self.max_action = max_action

    def select_action(self, state):
        """在评估模式下，根据当前状态选择确定性动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # 在评估时不使用随机性，直接用均值
        _, _, action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """执行一次参数更新，对应论文 Algorithm 1 的主要循环"""
        # 1. 从 replay_buffer 中采样一个批次的数据 (state, action, reward, next_state, done)

        # 2. 计算 Critic 损失：
        #    a. 使用 actor 从 next_state 计算 next_action 和 log_prob。
        #    b. 使用 critic_target 计算目标Q值，这是SAC的关键部分：
        #       target_Q = reward + (1 - done) * gamma * (min(Q1_target, Q2_target) - alpha * log_prob)
        #    c. 计算当前Q值 (current_Q1, current_Q2)。
        #    d. Critic的损失是当前Q值和目标Q值的均方误差(MSE)。

        # 3. 更新 Critic 网络。

        # 4. 计算 Actor 损失：
        #    a. 使用 actor 从 state 重新计算动作和 log_prob。
        #    b. Actor 的目标是最大化Q值和策略熵：
        #       actor_loss = (alpha * log_prob - min(Q1, Q2)).mean()

        # 5. 更新 Actor 网络。

        # 6. 更新温度参数 alpha：
        #    a. alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
        #    b. 更新 log_alpha。

        # 7. 软更新 Critic_target 网络：
        #    target_params = tau * current_params + (1 - tau) * target_params
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_q = reward + (1 - done) * self.gamma * (target_q - alpha * next_log_prob)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob, _ = self.actor(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
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
