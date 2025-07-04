import torch
import torch.nn.functional as F
from copy import deepcopy
from .network import Actor, Critic


class SACAgent:
    """
    实现了Soft Actor-Critic (SAC)算法的智能体。
    """

    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device

        # TODO: 初始化所有网络和目标网络。
        # Actor, Critic, Critic_target。
        # Critic_target 是 Critic 的深拷贝，其参数会延迟更新。
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        # TODO: 初始化优化器，分别为Actor和Critic设置。
        # 学习率等超参数参考论文表1。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # TODO: 初始化自动熵调优相关的参数。
        # alpha（温度参数）是SAC的核心，它平衡奖励和策略熵。
        # 我们让alpha可以被自动学习和调整。
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()

        self.max_action = max_action

    def select_action(self, state):
        # TODO: 实现动作选择逻辑。
        # 在评估模式下，此函数根据当前状态选择一个动作。
        # 1. 将 state 转换为 PyTorch 张量。
        # 2. 使用 actor 网络获取确定性动作（使用均值）。
        # 3. 将动作张量转换回 numpy 数组。
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # 在评估时不使用随机性，直接用均值
        _, _, action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # TODO: 实现单步训练逻辑，对应论文 Algorithm 1 的内循环。
        # 1. 从 replay_buffer 中采样一个批次的数据 (state, action, reward, next_state, done)。

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
        pass

    def save(self, filename):
        # TODO: 实现保存模型权重的逻辑。
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        # TODO: 实现加载模型权重的逻辑。
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
