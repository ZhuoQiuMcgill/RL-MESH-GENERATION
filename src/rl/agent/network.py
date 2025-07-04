import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# TODO: 根据论文3.1.3节和图11的结论，这里定义了SAC算法所需的神经网络结构。
# 最佳结构S2被选为默认配置：[128, 128, 128]。

class Actor(nn.Module):
    """
    策略网络 (Actor): 输入状态(state), 输出动作(action)。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        # TODO: 实现策略网络的结构。
        # 它应该有几个线性层(Linear)和激活函数(ReLU)。
        # 网络的最后两层应分别输出动作的均值(mu)和标准差(log_std)，用于构建一个正态分布，
        # 从而实现SAC的随机策略。
        # max_action 用于将输出的动作缩放到正确的范围内。
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # TODO: 实现网络的前向传播。
        # 1. state通过线性层和ReLU激活函数。
        # 2. 计算均值 mu 和对数标准差 log_std。
        # 3. 限制 log_std 的范围以保证训练稳定。
        # 4. 从构建的正态分布 N(mu, std) 中采样动作，并应用 Tanh 函数进行归一化。
        # 5. 计算采样动作的对数概率 log_prob，这是SAC算法最大化熵的关键部分。
        # 6. 返回最终的动作、对数概率和均值。
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制范围
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        u = dist.rsample()  # 重参数化技巧
        action = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)

        return action * self.max_action, log_prob.sum(-1, keepdim=True), torch.tanh(mu) * self.max_action


class Critic(nn.Module):
    """
    Q值网络 (Critic): 输入状态(state)和动作(action), 输出Q值。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # TODO: 实现Q值网络的结构。
        # 根据论文，SAC使用两个独立的Q网络（Twin Critics）来缓解Q值过高估计的问题。
        # 第一个Q网络
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

        # 第二个Q网络
        self.l5 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.l7 = nn.Linear(hidden_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # TODO: 实现前向传播，计算两个Q值。
        # 将 state 和 action 拼接后作为输入。
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2
