import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# 根据论文3.1.3节和图11的结论，这里定义了SAC算法所需的神经网络结构。
# 最佳结构S2被选为默认配置：[128, 128, 128]。

class Actor(nn.Module):
    """
    策略网络 (Actor): 输入状态(state), 输出动作(action)。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        # 策略网络由三层全连接层和ReLU激活函数构成。
        # 最后两层分别输出动作均值(mu)和对数标准差(log_std)。
        # max_action 控制动作的输出范围。
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        """前向传播，返回采样动作、对数概率以及均值动作"""
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
        # 根据论文，SAC 使用两个独立的 Q 网络 (Twin Critics) 来缓解 Q 值的过高估计。
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
        """计算两个 Q 值"""
        # 将 state 和 action 拼接后作为输入
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
