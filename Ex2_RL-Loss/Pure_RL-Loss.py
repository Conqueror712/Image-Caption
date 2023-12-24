"""
【背景及使用原因】
在深度学习中，常常通过最小化交叉熵损失来训练模型，而模型的好坏则由某种评测指标来衡量。
这种情况下，交叉熵损失可以看作是默认的训练目标，而评测指标是我们真正关心的指标。
但很多时候，优化交叉熵损失并不一定能直接优化我们关心的评测指标。
比如在分类任务中，交叉熵损失会关注每个类别是否正确预测，与我们的实际目标，比如整体预测准确率，可能不一致。
这就是所谓的默认实现的交叉熵损失和评测指标不一致情况。

在这种情况下，可以使用基于强化学习的方法来设定损失函数，使之直接优化我们关心的指标。
比方说，对于策略梯度方法而言，构造奖励函数以及策略网络，通过互动得到的奖励来更新策略网络，奖励函数就是评测指标。

举一个简单的例子，如果我们的评测指标是准确率，那么每次预测对我们就给予+1的奖励，预测错我们就不给奖励。
我们的策略网络就是我们的预测模型，输出的就是预测结果。
然后我们利用策略梯度方法，不断通过互动得到的奖励来更新我们的预测模型，使之更好地优化我们关心的指标。
"""

import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

def policy_gradient_update(model, states, actions, rewards, optimizer):
    # 获取模型预测的动作概率
    action_probs = model(states)
    picked_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 根据公式计算损失
    loss = (-torch.log(picked_action_probs) * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model = Model(input_size=10, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

states = torch.randn(100, 10)
actions = torch.randint(0, 2, (100,))
rewards = torch.randn(100)

policy_gradient_update(model, states, actions, rewards, optimizer)

"""
如何将强化学习损失函数放进train代码中呢：
# 示例的训练过程
for epoch in range(num_epochs):
    # 对于每个批次的数据
    for batch_data in data_loader:        
        # 从批次数据中获取输入，动作和奖励
        states, actions, rewards = batch_data
        # 使用强化学习损失函数更新模型
        policy_gradient_update(model, states, actions, rewards, optimizer)

# 验证或测试过程
for batch_data in validation_data_loader:
    # 从批次数据中获取输入
    states = batch_data
    # 用模型对输入进行预测
    action_probabilities = model(states)
    # 根据需求评估或使用预测结果
    ...
"""