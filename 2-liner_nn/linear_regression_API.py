import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# feature (1000, len(true_w))  labels (1000, 1) 表示有两个参数，一个输出的全连接层

batch_size = 10
data_iter = d2l.load_array((features, labels), batch_size)
# data_iter是batch_size组(features, labels)的组合
print(next(iter(data_iter)))

# 定义一个两个输入一个输出的网络层
net = nn.Sequential(nn.Linear(2, 1))

# net[0]表示第一个层，本代码只有一层
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):

    for X, y in data_iter:
        l = loss(net(X) ,y)       # 前向传播，计算该样本的损失
        trainer.zero_grad()       # 清空梯度
        l.backward()              # 反向传播，计算梯度
        trainer.step()            # 更新参数
    l = loss(net(features), labels)

    print(f'epoch {epoch + 1}, loss {l:f}')
