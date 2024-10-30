
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

x = torch.arange(24).reshape((1, 4, 3, 2))

y = torch.arange(72).reshape((3, 4, 3, 2))

print(x.shape, y.shape)
print(x + y)

X = torch.arange(20, dtype=torch.float32).reshape((4,5))
print(X[1:3, 2:4]) # 数据切片

# 广播机制：1.向量x或向量y的n维中至少有一个维度是一维的，该维度可以广播扩展
#          2.剩下的维度必须要么相同，要么为 1，这样才可以通过广播机制进行扩展。