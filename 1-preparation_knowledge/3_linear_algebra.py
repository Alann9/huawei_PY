import torch
import numpy as np

# 标量
x = torch.tensor(5)
y = torch.tensor(2.0)
# print(x)

# 向量：标量值组成的列表
x = torch.arange(4)
# print(x)

# 矩阵：正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶
X = torch.arange(20, dtype=int).reshape(5, 4)
# print(X)

# 张量： 具有任意数量轴的n维数组
X = torch.arange(24).reshape(2, 3, 4)
Y = X.clone()
Z = torch.arange(12).reshape(1, 3, 4)
a = 2

# 张量按元素乘(遵守广播机制)
# print(X * Y, a * X, X * Z)

# 降维
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
A_sum_axis1 = A.sum(axis=0)
# print(A, A_sum_axis1)

# 点积：向量与向量之间的运算,向量相乘求和
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
# 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
# print(x, y, torch.dot(x, y), torch.sum(x * y))

# 矩阵 向量积 ,矩阵 矩阵乘法遵守矩阵乘法

# 范数
