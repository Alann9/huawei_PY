import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None

y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)