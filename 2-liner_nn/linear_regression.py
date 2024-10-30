import random
import torch
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))      # X(1000, len(w))
    y = torch.matmul(X, w) + b                          # y(1000, 1)
    y += torch.normal(0, 0.01, y.shape)                 # 添加噪声
    return X, y.reshape((-1, 1))                        # -1表示大小自动判断，1表示第二维大小为1，即调整为列向量

features, labels = synthetic_data(true_w, true_b, 1000)
'''
print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()

d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()  # 显示图形
'''

# 读取数据集:将数据集打乱然后每次抽batch_size个数据出来实验
def data_iter(batch_size, features, labels):
    num_examples = len(features)       

    indices = list(range(num_examples))                # 生成样本数量个索引        
    random.shuffle(indices)                            # 打乱索引

    for i in range(0, num_examples, batch_size):

        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] # 生成器

batch_size = 10
'''
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
'''

# 线性回归模型
def linreg(X, w, b):
    
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(y_hat, y):
    
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 优化算法：小批量随机梯度下降
def sgd(params, lr, batch_size):

    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # w:(2,1)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
num_epochs = 3
train_l = squared_loss(linreg(features, w, b), labels)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        print(X.shape)
        l = squared_loss(linreg(X, w, b), y)  # X和y的小批量损失
        #print(l)
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(w, b)
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')