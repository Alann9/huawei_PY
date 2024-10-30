import torch
from IPython import display
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 加载70000/batch_size张图片
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# num_inputs输入维度：28 * 28 = 784， num_outputs输出维度：10个类别
num_inputs = 784
num_outputs = 10

# 随机初始化参数，并开启梯度跟踪
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义网络
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 损失函数（交叉熵）
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# 学习率和迭代次数
lr = 0.1
num_epochs = 10

# 参数优化
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.predict_ch3(net, test_iter, n= 10)
plt.show()