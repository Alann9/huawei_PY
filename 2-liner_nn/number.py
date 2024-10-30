import torch
from IPython import display
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision
from torch.utils import data

batch_size = 256

if __name__ == "__main__":
    train_iter, test_iter = d2l.load_data_mnist(batch_size)
    for X, y in train_iter:
        d2l.show_images(X[:10].squeeze(), 2, 5)  # 显示 2 行 5 列的图片
        plt.show()
        break

