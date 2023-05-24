"""
Concise implementation of linear regression by calling library function.
"""
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms


def get_dataloader_workers():
    return torch.get_num_threads()


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

batch_size = 256
# Load data.
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# Define net.
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# Initialize parameters.
net.apply(init_weights)
# Define loss function.
loss = nn.CrossEntropyLoss(reduction="none")
# Define optimizer.
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# Train
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
