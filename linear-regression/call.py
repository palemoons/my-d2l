"""
Concise implementation of linear regression by calling library function.
"""
# import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l


def data_generation(data_arrays, batch_size, is_train=True):
    """Contruct data iterator based on PyTorch"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# Generate dataset.
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# Set batch_size
batch_size = 10
# Declare data generator.
data_iter = data_generation((features, labels), batch_size)
# Define network.
# A fully connected layer.
net = torch.nn.Sequential(torch.nn.Linear(2, 1))
# Initalize net params.
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# Define loss function(Mean Square Error)
loss = torch.nn.MSELoss(reduction='mean')
# Define optimizer(mini-batch stochastic gradient descent)
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# Train.
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # Clear gradients.
        l.backward()  # Calculate gradients of loss function.
        trainer.step()  # Performs a single optimization step (parameter update).
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")

# Get trained params.
w = net[0].weight.data
print("w的估计误差：", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("b的估计误差：", true_b - b)
