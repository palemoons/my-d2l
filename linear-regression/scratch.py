"""
手搓线性回归方法
"""
import random
import torch


def gen_dataset(w, b, numbers):
    """
    generate sample dataset.

    params:
        w: weight
        b: bias
        numbers: sample numbers

    return:
        X: features
        y: labels
    """
    X = torch.normal(0, 1, (numbers, len(w)))
    y = torch.matmul(X, w) + b
    # add noise
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_generation(batch_size, features, labels):
    """
    data iterator for training.

    params:
        batch_size: batch size
        features: features
        labels: labels

    return:
        X: features
        y: labels
    """
    numbers = len(features)
    indices = list(range(numbers))
    random.shuffle(indices)
    for i in range(0, numbers, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, numbers)])
        yield features[batch_indices], labels[batch_indices]


def model(X, w, b):
    """
    linear regression model.

    params:
        X: features
        w: weight
        b: bias

    return:
        y: labels
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    squared loss function.
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    simple stochastic gradient descent.
    """
    with torch.no_grad():
        # disable gradient calculation
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # clear gradient


# hyper-parameter
lr = 0.03
epochs = 3
batch_size = 10

# dataset
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = gen_dataset(true_w, true_b, 1000)

# initialize w, b
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(epochs):
    # mini-batch stochastic gradient descent
    for X, y in data_generation(batch_size, features, labels):
        l = squared_loss(model(X, w, b), y)  # declare loss function on w, b
        l.sum().backward()  # calculate loss function's gradient of 10 samples
        with torch.no_grad():
        # disable gradient calculation
          for param in [w, b]:
              param -= lr * param.grad / batch_size
              param.grad.zero_() # clear gradient
    with torch.no_grad():
        train_l = squared_loss(model(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
