from torch import nn


def make_model(name):
    if name == 'linear':
        return nn.Sequential(nn.Flatten(1), nn.Linear(784, 10))
    if name == 'mlp':
        return nn.Sequential(nn.Flatten(1), nn.Linear(784, 64),
                             nn.ReLU(), nn.Linear(64, 10))
    if name == 'cnn1':
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Flatten(1),
                             nn.Linear(8 * 14 * 14, 10))
    if name == 'cnn2':
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                             nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(1),
                             nn.Linear(16 * 7 * 7, 10))
    raise ValueError(name)
