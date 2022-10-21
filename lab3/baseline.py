import torch
from torch import mean, relu
from torch.nn import Module, Linear


class Baseline(Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.mean = mean
        self.fc1 = Linear(in_features=300, out_features=150, bias=True)
        self.fc2 = Linear(in_features=150, out_features=150, bias=True)
        self.fc3 = Linear(in_features=150, out_features=1, bias=True)
        self.relu = relu

    def forward(self, data):
        x = self.embeddings(data)
        x = self.mean(x, dim=1, dtype=torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
