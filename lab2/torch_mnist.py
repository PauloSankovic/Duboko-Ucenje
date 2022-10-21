from pathlib import Path

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out3'

config = {
    'max_epochs': 8,
    'batch_size': 50,
    'weight_decay': 1e-3,
    'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}},
    'verbose': True
}


class TorchConvolutionModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                               bias=True)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                               bias=True)
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.fc1 = nn.Linear(in_features=7 * 7 * 32, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc2:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc2.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = torch.relu(h)

        h = self.conv2(h)
        h = self.pool2(h)
        h = torch.relu(h)

        h = h.view(h.shape[0], -1)

        h = self.fc1(h)
        h = torch.relu(h)
        return self.fc2(h)


def train(model: TorchConvolutionModel, dataset: DataLoader, loss_function):
    for epoch in range(1, config['max_epochs'] + 1):
        if epoch in config['lr_policy']:
            lr = config['lr_policy'][epoch]['lr']

        optimizer = SGD(model.parameters(), lr=lr, weight_decay=config['weight_decay'])

        current_loss = 0
        for it, (x, y) in enumerate(dataset):
            logits = model.forward(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()

            current_loss += float(loss)
            optimizer.zero_grad()

        print(f"Epoch {epoch}:\n\tLoss > {current_loss / len(dataset)}")


def evaluate(model: TorchConvolutionModel, dataset: DataLoader, loss_function):
    current_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for it, (x, y) in enumerate(dataset):
            logits = model.forward(x)
            loss = loss_function(logits, y)

            logits = softmax(logits, dim=1)
            for i, p in enumerate(logits):
                if y[i] == torch.max(p.data, 0)[1]:
                    correct += 1

            current_loss += float(loss)

    loss = current_loss / len(dataset)
    accuracy = correct / len(dataset.dataset)

    print(f"loss >  {loss:.06}, accuracy: {accuracy * 100:.03}%")


def main() -> None:
    mnist_train_set = MNIST(DATA_DIR, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test_set = MNIST(DATA_DIR, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    mnist_train_set, mnist_valid_set = random_split(
        mnist_train_set,
        [int(0.8 * len(mnist_train_set)), int(0.2 * len(mnist_train_set))]
    )

    train_dl = DataLoader(dataset=mnist_train_set, batch_size=config['batch_size'], shuffle=True)
    valid_dl = DataLoader(dataset=mnist_valid_set, batch_size=config['batch_size'], shuffle=True)
    test_dl = DataLoader(dataset=mnist_test_set, batch_size=config['batch_size'], shuffle=True)

    model = TorchConvolutionModel()
    loss_function = CrossEntropyLoss()

    train(model, train_dl, loss_function)

    print("Validation dataset:")
    evaluate(model, valid_dl, loss_function)
    print("\nTest dataset:")
    evaluate(model, test_dl, loss_function)


if __name__ == '__main__':
    main()
