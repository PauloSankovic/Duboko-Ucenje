from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, lr_scheduler

from utils import *

DATA_DIR = Path(__file__).parent / 'datasets' / 'cifar-10-batches-py'
SAVE_DIR = Path(__file__).parent / 'out4'

config = {
    'max_epochs': 8,
    'batch_size': 50,
    'lr': 1e-1,
    'weight_decay': 1e-3,
    'c': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}


def init_data():
    img_height = 32
    img_width = 32
    num_channels = 3

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, data_mean, data_std


class TorchConvolutionModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=10, bias=True)

        self.loss_function = CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc3:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc3.reset_parameters()

    def forward(self, x):
        # N * 3 * 32 * 32
        h = self.conv1(x)
        # N * 16 * 28 * 28
        h = torch.relu(h)
        h = self.pool1(h)

        # N * 16 * 13 * 13
        h = self.conv2(h)
        # N * 32 * 9 * 9
        h = torch.relu(h)
        h = self.pool2(h)

        # N * 32 * 4 * 4
        h = h.view(h.shape[0], -1)

        # 512
        h = self.fc1(h)
        h = torch.relu(h)

        h = self.fc2(h)
        h = torch.relu(h)

        return self.fc3(h)

    def get_loss(self, logits, target):
        return self.loss_function(logits, target)


def evaluate(model: TorchConvolutionModel, X, Y):
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y)
    model.eval()

    with torch.no_grad():
        logits = model.forward(X)
        predictions = np.argmax(logits, axis=1)
        loss = model.get_loss(logits, Y)

    accuracy = accuracy_score(Y, predictions)

    return loss, accuracy


def main() -> None:
    plot_data = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}

    train_x, train_y, valid_x, valid_y, test_x, test_y, data_mean, data_std = init_data()

    model = TorchConvolutionModel()
    print(model)

    optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(config['max_epochs']):
        X, Y = shuffle_data(train_x, train_y)
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        batch_size = config['batch_size']
        n_batch = len(train_x) // batch_size
        for batch in range(n_batch):
            # broj primjera djeljiv s veličinom grupe batch_size
            batch_X = X[batch * batch_size:(batch + 1) * batch_size, :]
            batch_Y = Y[batch * batch_size:(batch + 1) * batch_size]

            logits = model.forward(batch_X)
            loss = model.loss_function(logits, batch_Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, n_batch, float(loss)))

            if batch % 200 == 0:
                draw_conv_filters(epoch, batch, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

        train_loss, train_acc = evaluate(model, train_x, train_y)
        print(f"Train: loss > {train_loss:.06f}, accuracy > {train_acc * 100:.02f}%")
        val_loss, val_acc = evaluate(model, valid_x, valid_y)
        print(f"Validation: loss > {val_loss:.06f}, accuracy > {val_acc * 100:.02f}%")

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [scheduler.get_last_lr()]
        scheduler.step()

    plot_training_progress(SAVE_DIR, plot_data)


if __name__ == '__main__':
    main()
