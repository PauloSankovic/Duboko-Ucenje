import torch
from torch import nn
from torch.nn import Parameter, ParameterList
from torch.optim import SGD

import numpy as np

import data


class PTDeep(nn.Module):
    def __init__(self, layers: list, activation_fun):
        super().__init__()

        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weight = Parameter(torch.randn((layers[i], layers[i + 1]), requires_grad=True))
            bias = Parameter(torch.zeros((1, layers[i + 1])), requires_grad=True)

            weights.append(weight)
            biases.append(bias)

        self.weights = ParameterList(weights)
        self.biases = ParameterList(biases)
        self.activation_fun = activation_fun

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        s = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            s = self.activation_fun(s.float().mm(W) + b)
        return torch.softmax(s.float().mm(self.weights[-1]) + self.biases[-1], dim=1)

    def get_loss(self, X, Yoh_) -> torch.Tensor:
        Y = self.forward(X)
        return - torch.mean(torch.sum(torch.log(Y + 1e-13) * Yoh_, dim=1))

    def count_params(self):
        params = []
        for name, param in self.named_parameters():
            params.append((name, param.shape))

        total_parameters = sum(p.numel() for p in self.parameters())
        return params, total_parameters


def train(model: PTDeep, X, Yoh_, param_niter, learning_rate, param_lambda=0, verbose=False):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """
    # inicijalizacija optimizatora
    # ...
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            print(f"{i} > loss = {loss:.06f}")


def eval(model: PTDeep, X: np.ndarray) -> np.ndarray:
    """Arguments:
       - model: type: PTLogreg
       - X: actual data-points [NxD], type: np.array
       Returns: predicted class probabilities [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    return np.argmax(model.forward(torch.from_numpy(X)).detach().cpu().numpy(), axis=1)


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)

    # definiraj model:
    pt_deep = PTDeep([2, 10, 2], torch.sigmoid)
    params, total_parameters = pt_deep.count_params()
    print("Parameters:", params)
    print("Total parameters:", total_parameters)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    X = torch.tensor(X, dtype=torch.float)
    Yoh_ = torch.tensor(Yoh_, dtype=torch.float)
    train(pt_deep, X, Yoh_, 1000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    Y = eval(pt_deep, X.numpy())

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    data.graph_surface(lambda x: eval(pt_deep, x), bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    data.plt.show()
