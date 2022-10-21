import torch
import numpy as np
from torch import nn
from torch.nn import Parameter
from torch.optim import SGD

import data


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        self.W = Parameter(torch.randn(size=(D, C), dtype=torch.float, requires_grad=True))
        self.b = Parameter(torch.zeros(size=(C,), dtype=torch.float, requires_grad=True))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        logits = X.float().mm(self.W) + self.b
        return torch.softmax(logits, dim=1)

    def get_loss(self, X, Yoh_) -> torch.Tensor:
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        Y = self.forward(X)
        return - torch.mean(torch.sum(torch.log(Y + 1e-13) * Yoh_, dim=1))


def train(model: PTLogreg, X, Yoh_, param_niter, learning_rate, param_lambda=0, verbose=False):
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
        loss = model.get_loss(X, Yoh_) + param_lambda * torch.norm(model.W)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            print(f"{i} > loss = {loss:.06f}")


def eval(model: PTLogreg, X: np.ndarray) -> np.ndarray:
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
    X, Y_ = data.sample_gmm_2d(4, 3, 10)
    Yoh_ = data.class_to_onehot(Y_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    X = torch.tensor(X, dtype=torch.float)
    Yoh_ = torch.tensor(Yoh_, dtype=torch.float)
    train(ptlr, X, Yoh_, 1000, learning_rate=0.01)

    # dohvati vjerojatnosti na skupu za učenje
    Y = eval(ptlr, X.numpy())

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    data.graph_surface(lambda x: eval(ptlr, x), bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    data.plt.show()
