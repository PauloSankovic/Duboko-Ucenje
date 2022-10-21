import numpy as np
from data import *


def sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)


def binlogreg_train(X: np.ndarray, Y_: np.ndarray, param_niter=100000, param_delta=0.1, verbose=False):
    """
      Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

      Povratne vrijednosti
        w, b: parametri logističke regresije
    """
    w = np.random.randn(X.shape[1])
    b = 0

    # gradijentni spust
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b               # N x 1

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores)                 # N x 1

        # gubitak
        loss = - np.sum(np.log(probs))          # scalar

        # dijagnostički ispis
        if verbose and i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_                 # N x 1

        # gradijenti parametara
        N = len(X)
        grad_w = 1 / N * np.dot(dL_dscores, X)  # D x 1
        grad_b = 1 / N * sum(dL_dscores)        # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """
      Argumenti
          X:    podatci, np.array NxD
          w, b: parametri logističke regresije 

      Povratne vrijednosti
          probs: vjerojatnosti razreda c1
    """
    return sigmoid(np.dot(X, w) + b)


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)

    return classify


if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = probs > 0.5

    # report performance
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_ap(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
