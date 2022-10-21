from data import *


class FcAnn:
    def __init__(self, input_n: int, hidden_n: int, classes_n: int):
        # xavier distribution
        self.W1 = np.random.normal(loc=0, scale=1 / np.mean([input_n, hidden_n]), size=(input_n, hidden_n))
        self.b1 = np.zeros((1, hidden_n))
        self.W2 = np.random.normal(loc=0, scale=1 / np.mean([hidden_n, classes_n]), size=(hidden_n, classes_n))
        self.b2 = np.zeros((1, classes_n))

    def train(self, X, Y_, param_niter: int, param_delta: float, param_lambda: float, verbose: bool = True):
        for i in range(param_niter):
            # forward
            s1 = X @ self.W1 + self.b1
            h1 = relu(s1)
            s2 = h1 @ self.W2 + self.b2
            Y = softmax(s2)

            if verbose and i % 10000 == 0:
                loss = self.cross_entropy_loss(Y, Y_) + param_lambda * (
                        np.linalg.norm(self.W1) + np.linalg.norm(self.W2))
                print(f"{i} > loss: {loss:.06f}")

            ## backprop
            # gradijenti gubitka obzirom na linearnu mjeru drugog sloja u svim podatcima - N x C
            grads_W2_2 = Y
            grads_W2_2[range(Y.shape[0]), Y_] -= 1
            grads_W2_2 /= Y.shape[0]

            # gradijenti gubitka obzirom na parametre drugog sloja - C x H
            grads_W2 = h1.T @ grads_W2_2
            grads_b2 = sum(grads_W2_2)

            # gradijenti gubitka obzirom na nelinearni izlaz prvog sloja u svim podatcima - N x H
            grads_W1_1 = grads_W2_2 @ self.W2.T

            # gradijent gubitka obzirom na linearnu mjeru prvog sloja u svim podatcima - N x H
            grads_W1_1[s1 < 0] = 0

            # gradijenti gubitka obzirom na parametre prvog sloja - H x D i H x 1
            grads_W1 = X.T @ grads_W1_1
            grads_b1 = sum(grads_W1_1)

            self.W2 -= param_delta * grads_W2
            self.b2 -= param_delta * grads_b2
            self.W1 -= param_delta * grads_W1
            self.b1 -= param_delta * grads_b1

    def cross_entropy_loss(self, Y, Y_):
        return - np.mean(np.sum(np.log(Y[range(Y.shape[0]), Y_] + 1e-13)))

    def classify(self, X):
        s1 = X @ self.W1 + self.b1
        h1 = relu(s1)
        s2 = h1 @ self.W2 + self.b2
        py = softmax(s2)
        return np.argmax(py, axis=1)


def relu(X):
    return np.maximum(0, X)


def softmax(X):
    e_x = np.exp(X)
    return e_x / (np.sum(e_x, axis=1))[:,
                 np.newaxis]  # increase the dimension of the existing array by one more dimension


if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(6, 2, 10)
    model = FcAnn(2, 5, 2)
    model.train(X, Y_, 100000, 0.05, 1e-3)
    Y = model.classify(X)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(model.classify, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y)
    plt.show()
