from data import *


def my_dummy_decision(X):
    return X[:, 0] + X[:, 1] - 5


if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = my_dummy_decision(X) > 0.5

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()
