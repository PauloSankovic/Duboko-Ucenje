from data import *

def my_dummy_decision(X):
    return X[:, 0] + X[:, 1] - 5


if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gmm_2d(4, 2, 30)

    # get the class predictions
    Y = my_dummy_decision(X) > 0.5

    # define function domain
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    # graph the surface plot
    graph_surface(my_dummy_decision, bbox, offset=0)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()