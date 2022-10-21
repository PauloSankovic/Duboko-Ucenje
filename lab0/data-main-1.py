from data import *

if __name__ == '__main__':
    np.random.seed(100)
    g = Random2DGaussian()
    X = g.get_sample(100)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
