import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    def __init__(self, min_x: int = 0, max_x: int = 10, min_y: int = 0, max_y: int = 10):
        dx = max_x - min_x
        dy = max_y - min_y
        self.mean = (min_x, min_y) + np.random.random_sample(2) * (dx, dy)

        eigen_values = (np.random.random_sample(2) * (dx / 5, dy / 5)) ** 2
        theta = np.random.random_sample() * 2 * np.pi
        rotation_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

        self.covariance_matrix = np.dot(np.dot(np.transpose(rotation_matrix), np.diag(eigen_values)), rotation_matrix)

    def get_sample(self, n: int) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.covariance_matrix, n)


def sample_gmm_2d(n_components: int, n_classes: int, n_samples: int) -> (np.ndarray, np.ndarray):
    # create the distributions and ground-truth labels
    Gs = []
    Ys = []
    for i in range(n_components):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(n_classes))

    # sample the dataset
    X = [G.get_sample(n_samples) for G in Gs]
    Y_ = [[Y] * n_samples for Y in Ys]

    return np.vstack(X), np.hstack(Y_)


def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh


def eval_perf_multi(Y, Y_):
    precision = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        precision.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, precision, M


def graph_data(X, Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X:       data-points
        Y_:      ground truth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points

    Returns:
        None
    """
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 100

    # draw the correctly classified data-points
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good], s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified data-points
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad], s=sizes[bad], marker='s', edgecolors='black')


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """
    Creates a surface plot (visualize with plt.show)

        fun    ... decizijska funkcija (Nx2)->(Nx1)
        rect   ... željena domena prikaza zadana kao:
             ([x_min,y_min], [x_max,y_max])
        offset ... "nulta" vrijednost decizijske funkcije na koju 
             je potrebno poravnati središte palete boja;
             tipično imamo:
             offset = 0.5 za probabilističke modele 
                (npr. logistička regresija)
             offset = 0 za modele koji ne spljošćuju
                klasifikacijske mjere (npr. SVM)
        width,height ... rezolucija koordinatne mreže
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])
