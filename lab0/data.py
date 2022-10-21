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


def sample_gauss_2d(C: int, N: int) -> (np.ndarray, np.ndarray):
    Gs = [Random2DGaussian() for _ in range(C)]

    # X.shape = (C * N, 2)
    X = [G.get_sample(N) for G in Gs]

    # Y.shape = (C * N, 1)
    Y = [[Y] * N for Y in range(C)]

    return np.vstack(X), np.hstack(Y)


def calculate_recall(tp: int, fn: int) -> float:
    """
    Number of true positives / number of positive class members (Y == 1)
    """
    return tp / (tp + fn)


def calculate_precision(tp: int, fp: int) -> float:
    """
    Number of true positives / number of positive predictions 
    """
    return tp / (tp + fp)


def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Number of correct predictions / total number of predictions
    """
    return (tp + tn) / (tp + fn + tn + fp)


def eval_perf_binary(Y: np.ndarray, Y_: np.ndarray) -> (float, float, float):
    tp = sum(np.logical_and(Y == Y_, Y_))
    fn = sum(np.logical_and(Y != Y_, Y_))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))

    accuracy = calculate_accuracy(tp, tn, fp, fn)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)

    return accuracy, precision, recall


def eval_ap(ranked_labels):
    def precision(i: int) -> float:
        labels = ranked_labels[i:]

        tp = sum(labels)
        fp = len(labels) - tp

        return calculate_precision(tp, fp)

    return sum(precision(i) * ranked_labels[i] for i in range(len(ranked_labels))) / sum(ranked_labels)


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
    sizes[special] = 40

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
