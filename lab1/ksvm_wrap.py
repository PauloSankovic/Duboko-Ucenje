from numpy.testing._private.parameterized import param
from sklearn.svm import SVC
from data import *


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c: int, param_svm_gamma: str):
        """
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre
        """
        self.classifier = SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.classifier.fit(X, Y_)

    def predict(self, X):
        """
        Predviđa i vraća indekse razreda podataka X
        """
        return self.classifier.predict(X)

    def get_scores(self, X):
        """
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.
        """
        return self.classifier.decision_function(X)

    def support(self):
        """
        Indeksi podataka koji su odabrani za potporne vektore
        """
        # get indices of support vectors
        return self.classifier.support_


def zadatak_2():
    X, Y_ = sample_gmm_2d(5, 2, 100)
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')
    accuracy, precision, M = eval_perf_multi(Y_, model.predict(X))
    print("Točnost:", accuracy)


def zadatak_3():
    X, Y_ = sample_gmm_2d(6, 2, 10)
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')
    Y = model.predict(X)

    accuracy, precision, M = eval_perf_multi(Y_, Y)
    print("Točnost:", accuracy)

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(model.predict, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y, special=model.support())
    plt.show()


if __name__ == '__main__':
    np.random.seed(100)
    zadatak_3()
