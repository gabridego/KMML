from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from kernels import *


class KernelRidgeClassifier(BaseEstimator):

    def __init__(self, C=1.0, kernel='rbf', sigma=10):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        # map labels in {-1, 1}
        Y = LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(y)
        # initialize kernel
        self.K = kernels[self.kernel](X, self.sigma)
        K = self.K.similarity_matrix()
        # get second term of KRR
        diag = np.zeros_like(K)
        np.fill_diagonal(diag, self.C * len(X))
        # compute coefficients for each class, one-vs-all
        # @ is matrix multiplication, equivalent to np.matmul
        self.alpha = []
        for c in sorted(set(y)):
            self.alpha.append(np.linalg.inv(K + diag) @ Y[:,c])
        self.alpha = np.array(self.alpha)
        return self

    def predict(self, X):
        preds = []
        for x in X:
            similarity = self.K.similarity(x)
            preds.append(np.argmax([np.sum([alpha * similarity for a in alpha]) for alpha in self.alpha]))
        return np.array(preds)
