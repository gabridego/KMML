import time

from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from kernels import *


class KernelRidgeRegressor(BaseEstimator):

    def __init__(self, C=1.0, kernel='rbf', gamma=10):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        # initialize kernel
        self.K = kernels[self.kernel](X, self.gamma)
        print("Start computing kernel similarity matrix...")
        start = time.time()
        K = self.K.similarity_matrix()
        end = time.time()
        print(f"Kernel similarity matrix computed in {end - start:.2f} seconds")

        # compute first term
        diag = np.zeros_like(K)
        np.fill_diagonal(diag, self.C * len(X))
        K += diag
        self.alpha = solve(K, y, assume_a='pos')
        return self

    def predict(self, X):
        print("Predicting...")
        preds = []
        for x in tqdm(X):
            similarity = self.K.similarity(x)
            preds.append(np.dot(self.alpha, similarity))
        return np.array(preds)


class KernelRidgeClassifier(BaseEstimator):

    def __init__(self, C=1.0, kernel='rbf', gamma=10, verbose=False):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.verbose = verbose
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        # map labels in {-1, 1}
        Y = LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(y)
        # initialize kernel
        self.K = kernels[self.kernel](X, self.gamma)
        print("Start computing kernel similarity matrix...")
        start = time.time()
        K = self.K.similarity_matrix()
        end = time.time()
        print(f"Kernel similarity matrix computed in {end - start:.2f} seconds")

        # compute first term
        diag = np.zeros_like(K)
        np.fill_diagonal(diag, self.C * len(X))
        K += diag
        # compute coefficients for each class, one-vs-all
        self.alpha = []
        for c in tqdm(sorted(set(y))):
            self.alpha.append(solve(K, Y[:, c], assume_a='pos'))
        self.alpha = np.array(self.alpha)
        return self

    def predict(self, X):
        print("Predicting...")
        preds = []
        for x in tqdm(X):
            similarity = self.K.similarity(x)
            preds.append(np.argmax([np.dot(alpha, similarity) for alpha in self.alpha]))
        return np.array(preds)
