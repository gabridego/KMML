import time

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from kernels import *


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
        if self.verbose:
            print("Start computing kernel similarity matrix...")
            start = time.time()
        K = self.K.similarity_matrix()
        if self.verbose:
            end = time.time()
            print(f"Kernel similarity matrix computed in {end - start:.2f} seconds")

        # get second term of KRR
        diag = np.zeros_like(K)
        np.fill_diagonal(diag, self.C * len(X))
        # compute coefficients for each class, one-vs-all
        # @ is matrix multiplication, equivalent to np.matmul
        self.alpha = []
        for c in sorted(set(y)):
            if self.verbose:
                print(f"Start computing alpha for class {c}...")
                start = time.time()
            self.alpha.append(np.linalg.inv(K + diag) @ Y[:,c])
            if self.verbose:
                end = time.time()
                print(f"alpha for class {c} computed in {end - start:.2f} seconds")
        self.alpha = np.array(self.alpha)
        return self

    def predict(self, X):
        preds = []
        for i, x in enumerate(X):
            if self.verbose:
                print(f"Start computing similarity for sample {i}...")
                start = time.time()
            similarity = self.K.similarity(x)
            if self.verbose:
                end = time.time()
                print(f"Similarity for sample {i} computed in {end - start:.2f} seconds")

            if self.verbose:
                print(f"Start computing prediction for sample {i}...")
                start = time.time()
            preds.append(np.argmax([np.dot(alpha, similarity) for alpha in self.alpha]))
            if self.verbose:
                end = time.time()
                print(f"Prediction for sample {i} computed in {end - start:.2f} seconds")
        return np.array(preds)
