import time

import cvxopt
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from kernels import *
from utils import augment_dataset, HOGExtractor

cvxopt.solvers.options['show_progress'] = False


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

    def __init__(self, C=1.0, kernel='rbf', gamma=10):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
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


class AugmentedHogsKernelRidgeClassifier(BaseEstimator):

    def __init__(self, C=1.0, kernel='rbf', gamma=10, **aug_args):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.aug_args = aug_args
        self.hog_extractor = HOGExtractor()
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        # augment dataset
        X, y = augment_dataset(X, y, **self.aug_args)
        # get HOGs
        X = self.hog_extractor.transform(X)
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
        # get hogs
        X = self.hog_extractor.transform(X)
        preds = []
        for x in tqdm(X):
            similarity = self.K.similarity(x)
            preds.append(np.argmax([np.dot(alpha, similarity) for alpha in self.alpha]))
        return np.array(preds)


class KernelSVC(BaseEstimator):
    """
    $$\min_{\alpha}\frac{1}{2}\alpha^Ty^TKy\alpha - \sum_i\alpha_i\ s.t.\ \alpha^Ty = 0,\ 0 \leq \alpha_i \leq C$$
    """

    def __init__(self, C=1.0, kernel='rbf', gamma=10):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.K = None
        self.alpha = None

    def fit(self, X, y):
        n = len(X)
        # map labels in {-1, 1}
        Y = LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(y)
        # initialize kernel
        self.K = kernels[self.kernel](X, self.gamma)
        print("Start computing kernel similarity matrix...")
        start = time.time()
        K = self.K.similarity_matrix()
        end = time.time()
        print(f"Kernel similarity matrix computed in {end - start:.2f} seconds")

        # class-independent factors
        q = cvxopt.matrix(np.ones(n) * -1)
        b = cvxopt.matrix(0.0)
        G_low = np.eye(n) * -1
        G_up = np.eye(n)
        G = cvxopt.matrix(np.vstack((G_low, G_up)))
        h_low = np.zeros(n)
        h_up = np.ones(n) * self.C
        h = cvxopt.matrix(np.hstack((h_low, h_up)))
        # compute coefficients for each class, one-vs-all
        self.alpha = []
        for c in tqdm(sorted(set(y))):
            P = cvxopt.matrix(np.outer(Y[:, c], Y[:, c]) * K)
            A = cvxopt.matrix(Y[:, c], (1, n), tc='d')
            result = cvxopt.solvers.qp(P, q, G, h, A, b)
            self.alpha.append(np.ravel(result['x']) * Y[:, c])
        self.alpha = np.array(self.alpha)
        return self

    def predict(self, X):
        print("Predicting...")
        preds = []
        for x in tqdm(X):
            similarity = self.K.similarity(x)
            preds.append(np.argmax([np.dot(alpha, similarity) for alpha in self.alpha]))
        return np.array(preds)
