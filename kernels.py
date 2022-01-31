import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import pdist, squareform


class Kernel:

    def __init__(self, X, gamma):
        self.X = X
        self.gamma = gamma


class LinearKernel(Kernel):

    def __init__(self, X, gamma=None):
        super().__init__(X, gamma)

    def similarity_matrix(self):
        K = self.X @ self.X.T
        return K

    def similarity(self, x):
        return x @ self.X.T


class GaussianKernel(Kernel):

    def __init__(self, X, gamma):
        super().__init__(X, gamma)

    def similarity_matrix(self):
        X_norm = np.sum(self.X ** 2, axis=-1)
        K = np.exp(-self.gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(self.X, self.X.T)))
        return K

    def similarity(self, x):
        return np.array([np.exp(- self.gamma * (np.linalg.norm(x_i - x) ** 2)) for x_i in self.X])


class LaplacianKernel(Kernel):

    def __init__(self, X, gamma):
        super().__init__(X, gamma)

    def similarity_matrix(self):
        K = squareform(np.exp(-self.gamma * pdist(self.X, 'cityblock'))) + np.eye(len(self.X))
        return K

    def similarity(self, x):
        return np.array([np.exp(- self.gamma * np.linalg.norm(x_i - x, ord=1)) for x_i in self.X])


class ExponentialKernel(Kernel):

    def __init__(self, X, gamma):
        super().__init__(X, gamma)

    def similarity_matrix(self):
        K = self.X @ self.X.T
        K = np.exp(self.gamma * (K - 1))
        return K

    def similarity(self, x):
        return np.exp(self.gamma * (x @ self.X.T - 1))


kernels = {'linear': LinearKernel, 'rbf': GaussianKernel,
           'laplacian': LaplacianKernel, 'exp': ExponentialKernel}
