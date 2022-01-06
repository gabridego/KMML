import numpy as np
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
        # K = squareform(np.exp(- self.gamma * pdist(self.X, 'sqeuclidean')))
        # K += np.eye(K.shape[0])
        X_norm = np.sum(self.X ** 2, axis=-1)
        K = np.exp(-self.gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(self.X, self.X.T)))
        return K

    def similarity(self, x):
        return np.array([np.exp(- self.gamma * (np.linalg.norm(x_i - x) ** 2)) for x_i in self.X])


kernels = {'linear': LinearKernel, 'rbf': GaussianKernel}
