import numpy as np


class Kernel:

    def __init__(self, X, sigma):
        self.X = X
        self.sigma = sigma


class LinearKernel(Kernel):

    def __init__(self, X, sigma=None):
        super().__init__(X, sigma)

    def similarity_matrix(self):
        l = len(self.X)
        K = np.empty([l, l])
        for i in range(l):
            for j in range(i, l):
                K[i, j] = K[j, i] = np.dot(self.X[i], self.X[j])
        return K

    def similarity(self, x):
        return np.array([np.dot(x_i, x) for x_i in self.X])


class GaussianKernel(Kernel):

    def __init__(self, X, sigma):
        super().__init__(X, sigma)

    def similarity_matrix(self):
        l = len(self.X)
        K = np.empty([l, ll])
        for i in range(ll):
            for j in range(i, ll):
                K[i, j] = K[j, i] = np.exp(-np.linalg.norm(self.X[i] - self.X[j]) ** 2 / (2 * self.sigma ** 2))
        return K

    def similarity(self, x):
        return np.array([np.exp(-np.linalg.norm(x_i - x) ** 2 / (2 * self.sigma ** 2)) for x_i in self.X])


kernels = {'linear': LinearKernel, 'rbf': GaussianKernel}
