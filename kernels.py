import numpy as np


class Kernel:

    def __init__(self, X, gamma):
        self.X = X
        self.gamma = gamma


class LinearKernel(Kernel):

    def __init__(self, X, gamma=None):
        super().__init__(X, gamma)

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

    def __init__(self, X, gamma):
        super().__init__(X, gamma)

    def similarity_matrix(self):
        l = len(self.X)
        K = np.empty([l, l])
        for i in range(l):
            for j in range(i, l):
                K[i, j] = K[j, i] = np.exp(- self.gamma * (np.linalg.norm(self.X[i] - self.X[j]) ** 2))
        return K

    def similarity(self, x):
        return np.array([np.exp(- self.gamma * (np.linalg.norm(x_i - x) ** 2)) for x_i in self.X])


kernels = {'linear': LinearKernel, 'rbf': GaussianKernel}
