import numpy as np
import matplotlib.pyplot as plt


class StandardScaler:

    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, X):
        for f in range(X.shape[-1]):
            self.means.append(np.mean(X[:, f]))
            self.stds.append(np.std(X[:, f]))

    def transform(self, X):
        if X.ndim == 1:
            raise ValueError(f"Expected 2D array, got 1D array instead:\narray={X}.\n"
                             "Reshape your data either using array.reshape(-1, 1) if "
                             "your data has a single feature or array.reshape(1, -1) "
                             "if it contains a single sample.")
        new_X = np.empty_like(X)
        for f in range(X.shape[-1]):
            new_X[:, f] = (X[:, f] - self.means[f]) / self.stds[f]
        return new_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def scale(A):
    return (A - np.min(A)) / (np.max(A) - np.min(A))


def show_image(img, label=None):
    if label:
        print(label)
    # img = scale(img)
    r = img[:1024].reshape([32, 32])
    g = img[1024:2048].reshape([32, 32])
    b = img[2048:].reshape([32, 32])
    img = np.dstack((r, g, b))
    plt.imshow(scale(img))
    plt.show()
