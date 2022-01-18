import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog


class StandardScaler:

    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, X, y=None):
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

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    
class HOGExtractor:
    
    def __init__(self, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        new_X = []
        for x in X:
            r = x[:1024].reshape([32, 32])
            g = x[1024:2048].reshape([32, 32])
            b = x[2048:].reshape([32, 32])
            tensor = np.dstack((r, g, b))
            new_X.append(hog(tensor, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, channel_axis=-1))
        return np.array(new_X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class ColorHistogramExtractor:
    
    def __init__(self, bins=(8, 8, 8)):
        self.bins = bins
        self.range = None
    
    def fit(self, X, y=None):
        rs = X[:, :1024]
        r_lim = (np.min(rs), np.max(rs))
        gs = X[:, 1024:2048]
        g_lim = (np.min(gs), np.max(gs))
        bs = X[:, 2048:]
        b_lim = (np.min(bs), np.max(bs))
        self.range = (r_lim, g_lim, b_lim)
    
    def transform(self, X):
        new_X = []
        for x in X:
            r = x[:1024]
            g = x[1024:2048]
            b = x[2048:]
            tensor = np.dstack((r, g, b)).squeeze()
            
            H, _ = np.histogramdd(tensor, bins=self.bins, range=self.range)
            new_X.append(H.reshape(-1))
        return np.array(new_X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
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
