import numpy as np
import matplotlib.pyplot as plt


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
