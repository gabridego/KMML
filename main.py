import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid


def load_data(X_train, y_train, X_test):
    Xtr = np.array(pd.read_csv(X_train, header=None, sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv(y_train, sep=',', usecols=[1])).squeeze()
    Xte = np.array(pd.read_csv(X_test, header=None, sep=',', usecols=range(3072)))

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return Xtr, Ytr, Xte, labels


def train():
    pass


if __name__ == '__main__':
    pass
