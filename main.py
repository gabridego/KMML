import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid


def parse_args():
    parser = argparse.ArgumentParser(description="Kernel methods for ML")

    parser.add_argument('--x-train', type=str, help="path to training sample")
    parser.add_argument('--y-train', type=str, help="path to training labels")
    parser.add_argument('--x-test', type=str, nargs='?', help="path to test sample")
    parser.add_argument('--y-test', type=str, nargs='?', help="path to store CSV file with predictions")

    parser.add_argument('--model', type=str, choices=['krr'], help="ML model to use")
    parser.add_argument('--c', type=float, default=1.0, help="regularization parameter")
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf'], help="kernel function")
    parser.add_argument('--sigma', type=float, default=10, help="sigma coefficient for rbf kernel")
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='print messages')
    parser.add_argument('--use-cv', default=False, action='store_true', help="cross validation")


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
