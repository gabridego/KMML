import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid

from models import KernelRidgeRegressor, KernelRidgeClassifier
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Kernel methods for ML")

    parser.add_argument('--x-train', type=str, help="path to training sample")
    parser.add_argument('--y-train', type=str, help="path to training labels")
    parser.add_argument('--x-test', type=str, nargs='?', help="path to test sample")
    parser.add_argument('--y-test', type=str, nargs='?', help="path to store CSV file with predictions")

    parser.add_argument('--model', type=str, choices=['krr'], help="ML model to use")
    parser.add_argument('--c', type=float, default=1.0, help="regularization parameter")
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf'], help="kernel function")
    parser.add_argument('--gamma', type=float, default=10, help="gamma coefficient for rbf kernel")
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
    Xtr = np.array(pd.read_csv('data/Xtr.csv', header=None, sep=',', usecols=range(3072)))
    Xte = np.array(pd.read_csv('data/Xte.csv', header=None, sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv('data/Ytr.csv', sep=',', usecols=[1])).squeeze()

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    X, y = augment_dataset(Xtr, Ytr)


    X, y = make_regression(1000, n_features=500)

    reg1 = KernelRidgeRegressor(0.01/len(X), kernel='rbf', gamma=1)
    reg1.fit(X, y)
    print(reg1.K.similarity_matrix())
    print(reg1.alpha)
    y_pred1 = reg1.predict(X)

    reg2 = KernelRidge(0.01, kernel='rbf', gamma=1)
    reg2.fit(X, y)
    y_pred2 = reg2.predict(X)

    print(mean_squared_error(y, y_pred1))
    print(mean_squared_error(y, y_pred2))

    # X, y = make_classification(1000, 100, n_informative=10, n_classes=10)
    # X_train, y_train = X[:800], y[:800]
    # X_test, y_test = X[800:], y[800:]
    #
    # clf1 = KernelRidgeClassifier(0.01/len(X_train), kernel='linear')
    # clf1.fit(X_train, y_train)
    # y_pred1 = clf1.predict(X_test)
    #
    # clf2 = RidgeClassifier(0.01, fit_intercept=False)
    # clf2.fit(X_train, y_train)
    # y_pred2 = clf2.predict(X_test)
    #
    # print(accuracy_score(y_test, y_pred1))
    # print(accuracy_score(y_test, y_pred2))
    #
    # print(y_pred1)
    # print(y_pred2)
