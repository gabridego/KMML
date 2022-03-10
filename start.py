# Kernel Methods for Machine Learning
# Gabriele Degola, Marco Fioretti - Degola Fioretti
# MoSIG DSAI, MSIAM 2021/22
# Grenoble INP - Ensimag

import argparse
import time

import pandas as pd

from models import KernelRidgeClassifier
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(prog='start',
                                     description="Kernel methods for ML")

    parser.add_argument('--xtr', type=str, help="path to training sample")
    parser.add_argument('--ytr', type=str, help="path to training labels")
    parser.add_argument('--xte', type=str, help="path to test sample")
    parser.add_argument('--yte', type=str, default='Yte.csv', help="path to store CSV file with predictions")

    parser.add_argument('--c', type=float, default=0.00001, help="regularization parameter")
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'laplacian', 'exp'],
                        default='rbf', help="kernel function")
    parser.add_argument('--gamma', type=float, default=1, help="gamma coefficient for kernel")

    return parser.parse_args()


if __name__ == '__main__':
    # random seed is not set, predictions may slightly vary
    # np.random.seed(42)
    args = parse_args()

    # load data
    Xtr = np.array(pd.read_csv(args.xtr, header=None, sep=',', usecols=range(3072)))
    Xte = np.array(pd.read_csv(args.xte, header=None, sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv(args.ytr, sep=',', usecols=[1])).squeeze()
    print(f"data loaded:\n\t{len(Xtr)} training images\n\t{len(Ytr)} training labels\n\t{len(Xte)} test images")

    # data augmentation
    X, y = augment_dataset(Xtr, Ytr, flip_ratio=1, rot_ratio=1, rot_replicas=1, rot_angle=30)
    print(f"after data augmentation:\n\t{len(X)} training images\n\t{len(y)} training labels")

    # feature extraction
    hog_extractor = HOGExtractor()
    train_hogs = hog_extractor.fit_transform(X, y)
    print("HOGs extracted from training images")
    hog_test = hog_extractor.transform(Xte)
    print("HOGs extracted from test images")

    # train classifier
    clf = KernelRidgeClassifier(kernel=args.kernel, C=args.c, gamma=args.gamma)
    start = time.time()
    clf.fit(train_hogs, y)
    end = time.time()
    print(f"fit completed in {end - start:2f} seconds")

    # perform predictions
    start = time.time()
    Yte = clf.predict(hog_test)
    end = time.time()
    print(f"predict completed in {end - start:.2f} seconds")

    # export predictions
    Yte = {'Prediction': Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(args.yte, index_label='Id')
