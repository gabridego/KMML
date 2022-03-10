# Kernel Methods for Machine Learning

MoSIG DSAI, MSIAM 2021/22
Grenoble INP - Ensimag

## Introduction

This repository contains the code for the data challenge of the *Kernel Methods for Machine Learning* course. The task is image classification with kernel methods.

## Authors

- Gabriele Degola

Public score: 0.616
Private score: 0.602
Leaderboard position: 3

## Data

Images are drawn from the [*CIFAR-10*](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and organized as `csv` files: each row contains an image of 32 x 32 pixels.

## Run

The following command reproduces the best submission. Predictions are stored in `Yte.csv`.

```console
python start.py --xtr data/Xtr.csv --ytr data/Ytr.csv --xte data/Xte.csv
```

## Structure

Code is organized in the following files:

- [`start.py`](./start.py) is the main script;
- [`models.py`](./models.py) contains our implementation of kernel ridge regression;
- [`kernels.py`](./kernels.py) contains implemented kernels;
- [`utils.py`](./utils.py) contains functions and classes for data processing.


## Dependencies

Code relies on the following Python libraries:

- `numpy`, `scipy` for basic operations;
- `pandas` for data loading and storing;
- `tqdm` for showing progresses;
- `scikit-image` for computation of histograms of oriented gradients;
- `scikit-learn` for label binarization only, required for multi-class classification.
