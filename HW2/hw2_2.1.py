# -*- coding: utf-8 -*-
"""
Created on Mon Oct  19

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.1 of CSE 546 HW2
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

# Flags to control functionality
find_best_lam = False

# Best threshold from previous running of script
#
# From a previous grid search on training data:

# Define constants
best_lambda = 1.0#0.585276634659
best_thresh = 0.5
eta = 1.0e-5

seed = 42
frac = 0.1
num = 10
lammax = 10.0
scale = 1.5

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
X_train = X_train - np.mean(X_train, axis=0)

y_train_true = mu.mnist_filter(y_train, filterby=2)
print("True number of twos in training set:",np.sum(y_train_true))

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = mu.mnist_filter(y_test, filterby=2)
X_test = X_test - np.mean(X_test, axis=0)

# Build regularized binary classifier by minimizing log log
# Will need to optimize over lambda via a regularization path
if find_best_lam:
    print("Running regularization path to optmize lambda...")

    # Split training data into subtraining set, validation set
    X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

    # Filter y values to 0, 1 labels
    y_tr_true = mu.mnist_filter(y_tr, filterby=2)
    y_val_true = mu.mnist_filter(y_val, filterby=2)

    error_val, error_train, lams = \
    val.binlogistic_reg_path(X_tr, y_tr_true, X_val, y_val_true, model=cu.logistic_model,
    lammax=lammax,scale=scale, num=num, error_func=val.loss_01, thresh=best_thresh, best_w=False,
    eta=eta, adaptive=False, lossfn=val.logloss, saveloss=False)

    # Find minimum threshold, lambda from minimum validation error
    ind = np.nanargmin(error_val)
    best_lambda = lams[ind]
    print("Best lambda:",best_lambda)

    plt.plot(lams,error_val,color="blue",lw=3,label="Val")
    plt.plot(lams,error_train,color="green",lw=3,label="Train")
    plt.legend()
    plt.semilogx()
    plt.show()

# With a best fit lambda, threshold, refit
w0, w, loss_train, loss_test, iter_train = gd.gradient_ascent(cu.logistic_model, X_train, y_train_true,
                                                    lam=best_lambda, eta=eta, saveloss=True,
                                                    X_test=X_test, y_test=y_test_true,
                                                    adaptive=True, eps=1.0e-10)

plt.plot(iter_train, loss_train, color="green")
plt.plot(iter_train, loss_test, color="blue")
plt.show()