# -*- coding: utf-8 -*-
"""
Created on Oct 20 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 1.2 of CSE 546 HW2

"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import os
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import DML.regression.ridge_utils as ri
import DML.regression.regression_utils as ru

# Define constants
#best_lambda = 1.0e2
seed = 42
kwargs = {}
k = 10000
Nclass = 10 # classes are 0 - 9
nn_cache = "../Data/hw2-data/MNIST_nn_cache.npz"

# If I haven't already done it, load MNIST training data and transform it
if not os.path.exists(nn_cache):
    print("Cache does not exist, running MNIST neural network transformation...")

    # Load in MNIST data
    print("Loading MNIST Training data...")
    X_train, y_train = mu.load_mnist(dataset='training')

    # Transform X
    print("Performing naive neural network layer transformation on training set...")
    # Alter X data and generate v for use for testing set
    X_train, v = ru.naive_nn_layer(X_train, k=k)

    # Load in MNIST data
    print("Loading MNIST Testing data...")
    X_test, y_test = mu.load_mnist(dataset='testing')

    # Transform X
    print("Performing naive neural network layer transformation on testing set...")
    X_test = ru.naive_nn_layer(X_test, k=k, v=v)

    # Now save transformation matrix for later
    print("Caching...")
    np.savez(nn_cache,v=v)
# No cache, load transformation matrix
else:
    print("Reading from cache...")
    res = np.load(nn_cache)
    v = res["v"]

    # Load in MNIST data
    print("Loading MNIST Training data...")
    X_train, y_train = mu.load_mnist(dataset='training')

    # Transform X
    print("Performing naive neural network layer transformation on training set...")
    X_train = ru.naive_nn_layer(X_train, k=k, v=v)

    # Load in MNIST data
    print("Loading MNIST Testing data...")
    X_test, y_test = mu.load_mnist(dataset='testing')

    # Transform X
    print("Performing naive neural network layer transformation on testing set...")
    X_test = ru.naive_nn_layer(X_test, k=k, v=v)

# Fit for the class prediction regression coefficients on transformed training set
print("Fitting with ridge regression...")
best_lambda = val.estimate_lambda(X_train, scale=1.0)
print("Best lambda: %.3lf." % best_lambda)
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()
w0, w = ri.fit_ridge(X_train, y_train_true, lam=best_lambda)

# Using fit on training set, predict labels for train, test data by selecting whichever
# prediction is the largest (one vs all classification)
y_hat_train = cu.multi_linear_classifier(X_train, w, w0)
y_hat_test = cu.multi_linear_classifier(X_test, w, w0)

# Compute 01 Loss!
print("Training 01 Loss:",val.loss_01(y_train,y_hat_train))
print("Testing 01 Loss:",val.loss_01(y_test,y_hat_test))

# Compute square loss!
print("Training Square Loss:",val.square_loss(y_train,y_hat_train))
print("Testing Square Loss:",val.square_loss(y_test,y_hat_test))
