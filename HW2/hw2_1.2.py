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

# Flags to control functionality
find_best_lam = False

# Define constants
best_lambda = 1.0e6
best_thresh = 0.4
seed = 1
frac = 0.1
num = 5
kwargs = {}
lammax = 1.0e7
k = 3000
scale = 10.0
Nclass = 10 # classes are 0 - 9
nn_cache = "/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw2-data/MNIST_nn_cache.npz"

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

    print("Caching...")
    np.savez(nn_cache,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
             v=v)
else:
    print("Reading from cache...")
    res = np.load(nn_cache)
    X_train = res["X_train"]
    X_test = res["X_test"]
    y_train = res["y_train"]
    y_test = res["y_test"]
    v = res["v"]

# Fit!
# Fit for the class prediction regression coefficients
w0, w = cu.multi_linear_classifier_fit(X_train, y_train, Nclass, lam=best_lambda, thresh=best_thresh)

# Using fit on training set, predict labels for train, test data by selecting whichever
# prediction is the largest (one vs all classification)
y_hat_train = cu.multi_linear_classifier(X_train, w, w0)
y_hat_test = cu.multi_linear_classifier(X_test, w, w0)

# Compute 01 Loss!
print("Training 01 Loss:",val.loss_01(y_train,y_hat_train)/len(y_hat_train))
print("Testing 01 Loss:",val.loss_01(y_test,y_hat_test)/len(y_hat_test))

# Compute square loss!
print("Training Square Loss:",val.square_loss(y_train,y_hat_train)/len(y_hat_train))
print("Testing Square Loss:",val.square_loss(y_test,y_hat_test)/len(y_hat_test))