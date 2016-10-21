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
k = 10000
scale = 10.0
Nclass = 10 # classes are 0 - 9
nn_cache = "/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw2-data/MNIST_nn_cache"

# If I haven't already done it, load MNIST training data and transform it
if not os.path.exists(nn_cache):
    print("Cache does not exist, running MNIST neural network transformation...")

    # Load in MNIST data
    print("Loading MNIST Training data...")
    nn_X_train, nn_y_train = mu.load_mnist(dataset='training')

    # Transform X
    print("Performing naive neural network layer transformation...")
    nn_X_train = ru.naive_nn_layer(nn_X_train, k=k)

    print("Caching...")
    np.savez(nn_cache,nn_X_train=nn_X_train)
else:
    print("Reading from cache...")
    res = np.load(nn_cache)
    nn_X_train = res["nn_X_train"]