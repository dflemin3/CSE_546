# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to answer HW1 Question 7.4
"""
from __future__ import print_function, division

import validation as val
import regression_utils as ru
import lasso_utils as lu
import numpy as np
import scipy.io as io
import scipy.sparse as sp

# Flags to control functionality

# Fit (including regularization path!) the Yelp upvote data?
fit_upvotes = True

# Fit (including regularization path!) the Yelp review star data?
fit_stars = False

if fit_upvotes:

    # Define constants for splitting up data
    n_train = 4000
    n_val = 1000
    n_test = 1000

    test_frac = 1./6 # Fraction of total data to split into testing
    val_frac = 0.2 # Fraction of remaning training data to split up
    seed = 1 # RNG seed for reproducibility
    kwargs = {"sparse" : True}
    num = 5 # Length of reg path

    # TODO: something here that loads in data only if fit archive doesn't exist

    # Load a text file of integers:
    y = np.loadtxt("../Data/hw1-data/upvote_labels.txt", dtype=np.int)
    y = y.reshape(len(y),1)

    # Load a text file of feature names:
    featureNames = open("../Data/hw1-data/upvote_features.txt").read().splitlines()

    # Load a csv of floats as a sparse matrix:
    X = sp.csc_matrix(np.genfromtxt("../Data/hw1-data/upvote_data.csv", delimiter=","))

    # Split into training set, testing set
    X_train, y_train, X_test, y_test = val.split_data(X, y, frac=test_frac, seed=seed)

    # Now split training set into training set, validation set
    X_train, y_train, X_val, y_val = val.split_data(X_train, y_train, frac=val_frac, seed=seed)

    print("Train shapes:",X_train.shape,y_train.shape)
    print("Val shapes:",X_val.shape,y_val.shape)
    print("Test shapes:",X_test.shape,y_test.shape)

    # Set maximum lambda, minimum lambda
    lammin = 100.0
    lammax = lu.compute_max_lambda(X_train,y_train)
    print("Maximum lambda: %.3lf." % lammax)

    # Init error arrays
    err_val = np.zeros(num)
    err_train = np.zeros(num)

    # Run a regularization path
    print("Running regularization path for %d lambda bins." % num)
    err_val, err_train, lams = val.linear_reg_path(X_train, y_train, X_val, y_val,
												   lu.fit_lasso_fast, lammax=lammax, lammin=lammin,
												   num=num, error_func = val.RMSE, **kwargs)

    print("Done!")