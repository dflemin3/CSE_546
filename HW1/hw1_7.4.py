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
import matplotlib.pyplot as plt
import os

# Flags to control functionality

# Fit (including regularization path!) the Yelp upvote data?
fit_upvotes = True

# Fit (including regularization path!) the Yelp review star data?
fit_stars = False

if fit_upvotes:

    # Define constants for splitting up data
    test_frac = 1./6 # Fraction of total data to split into testing
    val_frac = 0.2 # Fraction of remaning training data to split up
    seed = 1 # RNG seed for reproducibility
    kwargs = {"sparse" : True}
    num = 20 # Length of reg path
    cache = "../Data/hw1-data/star_cache.npz"

    if not os.path.exists(cache):
        print("Cache does not exist, running analysis...")

        # Load a text file of integers:
        y = np.loadtxt("../Data/hw1-data/star_labels.txt", dtype=np.int)
        y = y.reshape(len(y),1)

        # Load a text file of feature names:
        featureNames = open("../Data/hw1-data/star_features.txt").read().splitlines()

        # Load a csv of floats as a sparse matrix:
        #X = sp.csc_matrix(np.genfromtxt("../Data/hw1-data/upvote_data.csv", delimiter=","))
        X = io.mmread("../Data/hw1-data/star_data.mtx").tocsc()

        # Split into training set, testing set
        X_train, y_train, X_test, y_test = val.split_data(X, y, frac=test_frac, seed=seed)

        # Now split training set into training set, validation set
        X_train, y_train, X_val, y_val = val.split_data(X_train, y_train, frac=val_frac,
                                                        seed=seed)

        print("Train shapes:",X_train.shape,y_train.shape)
        print("Val shapes:",X_val.shape,y_val.shape)
        print("Test shapes:",X_test.shape,y_test.shape)

        # Set maximum lambda, minimum lambda
        lammax = lu.compute_max_lambda(X_train,y_train)
        print("Maximum lambda: %.3lf." % lammax)

        # Init error arrays
        err_val = np.zeros(num)
        err_train = np.zeros(num)

        # Run a regularization path
        print("Running regularization path for %d lambda bins." % num)
        err_val, err_train, lams, nonzeros = val.linear_reg_path(X_train, y_train, X_val, y_val,
                                                       lu.fit_lasso_fast, lammax=lammax, scale=1.2,
                                                       num=num, error_func = val.RMSE,
                                                       save_nonzeros=True, **kwargs)

        # Cache results
        print("Caching results...")
        np.savez(cache,err_val=err_val,err_train=err_train,lams=lams,nonzeros=nonzeros)
    else:
        print("Reading from cache...")
        res = np.load(cache)
        err_val = res["err_val"]
        err_train = res["err_train"]
        lams = res["lams"]
        nonzeros = res["nonzeros"]

    print("Plotting...")
    plt.plot(lams,err_val,color="blue",lw=3,label="Validation Error")
    plt.plot(lams,err_train,color="green",lw=3,label="Training Error")
    plt.legend()
    plt.show()