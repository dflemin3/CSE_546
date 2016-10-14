# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to answer HW1 Question 7.4 for yelp star data
"""
from __future__ import print_function, division

import validation as val
import regression_utils as ru
import lasso_utils as lu
import numpy as np
import scipy.io as io
import scipy.sparse as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

# Flags to control functionality

# Fit (including regularization path!) the Yelp review upvote data?
fit_upvotes = True

# Fit (including regularization path!) the Yelp review star data?
fit_stars = True

# Do you want to see the plots?
make_plots = False

# Do you want to save the plots?
save_plots = False

seed = 42

###############################
#
# Star section
#
###############################

if fit_stars:
    print("Yelp review star analysis.")

    # Define constants for splitting up data
    test_frac = 1./6 # Fraction of total data to split into testing
    val_frac = 0.2 # Fraction of remaning training data to split up
    kwargs = {"sparse" : True}
    num = 30 # Length of reg path
    cache = "../Data/hw1-data/star_cache.npz"
    w_cache = "../Data/hw1-data/star_w_cache.npz"

    # Run!

    print("Loading data...")
    # Load a text file of integers:
    y = np.loadtxt("../Data/hw1-data/star_labels.txt", dtype=np.int)
    y = y.reshape(len(y),1)

    # Load a text file of feature names:
    featureNames = open("../Data/hw1-data/star_features.txt").read().splitlines()

    # Load a csv of floats as a sparse matrix:
    X = io.mmread("../Data/hw1-data/star_data.mtx").tocsc()

    # Split into training set, testing set
    X_train, y_train, X_test, y_test = val.split_data(X, y, frac=test_frac, seed=seed)

    # Now split training set into training set, validation set
    X_train, y_train, X_val, y_val = val.split_data(X_train, y_train, frac=val_frac,
                                                    seed=seed)

    print("Train shapes:",X_train.shape,y_train.shape)
    print("Val shapes:",X_val.shape,y_val.shape)
    print("Test shapes:",X_test.shape,y_test.shape)

    # Run analysis if answer cache doesn't exist
    if not os.path.exists(cache):
        print("Cache does not exist, running analysis...")

        # Set maximum lambda, minimum lambda
        lammax = lu.compute_max_lambda(X_train,y_train)
        print("Maximum lambda: %.3lf." % lammax)

        # Init error arrays
        err_val = np.zeros(num)
        err_train = np.zeros(num)

        # Run a regularization path
        print("Running regularization path for %d lambda bins." % num)
        err_val, err_train, lams, nonzeros = val.linear_reg_path(X_train, y_train, X_val, y_val,
                                                       lu.fit_lasso_fast, lammax=lammax, scale=1.15,
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

    # Find best lambda according to validation error
    # and use that to refit training data, test on test data
    best_ind = np.argmin(err_val)
    best_lam = lams[best_ind]
    print("Best lambda: %.3lf" % best_lam)
    print("Best validation RMSE: %.3lf" % err_val[best_ind])

    # Fitting done, now plot errors, nonzeros vs lambda
    if make_plots:
        print("Plotting...")

        # Errors as a function of lambda
        fig, ax = plt.subplots()

        ax.plot(lams,err_val,"o-",color="blue",lw=3,label="Validation Error")
        ax.plot(lams,err_train,"o-",color="green",lw=3,label="Training Error")
        ax.axvline(x=best_lam, ymin=-10, ymax =10, linewidth=3, color='k',
                    ls="--",label=r"Best $\lambda$")

        # Format
        ax.set_ylabel("RMSE")
        ax.set_xlabel(r"Regularization Constant $\lambda$")
        ax.legend(loc="lower right")

        if save_plots:
            fig.savefig("star_rmse.pdf")

        plt.show()

        # Nonzeros as a function of lambda
        fig, ax = plt.subplots()

        ax.plot(lams,nonzeros,"o-",color="blue",lw=3)
        ax.axvline(x=best_lam, ymin=-10, ymax =10, linewidth=3, color='k',
                    ls="--",label=r"Best $\lambda$")

        # Format
        ax.set_ylabel("Nonzeros")
        ax.set_xlabel(r"Regularization Constant $\lambda$")

        if save_plots:
            fig.savefig("star_nonzeros.pdf")

        plt.show()

    # Refit training data with optimal lambda if we haven't already
    if not os.path.exists(w_cache):
        print("Refitting training data with best lambda, testing on testing data...")
        w_0, w = lu.fit_lasso_fast(X_train, y_train,lam=best_lam)

        # Now save it
        np.savez(w_cache,w_0=w_0,w=w)
    else:
        print("Loading training set fit from cache...")
        res = np.load(w_cache)
        w_0 = res["w_0"]
        w = res["w"]

    r2_train = ru.r_squared(X_train, y_train, w, w_0, sparse=True)
    print("r^2 on the training set: %.3lf" % r2_train)
    r2_val = ru.r_squared(X_val, y_val, w, w_0, sparse=True)
    print("r^2 on the validation set: %.3lf" % r2_val)

    # Compute error on testing set
    y_hat_test = ru.linear_model(X_test, w, w_0, sparse=True)
    RMSE_test = val.RMSE(y_test, y_hat_test)
    r2_test = ru.r_squared(X_test, y_test, w, w_0, sparse=True)
    print("RMSE on the testing set: %.2lf" % RMSE_test)
    print("r^2 on the testing set: %.3lf" % r2_test)

    # Inspect solution and output top 10 weights in magnitude and their corresponding name
    sorted_w_args = np.array(np.fabs(w).flatten()).argsort()[::-1][:20]
    for i in range(20):
        print("Feature: %s, weight: %.2lf" % (featureNames[sorted_w_args[i]],
                                                w[sorted_w_args[i]]))

###############################
#
# Upvote section
#
###############################


if fit_upvotes:
    print("Yelp upvotes analysis.")

    # Define constants for splitting up data
    test_frac = 1./6 # Fraction of total data to split into testing
    val_frac = 0.2 # Fraction of remaning training data to split up
    kwargs = {"sparse" : True}
    num = 30 # Length of reg path
    cache = "../Data/hw1-data/upvote_cache.npz"
    w_cache = "../Data/hw1-data/upvote_w_cache.npz"

    # Run!

    print("Loading data...")
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
    X_train, y_train, X_val, y_val = val.split_data(X_train, y_train, frac=val_frac,
                                                    seed=seed)

    print("Train shapes:",X_train.shape,y_train.shape)
    print("Val shapes:",X_val.shape,y_val.shape)
    print("Test shapes:",X_test.shape,y_test.shape)

    # Run analysis if answer cache doesn't exist
    if not os.path.exists(cache):
        print("Cache does not exist, running analysis...")

        # Set maximum lambda, minimum lambda
        lammax = lu.compute_max_lambda(X_train,y_train)
        print("Maximum lambda: %.3lf." % lammax)

        # Init error arrays
        err_val = np.zeros(num)
        err_train = np.zeros(num)

        # Run a regularization path
        print("Running regularization path for %d lambda bins." % num)
        err_val, err_train, lams, nonzeros = val.linear_reg_path(X_train, y_train, X_val, y_val,
                                                       lu.fit_lasso_fast, lammax=lammax, scale=1.15,
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

    # Find best lambda according to validation error
    # and use that to refit training data, test on test data
    best_ind = np.argmin(err_val)
    best_lam = lams[best_ind]
    print("Best lambda: %.3lf" % best_lam)
    print("Best validation RMSE: %.3lf" % err_val[best_ind])

    # Fitting done, now plot errors, nonzeros vs lambda
    if make_plots:
        print("Plotting...")

        # Errors as a function of lambda
        fig, ax = plt.subplots()

        ax.plot(lams,err_val,"o-",color="blue",lw=3,label="Validation Error")
        ax.plot(lams,err_train,"o-",color="green",lw=3,label="Training Error")
        ax.axvline(x=best_lam, ymin=-10, ymax =10, linewidth=3, color='k',
                    ls="--",label=r"Best $\lambda$")

        # Format
        ax.set_ylabel("RMSE")
        ax.set_xlabel(r"Regularization Constant $\lambda$")
        ax.legend(loc="lower right")

        if save_plots:
            fig.savefig("upvote_rmse.pdf")

        plt.show()

        # Nonzeros as a function of lambda
        fig, ax = plt.subplots()

        ax.plot(lams,nonzeros,"o-",color="blue",lw=3)
        ax.axvline(x=best_lam, ymin=-10, ymax =10, linewidth=3, color='k',
                    ls="--",label=r"Best $\lambda$")

        # Format
        ax.set_ylabel("Nonzeros")
        ax.set_xlabel(r"Regularization Constant $\lambda$")

        if save_plots:
            fig.savefig("upvote_nonzeros.pdf")

        plt.show()

    # Refit training data with optimal lambda if we haven't already
    if not os.path.exists(w_cache):
        print("Refitting training data with best lambda, testing on testing data...")
        w_0, w = lu.fit_lasso_fast(X_train, y_train,lam=best_lam)

        # Now save it
        np.savez(w_cache,w_0=w_0,w=w)
    else:
        print("Loading training set fit from cache...")
        res = np.load(w_cache)
        w_0 = res["w_0"]
        w = res["w"]

    r2_train = ru.r_squared(X_train, y_train, w, w_0, sparse=True)
    print("r^2 on the training set: %.3lf" % r2_train)
    r2_val = ru.r_squared(X_val, y_val, w, w_0, sparse=True)
    print("r^2 on the validation set: %.3lf" % r2_val)

    # Compute error on testing set
    y_hat_test = ru.linear_model(X_test, w, w_0, sparse=True)
    RMSE_test = val.RMSE(y_test, y_hat_test)
    r2_test = ru.r_squared(X_test, y_test, w, w_0, sparse=True)
    print("RMSE on the testing set: %.2lf" % RMSE_test)
    print("r^2 on the testing set: %.3lf" % r2_test)

    # Inspect solution and output top 10 weights in magnitude and their corresponding name
    sorted_w_args = np.array(np.fabs(w).flatten()).argsort()[::-1][:20]
    for i in range(20):
        print("Feature: %s, weight: %.2lf" % (featureNames[sorted_w_args[i]],
                                                w[sorted_w_args[i]]))

print("Done!")
