#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:28:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves questions 6.1 and 6.2 of CSE 546 HW1
"""

from __future__ import print_function, division
import numpy as np
import mnist_utils as mu
import ridge_utils as ri
import regression_utils as ru
import matplotlib as mpl
import matplotlib.pyplot as plt
import validation as val

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

def mnist_two_filter(y):
    """
    Takes in a MNIST label vector and sets all instances of 2 to 1 and
    everything else to 0 to train a binary classifier.

    Parameters
    ----------
    y : array (n x 1)

    Returns
    -------
    y_twos : array (n x 1)
        but masked!
    """
    mask = (y == 2)
    y_twos = np.zeros_like(y)
    y_twos[mask] = 1

    return y_twos
#end function


def mnist_ridge_thresh(X, y, lam = 1):
    """
    This function finds the best w (dot) x criteria for labeling a MNIST
    digit as a 2.  For the purposes of this classification, 2 -> 1, while
    everything else -> 0.  Computed simply by taking median of predictions
    corresponding to indicies of 1's in truth vector (only use on training data!)

	DEPRECATED -- DO NOT USE

    Parameters
    ----------
    X : array (n x d)
        Data array (n observations, d features)
    w : array (d x 1)
        feature weight array
    lam : float
        regularization constant

    Returns
    -------
    thresh_best : float
        optimal threshold for picking out 2s
    sl_best : float
        minimum square loss
    """

    # Fit model
    w0, w = ri.fit_ridge(X, y, lam=lam)

    # Predict
    y_hat = ru.linear_model(X, w, w0)

    # Mask where 2s occur in truth
    mask = (y == 1)

    # Take median of corresponding predicted values
    thresh_best = np.median(y_hat[mask])

    return thresh_best
# end function


def mnist_ridge_lam(X, y, num = 20, minlam=1.0e-10, maxlam=1.0e10):
    """
    This function finds the best regularization constant lambda
    for labeling a MNIST digit as a 2.  For the purposes of this classification,
    2 -> 1, while everything else -> 0.  Optimize over lam by minimizing the
    0-1 loss.

	DEPRECATED -- DO NOT USE

    Parameters
    ----------
    X : array (n x d)
        Data array (n observations, d features)
    w : array (d x 1)
        feature weight array
    lam : float
        regularization constant
    num : int
        number of threshold gridpoints to search over
    minlam : float
        minimum lambda grid value
    maxlam : float
        maximum lambda grid value

    Returns
    -------
    lam_best : float
        optimal threshold for picking out 2s
    loss_best : float
        minimum loss
    """

    # Make array of lambdas, thresholds
    lams = np.logspace(np.log10(minlam),np.log10(maxlam),num)
    thresh = np.zeros_like(lams)
    loss = np.zeros_like(lams)

    # Loop over thresholds, evaluate model, see which is best
    for ii in range(len(lams)):
        print("Iteration, lambda:",ii,lams[ii])

        # Fit training set for model parameters
        w0, w = ri.fit_ridge(X, y, lam=lams[ii])

        # Predict
        y_hat = ru.linear_model(X, w, w0)

        # Compute threshold as median of predicted rows corresponding to twos
        mask = (y == 1)
        thresh[ii] = np.median(y_hat[mask])

        # Classify, then get square loss, 1/0 error
        y_hat = ri.ridge_bin_class(X, w, w0, thresh=thresh[ii])
        print("Predicted Number of 2s:",np.sum(y_hat))
        print("Predicted threshold:",thresh[ii])

        # Find minimum of loss to optimize lambda
        loss[ii] = ru.loss_01(y, y_hat)
        print("0-1 Loss:",loss[ii])

    # Get best threshold (min MSE on training set) and return it
    best_ind = np.argmin(loss)

    # Now plot it
    fig, ax = plt.subplots()

    ax.plot(lams,loss,"-o",lw=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"0-1 Loss")

    # Plot best fit
    plt.axvline(x=lams[best_ind], ymin=-100, ymax = 100,
                linewidth=3, color='k', ls="--")

    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig("sl_lam.pdf")

    return lams[best_ind], loss[best_ind], thresh[best_ind]

# end function


if __name__ == "__main__":

    # Flags to control functionality
    find_best_lam = False

    # Best threshold from previous running of script
    #
    # From a previous grid search on training data:

	# Define constants
    best_lambda = 1.0e4
    best_thresh = 0.4
    seed = 1
    frac = 0.1
    num = 5
    kwargs = {}
    lammax = 1.0e7
    scale = 10.0

    # Load in MNIST training data
    print("Loading MNIST Training data...")
    X_train, y_train = mu.load_mnist(dataset='training')

    y_train_true = mnist_two_filter(y_train)
    print("True number of twos in training set:",np.sum(y_train_true))

    # Perform grid search to find best regularization constant and threshold?
    if find_best_lam:
        print("Finding optimal lambda and threshold via regularization path.")

        thresh_arr = np.linspace(-0.2,1.,num)
        err_val = np.zeros((num,num))
        err_train = np.zeros((num,num))

        # Split training data into subtraining set, validation set
        X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

        # Filter y values to 0, 1 labels
        y_tr_true = mnist_two_filter(y_tr)
        y_val_true = mnist_two_filter(y_val)

        # Loop over thresholds
        for i in range(num):
            # Internally loop over lambdas in regularization path
            err_val[i,:], err_train[i,:], lams = val.linear_reg_path(X_tr, y_tr_true, X_val,
            														 y_val_true, ri.fit_ridge,
            														 lammax=lammax,
            														 scale=scale,
																	 num=num, error_func=val.loss_01,
																	 thresh=thresh_arr[i], **kwargs)

        # Find minimum threshold, lambda from minimum validation error
        ind_t,ind_l = np.unravel_index(err_val.argmin(), err_val.shape)
        best_lambda = lams[ind_l]
        best_thresh = thresh_arr[ind_t]
        print("Best lambda:",best_lambda)
        print("Best threshold:",best_thresh)

        plt.plot(lams,err_val[ind_t,:])
        plt.show()

    # Fit training set for model parameters using best fit lambda
    w0, w = ri.fit_ridge(X_train, y_train_true, lam=1)#best_lambda)

    # Predict, then get square loss, 1/0 error on training data
    y_hat_train = ru.linear_model(X_train, w, w0)
    y_hat_train_class = ri.ridge_bin_class(X_train, w, w0, thresh=best_thresh)
    sl_train = val.square_loss(y_train_true, y_hat_train)
    err_10_train = val.loss_01(y_train_true, y_hat_train_class)

    # Load testing set
    print("Loading MNIST Testing data...")
    X_test, y_test = mu.load_mnist(dataset='testing')
    y_test_true = mnist_two_filter(y_test)
    print("True number of twos in testing set:",np.sum(y_test_true))

    # Predict, then get square loss, 1/0 error on testing data
    y_hat_test = ru.linear_model(X_test, w, w0)
    y_hat_test_class = ri.ridge_bin_class(X_test, w, w0, thresh=best_thresh)
    sl_test = val.square_loss(y_test_true, y_hat_test)
    err_10_test = val.loss_01(y_test_true, y_hat_test_class)

    # Output!
    print("Square Loss on training, testing set: %.3lf, %.3lf" % (sl_train, sl_test))
    print("1/0 Loss on training, testing set: %.3lf, %.3lf" % (err_10_train, err_10_test))
