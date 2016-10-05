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
    
    # Make where 2s occur in truth
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
        
        # Find minimum of loss to optimize threshold
        loss[ii] = ru.loss_01(y, y_hat)
        print("0-1 Loss:",loss[ii])
    
    # Get best threshold (min MSE on training set) and return it
    best_ind = np.argmin(loss)
    
    # Now plot it
    fig, ax = plt.subplots()
    
    ax.plot(lams,loss,"-o",lw=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"0-1 Loss")
    
    # Plot best fit and asymptotes
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

    # Best lambda: 7.847600e+07
    # Best Square Loss: 3037.000
    # Best wx: 0.589
    
    best_lambda = 1#7.847600e+07
    best_thresh = 0.589
    
    # Load in MNIST training data
    print("Loading MNIST Training data...")
    X_train, y_train = mu.load_mnist(dataset='training')
    y_train_true = mnist_two_filter(y_train)
    print("True number of twos in training set:",np.sum(y_train_true))
        
    # Perform grid search to find best regularization constant?
    if find_best_lam:
        print("Finding optimal lambda and threshold...")
        # Find the best lambda based on 0/1 training error
        best_lambda, best_sl, best_thresh = mnist_ridge_lam(X_train, y_train_true)
        print("Best lambda: %e" % best_lambda)
        print("Best Square Loss: %.3lf" % best_sl)
        print("Best wx: %.3lf" % best_thresh)
    else:
        best_thresh = mnist_ridge_thresh(X_train, y_train_true, lam=best_lambda)
        print("Best wx: %.3lf" % best_thresh)
        
    # Fit training set for model parameters
    w0, w = ri.fit_ridge(X_train, y_train_true, lam=best_lambda)
    
    # Classify, then get square loss, 1/0 error
    y_hat_train = ri.ridge_bin_class(X_train, w, w0, thresh=best_thresh)
    sl_train = ru.square_loss(y_train_true, y_hat_train)
    err_10_train = ru.loss_01(y_train_true, y_hat_train)
    
    # Load testing set
    # Load in MNIST training data
    print("Loading MNIST Testing data...")
    X_test, y_test = mu.load_mnist(dataset='testing')
    y_test_true = mnist_two_filter(y_test)
    print("True number of twos in testing set:",np.sum(y_test_true))
        
    # Classify using training set fit, get square loss, 1/0 error
    y_hat_test = ri.ridge_bin_class(X_test, w, w0, thresh=best_thresh)
    sl_test = ru.square_loss(y_test_true, y_hat_test)
    err_10_test = ru.loss_01(y_test_true, y_hat_test)
    
    # Output!
    print("Square Loss on training, testing set: %.3lf, %.3lf" % (sl_train, sl_test))
    print("1/0 Loss on training, testing set: %.3lf, %.3lf" % (err_10_train, err_10_test))