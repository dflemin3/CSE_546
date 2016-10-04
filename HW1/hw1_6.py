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


def mnist_ridge_thresh(X, y, lam = 1, num = 20, minthresh=9., 
                       maxthresh=20.):
    """
    This function finds the best w (dot) x criteria for labeling a MNIST
    digit as a 2.  For the purposes of this classification, 2 -> 1, while
    everything else -> 0.  Optimize over the threshold by minimizing the square
    loss
    
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
    minthresh : float
        minimum threshold grid value
    maxthresh : float
        maximum threshold grid value
    
    Returns
    -------
    thresh_best : float
        optimal threshold for picking out 2s
    sl_best : float
        minimum square loss
    """
    
    # Filter for 2s for binary classifier "truth"
    y_true = mnist_two_filter(y)
    print("True number of 2s:",np.sum(y_true))
    
    w0, w = ri.fit_ridge(X, y, lam=lam)
    
    y_hat = w0 + np.dot(X,w)
    
    mask = (y_true == 1)
    
    thresh_best = np.median(y_hat[mask])
    
    return thresh_best
    """
    # Make array of thresholds to testout
    thresh = np.linspace(minthresh,maxthresh,num)
    
    loss = np.zeros_like(thresh)
    
    # Loop over thresholds, evaluate model, see which is best
    for ii in range(len(thresh)):
        print("Threshold iteration",ii)
        w0, w, y_hat = ri.ridge_bin_class(X, y, lam=lam, thresh=thresh[ii])
        print("Number of 2s:",np.sum(y_hat))
        
        # Find minimum of MSE to optimize threshold
        loss[ii] = ru.loss_01(y_true, y_hat)
        print("0-1 Loss:",loss[ii])
    
    # Get best threshold (min MSE on training set) and return it
    best_ind = np.argmin(loss)
    
    # Now plot it
    fig, ax = plt.subplots()
    
    ax.plot(thresh,loss,"-o",lw=3)
    ax.set_xlabel(r"$x \bullet w$")
    ax.set_ylabel(r"0-1 Loss")
    
    # Plot best fit and asymptotes
    plt.axvline(x=thresh[best_ind], ymin=-100, ymax = 100, 
                linewidth=3, color='k', ls="--")
    
    fig.tight_layout()
    fig.savefig("sl_thresh.pdf")
    
    return thresh[best_ind], loss[best_ind]
    """
# end function

    
if __name__ == "__main__":
    
    # Flags to control functionality
    find_best_thresh = False
    
    # Best threshold from previous running of script
    #
    # From a previous grid search on training data:
    # Best wx threshold: 0.421
    # Best MSE: 0.033

    #best_thresh = 0.421
    
    # Load in MNIST training data
    print("Loading MNIST Training data...")
    X_train, y_train = mu.load_mnist(dataset='training')
    
    y_train_true = mnist_two_filter(y_train)
    mask_train = (y_train_true == 1)
    
    best_thresh = mnist_ridge_thresh(X_train, y_train)
    print("Best wx threshold: %.3lf" % best_thresh)
        
    if find_best_thresh:
        print("Finding optimal threshold...")
        # Find the best threshold base on 0/1 error
        best_thresh, best_sl = mnist_ridge_thresh(X_train, y_train)
        print("Best wx threshold: %.3lf" % best_thresh)
        print("Best Square Loss: %.3lf" % best_sl)
        
    # Evaluate model on training set, get square loss, 1/0 error
    w0, w, y_hat = ri.ridge_bin_class(X_train, y_train, thresh=best_thresh)
    sl_train = ru.square_loss(y_train_true, y_hat)
    err_10_train = ru.loss_01(y_train_true, y_hat)
    
    # Load testing set
    # Load in MNIST training data
    print("Loading MNIST Testing data...")
    X_test, y_test = mu.load_mnist(dataset='testing')
    
    y_test_true = mnist_two_filter(y_test)
    mask_test = (y_test_true == 1)
        
    # Evaluate model on training set, get square loss, 1/0 error
    w0, w, y_hat = ri.ridge_bin_class(X_test, y_test, thresh=best_thresh)
    sl_test = ru.square_loss(y_test_true, y_hat)
    err_10_test = ru.loss_01(y_test_true, y_hat)
    
    # Output!
    print("Square Loss on training, testing set: %.3lf, %.3lf" % (sl_train, sl_test))
    print("1/0 Loss on training, testing set: %.3lf, %.3lf" % (err_10_train, err_10_test))