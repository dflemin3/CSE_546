# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:03:42 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to fit a linear function using Ridge Regression
and simple routines for building and optimizing a linear classifier again using
Ridge Regression
"""

from __future__ import print_function, division

import numpy as np
import regression_utils as ru

def fit_ridge(X, y, lam=1, sparse=False, w=None, w_0=None):
    """
    Given data x and labels y and optional penalty lam(bda), fit a ridge linear
    regression model using the algorithm described in Section 7.5.2 in Murphy.
    Thismethod is more numerically stable (no matrix inversion) than the
    brute-force solution.

    Parameters
    ----------
    X : array (n x d)
        features array (d features, n samples)
    y : vector (n x 1)
        labels
    lam : float (optional)
        regularization constant
	sparse : bool (optional)
		Whether or not data is sparse.  This does not change functionality but exists for
		compatibility.
	w : array (d x 1) (optional)
    	optional initial conditions for weight vector
    w_0 : float (optional)
    	optional initial condition for constant offset

    Note: Passing w, w_0 to this function does nothing as the ridge regression estimator
    is analytic but is included for backwards compatibility.

    Returns
    -------
    w0 : float
        Constant offset term
    w : vector (d x 1)
        linear weight vector
    """

    # Center data to avoid penalizing constant offset term
    yc = y - np.mean(y)
    Xc = X - np.mean(X,axis=0)

    # Compute sigma, tau as a function of lambda
    sigma = 1.0
    tau = sigma/np.sqrt(lam)

    # Make cholesky decomposition matrix to append to X
    # such that X's shape is (n + d) x d
    delta = (1.0/(tau**2))*np.identity(Xc.shape[-1])
    delta = np.linalg.cholesky(delta)

    # Augment y, x
    yc = np.vstack((y/sigma,np.zeros((Xc.shape[-1],1))))
    Xc = np.vstack((Xc/sigma,delta))

    # Compute weight vector
    w = np.linalg.inv(np.dot(np.transpose(Xc),Xc))
    w = np.dot(w,np.transpose(Xc))
    w = np.dot(w,yc)

    # Compute w0
    w0 = np.mean(y) - np.dot(np.transpose(np.mean(X,axis=0)),w)

    return w0, w
# end function


def ridge_bin_class(X, w, w0, thresh=0.5):
    """
    Use a ridge regression model computed in fit_ridge to use as a binary
    classifier.  In this case, if w dot x >= threshold, return 1, else return
    0 for that element

    Parameters
    ----------
    X : array (n x d)
        features array (d features, n samples)
    y : vector (n x 1)
        labels
    lam : float (optional)
        regularization constant
    thresh : float (optional)
        classification threshold

    Returns
    -------
    w0 : float
        Constant offset term
    w : vector (d x 1)
        linear weight vector
    y : vector (n x 1)
        predictions
    """

    # Evaluate model, return predictions according to threshold
    y_hat = ru.linear_model(X, w, w0)
    y_hat_class = np.zeros_like(y_hat)
    y_hat_class[y_hat >= thresh] = 1

    return y_hat_class
# end function


# Test it out!
if __name__ == "__main__":

	seed = 1
	sparse = False
	w, X, y = ru.generate_norm_data(10000,7,10,sparse=sparse,seed=seed)

	print(w.shape,X.shape,y.shape)

	print("Performing ridge regression...")
	print(fit_ridge(X,y,lam=10.0))
	print(w)