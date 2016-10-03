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

def fit_ridge(X, y, lam=1):
    """
    Given data x and labels y and optional penalty lam(bda), fit a ridge linear
    regression model using the algorithm described in Section 7.5.2 in Murphy. This
    method is more numerically stable (no matrix inversion) than the brute-force solution

    Parameters
    ----------
    X : array (n x d)
        features array (d features, n samples)
    y : vector (n x 1)
        labels
    lam : float (optional)
        regularization constant

    Returns
    -------
    w0 : float
        Constant offset term
    w : vector (d x 1)
        linear weight vector
    """
    
    # Center data to avoid penalizing constant offset term
    yc = y - np.mean(y)
    Xc = X - X.mean(axis=1, keepdims=True)
    
    # Compute sigma, tau
    sigma = 1.0
    tau = sigma/np.sqrt(lam)

    # Make cholesky decomposition matrix to append to X 
    # such that X is (n + d) x d    
    delta = (1.0/(tau**2))*np.identity(Xc.shape[-1])
    delta = np.linalg.cholesky(delta)

    # Augment y, x
    yc = np.vstack((y/sigma,np.zeros((Xc.shape[-1],1))))
    Xc = np.vstack((Xc/sigma,delta))

    # Perform QR Decomposition
    Q, R = np.linalg.qr(Xc)
    
    # Finish!
    w = np.dot(np.linalg.inv(R),Q.T)
    w = np.dot(w,yc)
    
    # Compute w0
    w0 = (np.sum(y) - np.sum(np.dot(X,w)))/len(y)
    
    return w0, w
    
# Test it out!
if __name__ == "__main__":
    
    w, X, y = ru.generate_norm_data(10000,5,10)
    
    print(w.shape,X.shape,y.shape)
    
    print("Performing ridge regression...")
    print(fit_ridge(X,y,lam=1))
    print(w)