# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:03:42 2016

@author: dflemin3

This file contains routines to fit a linear function using Ridge Regression
and simple routines for building and optimizing a linear classifier again using
Ridge Regression
"""

import numpy as np
import mnist_utils as mu

def fit_ridge(X, y, lam=1):
    """
    Given data x and labels y and optional penalty lam(bda), fit a ridge linear
    regression model

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
    w : vector (d x 1)
        linear weight vector
    """

    # First define matrix to be inverted
    w = lam*np.identity(len(y)) + np.dot(np.transpose(X),X)
    print(w.shape)
    return w
    # Invert!
    w = np.linalg.inv(w)
    print(w.shape)
    return w

    # Final multiplication
    w = np.dot(w,np.transpose(X))
    w = np.dot(w,y)

    return w

# Test
if __name__ == "__main__":
    print("Loading data...")
    images, labels = mu.load_mnist(dataset='training')

    print("Performing ridge regression...")
    print(fit_ridge(images,labels))