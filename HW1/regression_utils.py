# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:59:49 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains utility functions for generating fake data and testing the results
of regression algorithms for speed, accuracy, etc.
"""
from __future__ import print_function, division
import numpy as np
import scipy.sparse as sp

def generate_norm_data(n,k,d,sigma=1,sparse=False):
    """
    Generates independent data pairs (x_i,y_i) according to the following model:
    
    yi = w*_0 + w*_1x_i_1 + w*2 x_i_2 + ... w*k x_i_k + eps_i
    
    ==
    
    y = Xw + eps
    
    for eps_i = Gaussian noise of the form N(0,sigma^2)
    and each element of X (shape n x d) is from N(0,1)
    
    Parameters
    ----------
    n : int
        Number of samples
    k : int
        k < d number of features for dimensions d
    d : int
        number of dimensions
    sigma : float
        Gaussian error standard deviation
    sparse : bool
        Whether or not to consider input data matrix X is sparse
        
    Returns
    -------
    w : vector
        true weight vector
    X : n x d matrix
        data matrix
    y : n x 1 vector
    """
    assert(k < d), "k < d must hold for k: %lf, d: %lf" % (k,d)
    
    #Create w vector
    #Let w[0] = 0 and create a w∗ by setting the first k elements to ±10 
    #(choose any sign pattern) and the remaining elements to 0
    w = np.zeros(d).reshape((d,1))
    for i in range(1,k+1):
        if i < k/2:
            w[i] = 10
        else:
            w[i] = -10
    
    #Generate n x d data matrix X for each element from N(0,1)
    X = np.random.randn(n, d)
    
    if sparse:
        X = sp.csc_matrix(X)
    
    #Generate n x 1 array of Gaussian error samples from N(0,sigma^2)
    eps = np.random.randn(n).reshape(n,1)
    
    #Finally, generate a Gaussian noise vector eps with variance sigma^2 and 
    #form y = Xw* + w*_0 + eps for w*_0 assumed to be 0
    if sparse:    
        y = X.dot(w) + eps
    else:
        y = np.dot(X,w) + eps
        
    return w, X, y.reshape((len(y),1))
# end function


def linear_model(X, w, w0):
    """
    Evaluate a simple linear model of the form
    y = w0+ Xw
    
    Parameters
    ----------
    X : array (n x d)
        Data array (n observations, d features)
    w : array (d x 1)
        feature weight array
    w0 : float
        constant offset term
        
    Returns
    -------
    y : array (n x 1)
        prediction vector
    """
    
    return w0 + np.dot(X,w)
# end function


def MSE(y, y_hat):
    """
    Compute the mean squared error of a prediction
    
    Parameters
    ----------
    y : array (n x 1)
        array of observations
    y_hat : array (n x 1)
        array of predictions
    
    Returns
    -------
    mse : float
        mean squared error
    """
    
    return np.sum(np.power(y - y_hat,2))/len(y)
# end function


def square_loss(y, y_hat):
    """
    Compute the square loss of a binary classifier
    
    Parameters
    ----------
    y : array (n x 1)
        array of observations
    y_hat : array (n x 1)
        array of predictions
    
    Returns
    -------
    sl : float
        square loss
    """
    return np.sum(np.power(y - y_hat,2))
# end function


def loss_01(y, y_hat):
    """
    Compute the 0-1 loss of a binary classifier
    
    Parameters
    ----------
    y : array (n x 1)
        array of observations
    y_hat : array (n x 1)
        array of predictions
    
    Returns
    -------
    loss : float
        0-1 loss
    """
    return np.sum(y != y_hat)
# end function
    
    
    
    
    
    
    
    
    
    
    