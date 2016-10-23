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
from ..regression import regression_utils as ru
from ..data_processing import mnist_utils as mu

def sigmoid(x):
    """
    Compute sigmoid of x for sigmoid link function defined as follows:

    sigmoid(x) = 1/(1 + e^(x))

    Note for this implementation, I use x instead of -x to be consistent with
    class definitions.

    Parameters
    ----------
    x : float, array
        Input argument

    Returns
    -------
    sigmoid : float
        sigmoid of x

    """
    return 1.0/(1.0 + np.exp(x))
# end function


def generate_binary_data(n, k, d, sparse=False, w0 = 0.0, seed = None):
    """
    Generates independent data pairs (x_i,y_i) according to the following binary
    classification logistic regression model:

    P(Y = 0 | X, w) = 1 / (1 + exp(w0 + X^Tw)
    P(Y = 1 | X, w) = 1 - P(Y = 0 | X, w)

    where Y = 0 if w0 + X^Tw < 0, Y = 1 otherwise
    aka P < 0.5 -> Y = 1

    Parameters
    ----------
    n : int
        Number of samples
    k : int
        k < d number of features for dimensions d
    d : int
        number of dimensions
    sparse : bool
        Whether or not to consider input data matrix X is sparse
    w0 : float
    	constant offset
    seed : float
    	Numpy RNG seed (only set if given)
    Returns
    -------
    w : vector
        true weight vector
    X : n x d matrix
        data matrix
    y : n x 1 vector
    """

    # Generate synthetic linear data, ignore y then 0 it out
    w, X, y = ru.generate_norm_data(n, k, d, sigma=1.0, sparse=sparse, w0=w0, seed=seed)
    y = np.zeros_like(y)

    # Mask where 1s are since rest are 0s
    y[w0 + X.dot(w) > 0.0] = 1

    return w, X, y
# end function


def logistic_model(X, w, w0, sparse=False):
    """
    Binary logistic regression conditional probability of P(Y = 1 | x, w) applicable for
    classification problems where labels are either 0 or 1.  Returns a probability!

    Parameters
    ----------
    X : array (n x d)
        Input data
    w : vector (d x 1)
        weight coefficients
    w0 : float
        constant offset term
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    y_hat : array (n x 1)
        P(Y = 1 | x, w)
    """

    arg = w0 + X.dot(w)
    return 1.0 - 1.0/(1.0 + np.exp(arg)) # Since probabilities sum to 1
# end function


def softmax(X, w, w0, sparse=False):
    """
    Softmax multi-class logistic classifier estimator (P(Y = c | X, w)) for N classes

    Parameters
    ----------
    X : array (n x d)
        Input data
    w : array (d x N)
        weight coefficients
    w0 : vector (N x 1)
        constant offset term
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    y_hat : array (n x N)
        P(Y = c | X, w)
    """

    return np.exp(w0.T + X.dot(w))/np.sum(np.exp(w0.T + X.dot(w)),axis=1).reshape((X.shape[0],1))
# end function


def multi_logistic_grad(X, y, w, w0, sparse=False):
    """
    Compute the gradient of the loglikelihood function for multiclass logistic regression
    for both w and w0 for N classes

    Parameters
    ----------
    X : array (n x d)
        Input data
    y : array (n x N)
        labels
    w : array (d x N)
        weight coefficients
    w0 : vector (N x 1)
        constant offset term
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    wgrad : array (d x N)
        gradient of weight vector
    w0grad : float
        gradient of constant offset
    """
    arg = y - softmax(X, w, w0, sparse=sparse)

    wgrad = -X.T.dot(arg)
    w0grad = -np.sum(arg)
    return wgrad, w0grad
# end function


def bin_logistic_grad(X, y, w, w0, sparse=False):
    """
    Compute the gradient of the loglikelihood function for binary logistic regression for
    both w and w0

    Parameters
    ----------
    X : array (n x d)
        Input data
    y : array (n x 1)
        labels
    w : vector (d x 1)
        weight coefficients
    w0 : float
        constant offset term
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    wgrad : array (d x 1)
        gradient of weight vector
    w0grad : float
        gradient of constant offset
    """
    arg = y - logistic_model(X, w, w0, sparse=sparse)

    wgrad = -X.T.dot(arg)
    w0grad = -np.sum(arg)
    return wgrad, w0grad
# end function


def logistic_classifier(X, w, w0, thresh = 0.5, sparse = False):
    """
    Binary logistic regression classifier.  Returns 1 if P(Y = 1 | X, w) > thresh.

    Parameters
    ----------
    X : array (n x d)
        Input data
    w : vector (d x 1)
        weight coefficients
    w0 : float
        constant offset term
    thesh : float (optional)
        Classification threshold.  Defaults to 0.5.
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    y_hat : array (n x 1)
        P(Y = 1 | x, w)
    """

    py1 = logistic_model(X, w, w0, sparse=sparse)
    y_hat = np.zeros((X.shape[0],1))
    y_hat[py1 > thresh] = 1
    return y_hat

# end function


def multi_logistic_classifier(X, w, w0, thresh = 0.5, sparse = False):
    """
    Multiclass logistic regression classifier for N classes

    Parameters
    ----------
    X : array (n x d)
        Input data
    w : array (d x N)
        weight coefficients
    w0 : vector (N x 1)
        constant offset term
    thesh : float (optional)
        Classification threshold.  Defaults to 0.5.
    sparse : bool (optional, not implemented)
        whether or not X is a scipy.sparse array

    Returns
    -------
    y_hat : array (n x 1)
        P(Y = 1 | x, w)
    """

    y_hat = softmax(X, w, w0, sparse=sparse)
    return np.argmax(y_hat, axis=1).reshape((X.shape[0],1))

# end function

def multi_linear_classifier_fit(X, y, N, lam=1.0e6, thresh=0.4):
    """
    Using ridge regression, fit for the label of each sample where labels
    are ints from 0 to N-1.  Produces a len(y) x N prediction matrix.

    Parameters
    ----------
    X : array (n x d)
        features array (d features, n samples)
    y : vector (n x 1)
        labels
    n : int
        Number of classes
    lam : float (optional)
        regularization constant
	thresh : float (optional)

    Returns
    -------
    w0 : float
        Constant offset term
    w : vector (d x N)
        linear weight vector for N number of classes
    """
    # Precompute hat matrix H

    # Center data to avoid penalizing constant offset term
    Xc = X - np.mean(X,axis=0)

    # Compute sigma, tau as a function of lambda
    sigma = 1.0
    tau = sigma/np.sqrt(lam)

    # Make cholesky decomposition matrix to append to X
    # such that X's shape is (n + d) x d
    delta = (1.0/(tau**2))*np.identity(Xc.shape[-1])
    delta = np.linalg.cholesky(delta)

    # Augment y, x
    Xc = np.vstack((Xc/sigma,delta))

    # Compute weight vector
    H = np.linalg.inv(np.dot(np.transpose(Xc),Xc))
    H = np.dot(H,np.transpose(Xc))

    # Init matrix to hold weight vectors for each class
    w = np.zeros((X.shape[-1],N))
    w0 = np.zeros((N,1))

    # Now loop over classes and regress for each class
    for ii in range(N):

        # Filter y values to 0, 1 labels according to label
        y_filt = mu.mnist_filter(y,filterby=ii)

        # Now finish the ridge prediction for this class
        yc = y_filt - np.mean(y_filt)
        yc = np.vstack((yc/sigma,np.zeros((Xc.shape[-1],1))))

        # Save predictions
        w[:,ii] = np.squeeze(np.dot(H,yc))
        w0[ii] = np.mean(y) - np.dot(np.transpose(np.mean(X,axis=0)),w[:,ii])

    return w0, w
# end function


def multi_linear_classifier(X, w, w0):
    """
    Predict the class label of the samples given the feature x class label matrix
    and constant offset w0.

    Parameters
    ----------
    X : array (n x d)
        features array (d features, n samples)
    w : array (d x N)
        linear weight vector for N number of classes
    w0 : array (N x 1)
        Constant offset term

    Returns
    -------
    y_hat : array (n x 1)
        class label prediction vector
    """

    # Compute (n x N) prediction matrix for N classes, n samples
    return np.argmax(w0.T + X.dot(w),axis=1).reshape((X.shape[0],1))
# end functions