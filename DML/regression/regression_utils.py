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

def generate_norm_data(n, k, d, sigma=1.0,sparse=False, w0 = 0, seed = None):
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
    assert(k < d), "k < d must hold for k: %d, d: %d" % (k,d)

    if seed is not None:
		np.random.seed(seed)

    # Create w vector
    # Create a w* by setting the first k elements to +/-10
    # (choose any sign pattern) and the remaining elements to 0
    w = np.zeros(d).reshape((d,1))
    for i in range(k):
        if i < k/2:
            w[i] = 10
        else:
            w[i] = -10

    # Generate n x d data matrix X for each element from N(0,1)
    X = np.random.randn(n, d)

    if sparse:
        X = sp.csc_matrix(X)

    # Generate n x 1 array of Gaussian error samples from N(0,sigma)
    eps = sigma * np.random.randn(n).reshape(n,1)

    # Form y = Xw* + w*_0 + eps for w*_0 assumed to be 0
    if sparse:
        y = X.dot(w) + eps + w0
    else:
        y = np.dot(X,w) + eps + w0

    return w, X, y.reshape((len(y),1))
# end function


def linear_model(X, w, w0, sparse=False):
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
    sparse : bool
    	whether or not X is scipy.sparse.csc. defaults to True

    Returns
    -------
    y : array (n x 1)
        prediction vector
    """
    if sparse:
    	return w0 + X.dot(w)
    else:
    	return w0 + np.dot(X,w)
# end function


def r_squared(X, y, w, w0, sparse=False):
    """
	Compute r^2 for a simple linear model of the form
	y = w0+ Xw

	Parameters
	----------
	X : array (n x d)
		Data array (n observations, d features)
	w : array (d x 1)
		feature weight array
	w0 : float
		constant offset term
	sparse : bool
		whether or not X is scipy.sparse.csc. defaults to True

	Returns
	-------
	y : array (n x 1)
		prediction vector
	"""
    if sparse:
    	y_hat = w0 + X.dot(w)
    else:
    	y_hat = w0 + np.dot(X,w)

    SSres = np.sum(np.power(y - y_hat,2.0))
    SStot = np.sum(np.power(y - np.mean(y),2.0))
    return 1.0 - SSres/SStot
# end function


def precision_lasso(w_true, w_pred, eps = 1.0e-3):
	"""
	Calculate number of correct non-zeros in w_pred divided by the
	total number of zeros in w_pred for the lasso algorithm.

	Parameters
	----------
	w_true : array (d x 1)
		true weight array
	w_pred : array (d x 1)
		predicted weight array
	eps : float (optional)
		equality threshold

	Returns
	-------
	prec : int
		Number of correct estimated parameters
	"""

	# Find non-zeros in predictions, true
	true_nzero_mask = (np.fabs(w_true) > eps)
	pred_nzero_mask = (np.fabs(w_pred) > eps)

	# number of correct non-zeros / total non-zeros in predictions
	return np.sum(true_nzero_mask & pred_nzero_mask) / np.sum(pred_nzero_mask)
# end function


def recall_lasso(w_true, w_pred, eps = 1.0e-3):
	"""
	Number of correct non-zeros in w_pred divided by number of true non-zeros
	for the lasso algorithm.

	Parameters
	----------
	w_true : array (d x 1)
		true weight array
	w_pred : array (d x 1)
		predicted weight array
	eps : float (optional)
		equality threshold

	Returns
	-------
	rec : int
		Number of correct non-zeros in w_pred divided by number of true non-zeros.
	"""

	# Find non-zeros in predictions, true
	true_nzero_mask = (np.fabs(w_true) > eps)
	pred_nzero_mask = (np.fabs(w_pred) > eps)

	return np.sum(true_nzero_mask & pred_nzero_mask) / np.sum(true_nzero_mask)
# end function


def col_max_filt(X):
	"""
	Alter X according to the column-wise max(x_i, 0)

	Parameters
	----------
	X : array (n x d)

	Returns
	-------
	Xprime : array (n x d)
		Note: Xprime is X
	"""
	for ii in range(X.shape[-1]):
		if np.sum(X[:,ii]) < 0:
			X[:,ii] = np.zeros(X.shape[0])

	return X
# end function


def naive_nn_layer(X, k=10000):
	"""
	Perform a naive approximation to the first layer of a neural network to transform
	a d x 1 dimensional feature vector for a given sample to k x 1 via a linear
	combination of the original d features.  The mapping is given by the following

	h_i(x) = max(v_i dot x, 0)

	where v is a d x k matrix where each column is a d x 1 vector whose entries are
	sampled from the standard normal distribution.

	Parameters
	----------
	X : array (n x d)
		input data
	k : int
		number of features for of new sample

	Returns
	-------
	Xprime : array (n x k)
		transformed data
	"""

	# Generate feature mapping as random variates from standard normal distribution
	v = np.random.normal(size=(X.shape[-1],k))
	return col_max_filt(X.dot(v))
# end function






