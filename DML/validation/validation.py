# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to be used for model validation, selection, and error estimates.
"""

from __future__ import print_function, division

import numpy as np
from ..regression import regression_utils as ru
from ..regression import ridge_utils as ri
from ..classification import classifier_utils as cu
from ..optimization import gradient_descent as gd
#from numba import jit, float64


def normalize(X, y=None):
    """
    Normalize data via feature l2 norm.

    Parameters
    ----------
    X : array (n x d)
    y : array (n x 1) (optional)

    Returns
    -------
    X : array (n x d)
    y : array (n x 1) (optional)
    """

    # X norm
    norm = np.linalg.norm(X, ord=2, axis=0)

    # If any columns are all 0s, set that norm to 1
    norm[norm == 0] = 1.0

    if y is not None:
        ynorm = np.linalg.norm(y, order=2)

    if y is None:
        return X/norm
    else:
        return X/norm, y/ynorm

    return None
# end function


def MSE(X, y, w, w0):
    """
    Compute the mean squared loss of a prediction

    Parameters
    ----------
    X : array (n x d)
        input data
    y : array (n x 1)
        array of observations
    w : array (d x 1)
        weight vector
    w0 : float
        constant offset

    Returns
    -------
    ll : float
        ll
    """

    y_hat = w0 + X.dot(w)

    return np.sum(np.power(y - y_hat,2))/len(y)
# end function

#@jit(float64(float64[:,:], float64[:,:], float64[:,:], float64[:,:]),
#nopython=True, cache=True)
def MSE_multi(X, y, w, w0):
    """
    Compute the mean squared loss of a prediction for a multiclass predictor.

    Parameters
    ----------
    X : array (n x d)
        input data
    y : array (n x 1)
        array of observations
    w : array (d x 1)
        weight vector
    w0 : float
        constant offset

    Returns
    -------
    ll : float
        ll
    """

    y_hat = w0.T + np.dot(X,w)

    return np.sum((y - y_hat)**2)/len(y)
# end function


def RMSE(y, y_hat):
    """
    Compute the root mean squared error of a prediction

    Parameters
    ----------
    y : array (n x 1)
        array of observations
    y_hat : array (n x 1)
        array of predictions

    Returns
    -------
    rmse : float
        root mean squared error
    """
    return np.sqrt(MSE(y, y_hat))
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
    return np.sum(np.power(y - y_hat,2))/len(y)
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

    return np.sum(y != y_hat)/len(y)
# end function


def reconstruction_error(X, y_hat, mu):
    """
    Compute the mean reconstruction error for an unsupervised algorithm

    Parameters
    ----------
    X : array (n x d)
        input data
    y_hat : (n x 1)
        encoding vector
    mu : (k x d)
        centroid matrix
    """

    err = 0.0
    for ii in range(X.shape[0]):
        err += np.linalg.norm(X[ii,:] - mu[y_hat[ii],:],ord=2)

    return err/X.shape[0]
# end function


def logloss_bin(X, y, w, w0):
    """
    Compute the log loss of a probabilistic prediction for a binary classifier.

    Parameters
    ----------
    X : array (n x d)
        input data
    y : array (n x 1)
        array of observations
    w : array (d x 1)
        weight vector
    w0 : float
        constant offset

    Returns
    -------
    ll : float
        ll
    """
    y_hat = w0 + X.dot(w)
    return -np.sum(y*y_hat - np.log(1.0 + np.exp(y_hat)))/len(y)
# end function


def logloss_multi(X, y, w, w0, sparse=False):
    """
    Compute the -loglikelihood == loss of a probabilitstic prediction for a softmax
    multi-class logistic classifier for N classes.

    Parameters
    ----------
    X : array (n x d)
        input data
    y : array (n x N)
        array of observations
    w : array (d x N)
        weight vector
    w0 : float
        constant offset

    Returns
    -------
    ll : float
        ll
    """

    y_hat = cu.softmax(X, w, w0, sparse=sparse)
    return -np.sum(y*np.log(y_hat))/len(y)
# end function


def sign(x):
	"""
	If x > 0, return 1, else, return -1

	Parameters
	----------
	x : float, array

	Returns
	-------
	sign : int
		+/- 1
	"""

	if x >= 0:
		return 1.0
	else:
		return -1.0
# end function
sign = np.vectorize(sign) # Vectorize it!



def estimate_lambda(X, scale=1.0):
    """
    Estimate regularization constant lambda using dimensional analysis as

    lambda ~ scale * E[X^2]

    Parameters
    ----------
    X : array (n x d)
        training data
    scale : float (optional)

    Returns
    -------
    lambda : float
    """

    return scale * np.mean(X*X)
# end function


def summarize_loss(X_train, y_train, X_test, y_test, w, w0, y_train_label=None,
                   y_test_label=None, classfn=None, lossfn=None):
    """
    Print a loss summary for training and testing data.

    Note y_train and y_train_label are not always the same.  For example, y_train could
    be a matrix of 0, 1 while y_train_label could be a vector with digits 0-9 for a
    softmax classifier.

    Parameters
    ----------
    X_train : array (n_train x d)
        training input data
    y_train : array (n_train x 1)
        training labels
    X_test : array (n_test x d)
        testing input data
    y_test : array (n_test x 1)
        testing labels
    w : vector (d x 1)
        weight vector
    w0 : float
        constant offset
    classfn : function
        If provided, compute 0/1 loss for a classifier
    lossfn : function
        If provide, compute some loss metric using predictions
    y_train_label : array (n_train x 1)
    y_test_label : array (n_test x 1)

    Returns
    -------
    None (outputs to console...for now)
    """

    # Labels default to y values if no special parsing has to be done
    if y_train_label is None:
        y_train_label = y_train
    if y_test_label is None:
        y_test_label = y_test

    # Now compute, output 0-1 loss for training and testing sets
    if classfn is not None:
        y_hat_train = classfn(X_train, w, w0)
        y_hat_test = classfn(X_test, w, w0)

        error_train = loss_01(y_train_label, y_hat_train)
        error_test = loss_01(y_test_label, y_hat_test)
        print("Training, testing 0-1 loss: %.3lf, %.3lf" % (error_train, error_test))

    if lossfn is not None:
        # Now compute, output logloss for training and testing sets
        logl_train = lossfn(X_train, y_train, w, w0)
        logl_test = lossfn(X_test, y_test, w, w0)
        print("Training, testing logloss: %.3lf, %.3lf" % (logl_train, logl_test))

    return None
# end function


def split_data(X_train, y_train, frac=0.1, seed=None):
	"""
	Randomly partition the data into two sets, i.e., a training set and a validation set,
	where the latter is a fraction frac of the former.

	Parameters
	----------
	X_train : array (n x d)
		input data
	y_train : array (n x 1)
		input labels
	frac : float (optional)
		fraction of data to set aside for validation/other set. Defaults to 0.1
	seed : int (optional)
		Number to seed the RNG

	Returns
	-------
	X_tr : array ((1-frac)*n x d)
	y_tr : array ((1-frac)*n x 1)
	X_val : array (frac*n x d)
	y_val : array (frac*n x 1)
	"""
	# If seed set, pick the same validation set
	if seed is not None:
		np.random.seed(seed)

	# Randomly partition training input into training and validation
	ind = int(frac * X_train.shape[0])
	indices = np.random.permutation(X_train.shape[0])
	training_idx, val_idx = indices[ind:], indices[:ind]
	X_tr, X_val = X_train[training_idx,:], X_train[val_idx,:]
	y_tr, y_val = y_train[training_idx], y_train[val_idx]

	return X_tr, y_tr, X_val, y_val
# end function


def linear_reg_path(X_train, y_train, X_val, y_val, model, lammax=1000., scale=2.0,
					num=10, error_func=MSE, thresh=None, save_nonzeros=False, best_w=False,
					**kwargs):
	"""
	Perform a regularization path for a regularized linear regression technique by fitting
	the model on the training data and evaluating it on the validation data to determine
	the best regularization constant.

	Parameters
	----------
	X_train : array (n x d)
		training input data
	y_train : array (n x 1)
		training labels
	X_val : array (m x d)
		validation input data
	y_val : array (m x 1)
		validation labels
	model : function
		linear model
	lammax : float (optional)
		maximum regularization lambda.  Defaults to 1.0e3
	lammin : float (optional)
		minimum regularization lambda.  Defaults to 1.0e-3
	num : int (optional)
		number of lambdas to search over.  Defaults to 10
	error_func : function (optional)
		error function.  Defaults to mean square error (MSE)
	kwargs : dict
		any additional parameters required by the linear model

	Returns
	-------
	error : array (num x 1)
		error as a function of lambda
	lams : array (num x 1)
		lambda grid
	"""

	# If you're using thresholding for classifying, must use 0-1 loss
	if thresh is not None:
		assert(error_func is loss_01)

	# Using sparse data?
	if "sparse" in kwargs:
		sparse = kwargs["sparse"]
	else:
		sparse = False

	# Init lambda at lammax, init storage arrays
	lam = lammax
	lams = []
	error_val = np.zeros(num)
	error_train = np.zeros(num)

	# If using LASSO, can be useful to save number of nonzero weights
	if save_nonzeros:
		nonzeros = np.zeros(num)

	# Assume null solution to begin with
	w = np.zeros((X_train.shape[-1],1))
	w_0 = 0.0

	# Save the best fitting weight vector?
	if best_w:
		best_w0 = np.zeros(num)
		best_w = np.zeros((n,len(w)))

	# Main loop over lambdas
	for ii in range(num):
		print("Regularization path iteration: %d" % ii)
		# Fit on training data using previous w, w_0 as initial conditions
		w_0, w = model(X_train, y_train, lam=lam, w=w, w_0=w_0, **kwargs)

		# Threshold prediction for classification?
		if thresh is not None:
			y_hat_val = ri.ridge_bin_class(X_val, w, w_0, thresh=thresh)
			y_hat_train = ri.ridge_bin_class(X_train, w, w_0, thresh=thresh)
		else:
			# Evaluate model on validation to get predictions
			y_hat_val = ru.linear_model(X_val, w, w_0, sparse=sparse)
			y_hat_train = ru.linear_model(X_train, w, w_0, sparse=sparse)

		# Save number of nonzeros in predicted weight vector?
		if save_nonzeros:
			nonzeros[ii] = np.sum((w != 0))

		# Save w, w0s?
		if best_w:
			best_w0[ii] = w_0
			best_w[ii] = w

		# Evaluate error on validation and training set
		error_val[ii] = error_func(y_val, y_hat_val)
		error_train[ii] = error_func(y_train, y_hat_train)

		# Save, scale lambda for next iteration
		lams.append(lam)
		lam /= scale

	# Find best weight vector?
	if best_w:
		ind = np.argmin(error_val)

	if save_nonzeros and not best_w:
		return error_val, error_train, np.array(lams), nonzeros
	elif not save_nonzeros and best_w:
		return error_val, error_train, np.array(lams), best_w0[ind], best_w[ind]
	elif save_nonzeros and best_w:
		return error_val, error_train, np.array(lams), best_w0[ind], best_w[ind], nonzeros
	else:
		return error_val, error_train, np.array(lams)
# end function


def logistic_reg_path(X_train, y_train, X_val, y_val,
                         y_train_label = None, y_val_label = None,
                         grad=cu.bin_logistic_grad, lammax=1000.,
                         scale=2.0, num=10, error_func=loss_01, thresh=0.5, best_w=False,
                         eta = 1.0e-4, sparse=False, eps=5.0e-3, max_iter=1000,
                         adaptive=True, llfn=logloss_bin, savell=False, batchsize=100,
                         classfn=cu.logistic_classifier, **kwargs):
    """
    Perform a regularization path for a regularized logistic regressor by fitting
    the model on the training data and evaluating it on the validation data to determine
    the best regularization constant via batch gradient descent.  Behavior defaults to
    binary logistic regression and computing 0/1 loss.

    Note y_train and y_train_label are not always the same.  For example, y_train could
    be a matrix of 0, 1 while y_train_label could be a vector with digits 0-9 for a
    softmax classifier.

    Parameters
    ----------
    X_train : array (n x d)
        training input data
    y_train : array (n x 1)
        parsed training labels
    y_train_label : array (n x 1)
        training labels
    X_val : array (m x d)
        validation input data
    y_val : array (m x 1)
        validation labels
    y_val_labels : array (m x 1)
    grad : function
        function which computes gradient of w, w0
    lammax : float (optional)
        maximum regularization lambda.  Defaults to 1.0e3
    num : int (optional)
        number of lambdas to search over.  Defaults to 10
    error_func : function (optional)
        error function.  Defaults to log loss
    classfn : function (optional)
        function which classifies based on prediction
    kwargs : dict
        any additional parameters required by the linear model
    For other parameters, see parameters for gradient_descent

    Returns
    -------
    error : array (num x 1)
        error as a function of lambda
    lams : array (num x 1)
        lambda grid
    """

    # Using sparse data?
    if "sparse" in kwargs:
        sparse = kwargs["sparse"]
    else:
        sparse = False

    # Init lambda at lammax, init storage arrays
    lam = lammax
    lams = []
    error_val = np.zeros(num)
    error_train = np.zeros(num)

    # Labels default to y values if no special parsing has to be done
    if y_train_label is None:
        y_train_label = y_train
    if y_val_label is None:
        y_val_label = y_val

    # Assume null solution to begin with
    w = np.zeros((X_train.shape[-1],1))
    w0 = 0.0

    # Save the best fitting weight vector?
    if best_w:
        best_w0 = np.zeros(num)
        best_w_arr = np.zeros((num,len(w)))

    for ii in range(num):
        print("Regularization path iteration: %d, lam = %.3e" % (ii, lam))

        # Solve logistic regression using gradient descent on the training data
        # optimizing over the logloss
        # If no batchsize given, use BGD, else use minibatch SGD
        if batchsize is None:
            w0, w = gd.gradient_descent(grad, X_train, y_train, lam=lam, eta=eta,
                                            sparse=sparse, eps=eps, max_iter=max_iter,
                                            adaptive=adaptive, llfn=llfn, savell=False)
        else:
            w0, w = gd.stochastic_gradient_descent(grad, X_train, y_train, lam=lam, eta=eta,
                                            sparse=sparse, eps=eps, max_iter=max_iter,
                                            adaptive=adaptive, llfn=llfn, savell=False,
                                            batchsize=batchsize)

        # Now classify on both validation and training set!
        y_hat_val = classfn(X_val, w, w0, thresh=thresh, sparse=sparse)
        y_hat_train = classfn(X_train, w, w0, thresh=thresh, sparse=sparse)

        # Save w, w0s?
        if best_w:
            best_w0[ii] = w0
            best_w_arr[ii] = np.squeeze(w)

        # Evaluate error on validation and training set
        error_val[ii] = error_func(y_val_label, y_hat_val)
        error_train[ii] = error_func(y_train_label, y_hat_train)

        # Save, scale lambda for next iteration
        lams.append(lam)
        lam /= scale

    # Find best weight vector?
    if best_w:
        ind = np.argmin(error_val)

    if best_w:
        return error_val, error_train, np.array(lams), best_w0[ind], best_w_arr[ind].reshape((X_train.shape[-1],1))
    else:
        return error_val, error_train, np.array(lams)
# end function
