# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to be used for model validation, selection, and error estimates.
"""

import numpy as np
import regression_utils as ru
import ridge_utils as ri

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


def linear_reg_path(X_train, y_train, model, val_frac=0.1, lammax=1.0e3, lammin=1.0e-3, num=10,
					error_func = MSE, thresh=None, seed=None, **kwargs):
	"""
	Perform a regularization path for either Ridge Regression or LASSO regression by spliting
	the training test into training and validation and evaluating the error on the
	validation set.

	Parameters
	----------
	X_train : array (n x d)
		training input data
	y_train : array (n x 1)
		training labels
	model : function
		linear model
	val_frac : float (optional)
		fraction of training data to set aside for validation. Defaults to 0.1
	lammax : float (optional)
		maximum regularization lambda.  Defaults to 1.0e3
	lammin : float (optional)
		minimum regularization lambda.  Defaults to 1.0e-3
	num : int (optional)
		number of lambdas to search over.  Defaults to 10
	error_func : function (optional)
		error function.  Defaults to mean square error (MSE)
	seed : float (optional)
		RNG seed
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

	if "sparse" in kwargs:
		sparse = kwargs["sparse"]
	else:
		sparse = False

	# If seed set, pick the same validation set
	if seed is not None:
		np.random.seed(seed)

	# Randomly partition training input into training and validation
	ind = int(val_frac * X_train.shape[0])
	indices = np.random.permutation(X_train.shape[0])
	training_idx, val_idx = indices[:ind], indices[ind:]
	X_tr, X_val = X_train[training_idx,:], X_train[val_idx,:]
	y_tr, y_val = y_train[training_idx], y_train[val_idx]

	# Generate lambda array, empty error array
	lams = np.logspace(np.log10(lammin),np.log10(lammax),num)
	error = np.zeros_like(lams)

	# Loop over lambdas
	for ii in range(len(lams)):
		# Fit on training data
		w_0, w = model(X_tr, y_tr, lam=lams[ii], **kwargs)

		# Threshold prediction for classification?
		if thresh is not None:
			y_hat = ri.ridge_bin_class(X_val, w, w_0, thresh=thresh)
		else:
			# Evaluate model on validation to get predictions
			y_hat = ru.linear_model(X_val, w, w_0, sparse=sparse)

		# Evaluate error
		error[ii] = error_func(y_val, y_hat)

	return error, lams
# end function


def k_folds_linear(X, y, model, k=3, loss=MSE, **kwargs):
	"""
	Perform k-folds cross validation to estimate the error
	for linear regressors.

	Parameters
	----------
	X : array (n x d)
		training data
	y : array (n x 1)
		training labels
	model : function
		model to fit to data
	k : int (optional)
		number of folds to use.  Defaults to 3
	loss : function (optional)
		loss function to use for error estimate.  Defaults to mean squared error (MSE)
	kwargs : dict
		any keyword arguments required by model

	Returns
	-------
	error : float
		k-fold cross-validation error
	"""

	raise NotImplementedError("I haven't coded this one up yet!")
	return None
	"""

	for ii in range(k):
		# First split data to exclude kth fold (set aside for testing)
		#X_train = [k,:]
		#y_train = [k]

		#y_test = [k+1]

		# Fit model using training data with kth fold excluded
		#w_0, w = model(X_train, y_train, **kwargs)

	return None
	"""