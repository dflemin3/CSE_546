# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file implements various functions and algorithms pertaining optimization via
gradient descent/ascent.
"""

from __future__ import print_function, division
import numpy as np
from ..classification import classifier_utils as cu


def gradient_ascent(model, X, y, lam=1.0, eta = 1.0e0, w = None, w0 = None, sparse = False,
                     eps = 5.0e-3, max_iter = 500, adaptive = True, llfn = None,
                     savell = False, X_test = None, y_test = None):
    """
    Performs regularized batch gradient descent to optimize model with an update step:

    w_i^(t+1) <- w_i^(t) + eta * {-lambda*w_i^(t) + sum_j[x_i^j(y^j - y_hat^j)]}

    Model call is something like linear_model(X, w, w0, sparse=False)

    In practice, for a learning rate I use k*eta/n instead of eta where k is determined
    each step. k = 1/ sqrt(1.0 + step_number) where step_number starts from 0.

    Note: loss here is actually likelihood since gradient ascent seeks to maximize the
    likelihood.

    If batchsize is set to some int, then perform batch gradient ascent.

    Parameters
    ----------
    model : function
        Model which computes y_hat given X, w, w0
    X : array (n x d)
        training data
    y : array (n x 1)
        training labels
    lam : float (optional)
        l2 regularization constant
    eta : float (optional)
        learning rate
    w : array (d x 1) (optional)
        initial weight vector
    w0 : float (optional)
        initial offset term
    sparse : bool (optional)
        whether or not X is scipy.sparse
    eps : float (optional)
        convergence term
    max_iter : int (optional)
        maximum number of while loop iterations
    adapative : bool (optional)
        whether or not to use adaptive step sizes
    llfn : function (optional)
        which ll function to use with args y, y_hat.  Defaults to logll
    savell : bool (optional)
        whether or not to save the ll at each iteration

    Returns
    -------
    w : array (d x 1)
        best fit weight vector
    w0 : float
        best fit constant offset
    """

    #Define values
    n = y.shape[0]
    d = X.shape[1]

    # Init weight array and holder for convergence checks
    if w is None:
    	# No initial conditions given, assume 0s
    	w_pred = np.zeros((d,1))
    else:
    	# Initial conditions given, use those as first w_pred
		w_pred = np.copy(w)

    if w0 is None:
		w0 = 0.0

    # While not converged, do
    iters = 0
    converged = False

    # Save ll values on training set?
    if savell:
        ll_arr = []
        iter_arr = []

    if X_test is not None and y_test is not None:
        test_ll_arr = []

    # No loss function given -> use log loss
    if llfn is None:
        from ..validation import validation
        llfn = validation.loglike_bin

    # Precompute X transpose since it doesn't change
    XT = X.T

    # Dummy old loglikelihood
    old_ll = -1.0e10
    scale = 1.0/n

    while not converged:
    	# Too many iterations!
    	if iters >= max_iter:
    		print("Maximum iteration threshold hit %d." % iters)
    		print("Returning current solution: Convergence not guarenteed.")
    		print("lambda = %.3e" % lam)
    		print("Old ll, current ll: %e, %e" % (old_ll, ll))
    		if saveloss:
    		    return w0, w_pred, np.asarray(loss_arr), np.asarray(iter_arr)
    		else:
    		    return w0, w_pred

        # Precompute y_hat using w^(t), w0 since this doesn't change in a given iteration
        y_hat = model(X, w_pred, w0, sparse=sparse)
        arg = y - y_hat

        # Update w0 (not regularized!)
        w0 = w0 + eta * scale * np.sum(arg)

        # Loop over features to update according to gradient, learning rate
        # Do so in a vectorized manner
        w_pred = w_pred + eta * scale * (-lam * w_pred + XT.dot(arg))

        # Adapt step size: if loss new > old, decrease stepsize, increase otherwise
        y_hat = model(X, w_pred, w0, sparse=sparse) # P(Y=1|X,w)
        arg = y - y_hat

        ll = llfn(y, (w0 + X.dot(w_pred)))

        if savell:
            ll_arr.append(ll/len(y_hat))
            iter_arr.append(iters)

        # Compute testing set error for this iteration using fit from training set?
        if X_test is not None and y_test is not None:
            # Store test ll
            test_ll_arr.append(llfn(y_test, w0 + X_test.dot(w_pred))/len(y_test))

        # Using an adaptive step size?
        if adaptive:
            scale = 1.0/(n * (1.0 + iters))

        # Is it converged (is loglikelihood not improving?)
        if np.fabs(ll - old_ll)/np.fabs(old_ll) > eps:
            converged = False
        else:
            converged = True

        # Store old_cost, iterate
        old_ll = ll
        iters += 1

    if savell and not (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(iter_arr)
    elif savell and (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), np.asarray(iter_arr)
    else:
        return w0, w_pred
# end function


def batch_gradient_ascent(model, X, y, lam=1.0, eta = 1.0e0, w = None, w0 = None, sparse = False,
                     eps = 1.0e-3, max_iter = 500, adaptive = True, llfn = None,
                     savell = False, X_test = None, y_test = None, batchsize=0.1):
    """
    Performs regularized batch gradient ascent to optimize model with an update step:

    w_i^(t+1) <- w_i^(t) + eta * {-lambda*w_i^(t) + sum_j[x_i^j(y^j - y_hat^j)]}

    Model call is something like linear_model(X, w, w0, sparse=False)

    In practice, for a learning rate I use k*eta/n instead of eta where k is determined
    each step. k = 1/ sqrt(1.0 + step_number) where step_number starts from 0.

    Note: loss here is actually likelihood since gradient ascent seeks to maximize the
    likelihood.

    If batchsize is set to some int, then perform batch gradient ascent.

    Parameters
    ----------
    model : function
        Model which computes y_hat given X, w, w0
    X : array (n x d)
        training data
    y : array (n x 1)
        training labels
    lam : float (optional)
        l2 regularization constant
    eta : float (optional)
        learning rate
    w : array (d x 1) (optional)
        initial weight vector
    w0 : float (optional)
        initial offset term
    sparse : bool (optional)
        whether or not X is scipy.sparse
    eps : float (optional)
        convergence term
    max_iter : int (optional)
        maximum number of while loop iterations
    adapative : bool (optional)
        whether or not to use adaptive step sizes
    llfn : function (optional)
        which ll function to use with args y, y_hat.  Defaults to logll
    savell : bool (optional)
        whether or not to save the ll at each iteration
    batchsize : float (optional)
        fraction of data to use for batch

    Returns
    -------
    w : array (d x 1)
        best fit weight vector
    w0 : float
        best fit constant offset
    """

    #Define values
    n = y.shape[0]
    d = X.shape[1]

    # Init weight array and holder for convergence checks
    if w is None:
    	# No initial conditions given, assume 0s
    	w_pred = np.zeros((d,1))
    else:
    	# Initial conditions given, use those as first w_pred
		w_pred = np.copy(w)

    if w0 is None:
		w0 = 0.0

    # While not converged, do
    iters = 0
    converged = False

    # Save ll values on training set?
    if savell:
        ll_arr = []
        iter_arr = []

    if X_test is not None and y_test is not None:
        test_ll_arr = []

    # No loss function given -> use log loss
    if llfn is None:
        from ..validation import validation
        llfn = validation.loglike_bin

    # Precompute X transpose since it doesn't change
    XT = X.T

    # Dummy old loglikelihood
    old_ll = -1.0e10
    scale = 1.0/n

    while not converged:
    	# Too many iterations!
    	if iters >= max_iter:
    		print("Maximum iteration threshold hit %d." % iters)
    		print("Returning current solution: Convergence not guarenteed.")
    		print("lambda = %.3e" % lam)
    		print("Old ll, current ll: %e, %e" % (old_ll, ll))
    		if saveloss:
    		    return w0, w_pred, np.asarray(loss_arr), np.asarray(iter_arr)
    		else:
    		    return w0, w_pred

        # Create batch indices mask
        ind = int(batchsize * X.shape[0])
        inds = np.random.permutation(X.shape[0])
        inds = inds[:ind]

        # Precompute y_hat using w^(t), w0 since this doesn't change in a given iteration
        # using only batchsize of data
        y_hat = model(X[inds,:], w_pred, w0, sparse=sparse) # P(Y=1|X,w)
        arg = y[inds] - y_hat

        # Update w0 (not regularized!)
        w0 = w0 + eta * scale * np.sum(arg)

        # Loop over features to update according to gradient, learning rate
        # Do so in a vectorized manner
        w_pred = w_pred + eta * scale * (-lam * w_pred + XT[:,inds].dot(arg))

        # Now estimate loss on entire training set
        ll = llfn(y, (w0 + X.dot(w_pred)))

        if savell:
            ll_arr.append(ll/len(y))
            iter_arr.append(iters)

        # Compute testing set error for this iteration using fit from training set?
        if X_test is not None and y_test is not None:
            # Store test ll
            test_ll_arr.append(llfn(y_test, (w0 + X_test.dot(w_pred)))/len(y_test))

        # Using an adaptive step size?
        if adaptive:
            scale = 1.0/(batchsize * n * (1.0 + iters))

        # Is it converged (is loglikelihood not improving?)
        if np.fabs(ll - old_ll)/np.fabs(old_ll) > eps:
            converged = False
        else:
            converged = True

        # Store old_cost, iterate
        old_ll = ll
        iters += 1

    if savell and not (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(iter_arr)
    elif savell and (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), np.asarray(iter_arr)
    else:
        return w0, w_pred
# end function