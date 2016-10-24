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


def gradient_descent(grad, X, y, lam=1.0, eta = 1.0e-3, w = None, w0 = None, sparse = False,
                     eps = 5.0e-3, max_iter = 500, adaptive = True, llfn = None,
                     savell = False, X_test = None, y_test = None, multi=None, classfn=None,
                     train_label=None,test_label=None, loss01fn=None):
    """
    Performs regularized batch gradient descent to optimize model with an update step:

    w_i^(t+1) <- w_i^(t) - eta * {-lambda*w_i^(t) + gradient}

    Model call is something like linear_model(X, w, w0, sparse=False)

    In practice, for a learning rate I use k*eta/n instead of eta where k is determined
    each step. k = 1/(1.0 + sqrt(step_number)) where step_number starts from 0.

    Parameters
    ----------
    grad : function
        Model which computes gradient of w, w0 given X, w, w0, y
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
        which ll function to use with args y, y_hat.  Defaults to logloss_bin
    savell : bool (optional)
        whether or not to save the ll at each iteration
    multi : int (optional)
        If not None, multi sets number of classes
    classfn : func (optional)
        Function to classify.  If provided, used to compute/return 0/1 loss
    train_label : array (n x 1) (optional)
        training labels used for 0/1 loss calculation
    test_label : array (n x 1) (optional)
        testing labels used for 0/1 loss calculation

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

    # No multi == no multiclasses
    if multi is None:
        multi = 1

    # Init weight array and holder for convergence checks
    if w is None:
    	# No initial conditions given, assume 0s
    	w_pred = np.zeros((d,multi))
    else:
    	# Initial conditions given, use those as first w_pred
		w_pred = np.copy(w)

    if w0 is None:
        if multi is None:
            w0 = 0.0
        else: # Multiclass -> w0 becomes a vector
	        w0 = np.zeros((multi,1))

    # While not converged, do
    iters = 0
    converged = False

    # Save ll values on training set?
    if savell:
        ll_arr = []
        iter_arr = []

    # Save 01 loss?
    if classfn is not None:
        train_01_loss = []

    # Save ll values on testing set?
    if X_test is not None and y_test is not None:
        test_ll_arr = []
        if classfn is not None:
            test_01_loss = []

    # No loss function given -> use loss for a binary classifier
    if llfn is None:
        from ..validation import validation
        llfn = validation.logloss_bin

    # Dummy old super huge loss
    old_ll = 1.0e10
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

        # Precompute gradient using w^(t), w0 since this doesn't change in a given iteration
        wgrad, w0grad = grad(X, y, w_pred, w0, sparse=sparse)

        # Update w0 (not regularized!)
        w0 = w0 - eta * scale * w0grad

        # Loop over features to update according to gradient, learning rate
        # Do so in a vectorized manner
        w_pred = w_pred - eta * scale * (lam * w_pred + wgrad)

        # Compute loss for this fit
        ll = llfn(X, y, w_pred, w0)
        print(ll)

        if savell:
            ll_arr.append(ll/len(y))
            iter_arr.append(iters)

        # Compute 01 loss?
        if classfn is not None:
            train_01_loss.append(loss01fn(train_label,
                                 cu.multi_logistic_classifier(X, w_pred, w0))/len(train_label))

        # Compute testing set loss for this iteration using fit from training set?
        if X_test is not None and y_test is not None:
            # Store test ll
            test_ll_arr.append(llfn(X_test, y_test, w_pred, w0)/len(y_test))
            if classfn is not None:
                test_01_loss.append(loss01fn(test_label,
                                 cu.multi_logistic_classifier(X_test, w_pred, w0))/len(test_label))

        # Using an adaptive step size?
        if adaptive:
            scale = 1.0/(n * np.sqrt(1.0 + iters))

        # Is it converged (is loss not changing by some %?)
        if np.fabs(ll - old_ll)/np.fabs(old_ll) > eps:
            converged = False
        else:
            converged = True

        # Store old loss, iterate
        old_ll = ll
        iters += 1

    # Uh, so what do I return?
    if savell and not (X_test is not None and y_test is not None) and classfn is None:
        return w0, w_pred, np.asarray(ll_arr), np.asarray(iter_arr)
    elif savell and (X_test is not None and y_test is not None) and classfn is None:
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), np.asarray(iter_arr)
    elif savell and classfn is not None and not (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(train_01_loss), np.asarray(iter_arr)
    elif savell and classfn is not None and (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), \
        np.asarray(train_01_loss), np.asarray(test_01_loss), np.asarray(iter_arr)
    else:
        return w0, w_pred
# end function


def stochastic_gradient_descent(grad, X, y, lam=1.0, eta = 1.0e-3, w = None, w0 = None, sparse = False,
                     eps = 5.0e-3, max_iter = 500, adaptive = True, llfn = None,
                     savell = False, X_test = None, y_test = None, multi=None, classfn=None,
                     train_label=None,test_label=None, loss01fn=None, batchsize=None):
    """
    Performs regularized stochastic gradient descent to optimize model with an update step:

    w_i^(t+1) <- w_i^(t) - eta * {-lambda*w_i^(t) + gradient}

    Model call is something like linear_model(X, w, w0, sparse=False)

    In practice, for a learning rate I use k*eta/n instead of eta where k is determined
    each step. k = 1/(1.0 + sqrt(step_number)) where step_number starts from 0.

    If batchsize is set to some int, then perform minibatch stochastic gradient descent

    Parameters
    ----------
    grad : function
        Model which computes gradient of w, w0 given X, w, w0, y
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
        which ll function to use with args y, y_hat.  Defaults to logloss_bin
    savell : bool (optional)
        whether or not to save the ll at each iteration
    multi : int (optional)
        If not None, multi sets number of classes
    classfn : func (optional)
        Function to classify.  If provided, used to compute/return 0/1 loss
    train_label : array (n x 1) (optional)
        training labels used for 0/1 loss calculation
    test_label : array (n x 1) (optional)
        testing labels used for 0/1 loss calculation
    batchsize : int (optional)
        size of minibatch

    Returns
    -------
    w : array (d x 1)
        best fit weight vector
    w0 : float
        best fit constant offset
    """

    #Define values
    if batchsize is None:
        n = 1 # For SGD, use 1 point unless using minibatch
    else:
        n = batchsize
    d = X.shape[1]

    # No multi == no multiclasses
    if multi is None:
        multi = 1

    # Init weight array and holder for convergence checks
    if w is None:
    	# No initial conditions given, assume 0s
    	w_pred = np.zeros((d,multi))
    else:
    	# Initial conditions given, use those as first w_pred
		w_pred = np.copy(w)

    if w0 is None:
        if multi is None:
            w0 = 0.0
        else: # Multiclass -> w0 becomes a vector
	        w0 = np.zeros((multi,1))

    # While not converged, do
    iters = 0
    converged = False

    # Save ll values on training set?
    if savell:
        ll_arr = []
        iter_arr = []

    # Save 01 loss?
    if classfn is not None:
        train_01_loss = []

    # Save ll values on testing set?
    if X_test is not None and y_test is not None:
        test_ll_arr = []
        if classfn is not None:
            test_01_loss = []

    # No loss function given -> use loss for a binary classifier
    if llfn is None:
        from ..validation import validation
        llfn = validation.logloss_bin

    # Dummy old super huge loss
    old_ll = 1.0e10
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

        # If using batches, get collection of indicies, else pick out one for SGD
        if batchsize is not None:
            # Create batch indices mask
            inds = np.random.permutation(X.shape[0])
            inds = inds[:batchsize]
        # No batches: straight up SGD
        else:
            inds = np.random.randint(0, high=X.shape[0], size=1)

        # Precompute gradient using w^(t), w0 since this doesn't change in a given iteration
        # using either one or a batch of samples
        wgrad, w0grad = grad(X[inds,:], y[inds], w_pred, w0, sparse=sparse)

        # Update w0 (not regularized!)
        w0 = w0 - eta * scale * w0grad

        # Loop over features to update according to gradient, learning rate
        # Do so in a vectorized manner
        w_pred = w_pred - eta * scale * (lam * w_pred + wgrad)

        # Compute loss for this fit
        ll = llfn(X, y, w_pred, w0)
        print(ll)

        if savell:
            ll_arr.append(ll/len(y))
            iter_arr.append(iters)

        # Compute 01 loss?
        if classfn is not None:
            train_01_loss.append(loss01fn(train_label,
                                 cu.multi_logistic_classifier(X, w_pred, w0))/len(train_label))

        # Compute testing set loss for this iteration using fit from training set?
        if X_test is not None and y_test is not None:
            # Store test ll
            test_ll_arr.append(llfn(X_test, y_test, w_pred, w0)/len(y_test))
            if classfn is not None:
                test_01_loss.append(loss01fn(test_label,
                                 cu.multi_logistic_classifier(X_test, w_pred, w0))/len(test_label))

        # Using an adaptive step size?
        if adaptive:
            scale = 1.0/(n * np.sqrt(1.0 + iters))

        # Is it converged (is loss not changing by some %?)
        if np.fabs(ll - old_ll)/np.fabs(old_ll) > eps:
            converged = False
        else:
            converged = True

        # Store old loss, iterate
        old_ll = ll
        iters += 1

    # Uh, so what do I return?
    if savell and not (X_test is not None and y_test is not None) and classfn is None:
        return w0, w_pred, np.asarray(ll_arr), np.asarray(iter_arr)
    elif savell and (X_test is not None and y_test is not None) and classfn is None:
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), np.asarray(iter_arr)
    elif savell and classfn is not None and not (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(train_01_loss), np.asarray(iter_arr)
    elif savell and classfn is not None and (X_test is not None and y_test is not None):
        return w0, w_pred, np.asarray(ll_arr), np.asarray(test_ll_arr), \
        np.asarray(train_01_loss), np.asarray(test_01_loss), np.asarray(iter_arr)
    else:
        return w0, w_pred
# end function