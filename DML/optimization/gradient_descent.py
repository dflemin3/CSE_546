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
import sys
from ..classification import classifier_utils as cu


def gradient_descent(grad, X, y, lam=0.0, eta = 1.0e-3, w = None, w0 = None, sparse = False,
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
        l2 regularization constant.  Defaults to 0 (no regularization)
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
            ll_arr.append(ll)
            iter_arr.append(iters)

        # Compute 01 loss?
        if classfn is not None:
            train_01_loss.append(loss01fn(train_label,
                                 classfn(X, w_pred, w0)))

        # Compute testing set loss for this iteration using fit from training set?
        if X_test is not None and y_test is not None:
            # Store test ll
            test_ll_arr.append(llfn(X_test, y_test, w_pred, w0))
            if classfn is not None:
                test_01_loss.append(loss01fn(test_label,
                                 classfn(X_test, w_pred, w0)))

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
                     train_label=None,test_label=None, loss01fn=None, batchsize=None,
                     nout=15000):
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
    nout : int (optional)
        Number of inner loop iterations to run before saving things like loss, 0-1 loss, etc
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

        # Randomly permute X, y
        inds = np.random.permutation(X.shape[0])
        X_per = X[inds]
        y_per = y[inds]

        # Loop over samples
        ii = 0 # Keeps track of how many samples have been used for gradient estimation
        for batch in make_batches(X_per, y_per, size=n):
            X_b = batch[0].reshape((n,batch[0].shape[-1]))
            y_b = batch[1].reshape((n,batch[1].shape[-1]))

            # Precompute gradient using w^(t), w0 since this doesn't change in a given iteration
            # using either one or a batch of samples
            wgrad, w0grad = grad(X_b, y_b, w_pred, w0, sparse=sparse)

            # Update w0 (not regularized!)
            w0 = w0 - eta * scale * w0grad

            # Loop over features to update according to gradient, learning rate
            # Do so in a vectorized manner
            w_pred = w_pred - eta * scale * (lam * w_pred + wgrad)

            # Compute loss for this fit over entire set?
            if ii % nout == 0:
                ll = llfn(X, y, w_pred, w0)
                print(ll)

                if savell:
                    ll_arr.append(ll)
                    iter_arr.append(iters)

                # Compute 01 loss?
                if classfn is not None:
                    train_01_loss.append(loss01fn(train_label,
                                         classfn(X, w_pred, w0)))

                # Compute testing set loss for this iteration using fit from training set?
                if X_test is not None and y_test is not None:
                    # Store test ll
                    test_ll_arr.append(llfn(X_test, y_test, w_pred, w0))
                    if classfn is not None:
                        test_01_loss.append(loss01fn(test_label,
                                         classfn(X_test, w_pred, w0)))

                # Using an adaptive step size?
                if adaptive:
                    scale = 1.0/(n * (1.0 + iters))

                # Is it converged (is loss not changing by some %?)
                if np.fabs((ll - old_ll)/old_ll) > eps:
                    converged = False
                else:
                    converged = True
                    break

                # Store old loss, iterate
                old_ll = ll
                iters += 1

            ii = ii + n

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


def SGD_chunks(grad, X, y, lam=0.0, eta = 1.0e-3, w = None, w0 = None, sparse = False,
                     eps = 5.0e-3, max_iter = 500, adaptive = True, llfn = None,
                     savell = True, X_test = None, y_test = None, multi=None, classfn=None,
                     train_label=None,test_label=None, loss01fn=None, batchsize=None,
                     nout=None, transform=None, alpha=None):
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
        l2 regularization constant.  Defaults to 0 (no regularization)
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
    nout : int (optional)
        Length of an epoch. Number of inner loop iterations to run before saving
        things like loss, 0-1 loss, etc.  Defaults to length of training set.
    avg_w : bool (optional)
        Return the average weight vector over the last epoch.  Defaults to False
    transform : function (optional)
        function to transform a given batch.  Defaults to None.  Requires args
        of the form (batch, X_train, alpha)
    alpha : float (optional)
        transform function parameter

    Returns
    -------
    w : array (d x 1)
        best fit weight vector
    w0 : float
        best fit constant offset
    Tons of other stuff depending on what flags are set to True and stuff
    """

    #Define values
    if batchsize is None:
        n = 100 # Default to batch sizes of ~100
    else:
        n = batchsize

    # Estimate dimension size if performing some transformation
    # by performing transformation on small subset of data
    if transform is not None:
        X_b = transform(X[:2], X, alpha)
        d = X_b.shape[1]
    else:
        d = X.shape[1]

    # Default length of epoch
    if nout is None:
        nout = X.shape[0]

    # No multi == no multiclasses
    if multi is None:
        multi = 1

    # Init weight array and holder for convergence checks
    if w is None:
        # No initial conditions given, assume 0s
        w_pred = np.zeros((d,multi))
        avg_w = np.zeros_like(w_pred)
    else:
        # Initial conditions given, use those as first w_pred
        w_pred = np.copy(w)
        avg_w = np.zeros_like(w_pred)

    if w0 is None:
        if multi is None:
            w0 = 0.0
            avg_w0 = 0.0
        else: # Multiclass -> w0 becomes a vector
            w0 = np.zeros((multi,1))
            avg_w0 = np.zeros_like(w0)

    # Save ll values on training set?
    if savell:
        ll_arr = []
        iter_arr = []
        avg_ll_arr = []

    # Save 01 loss?
    if classfn is not None:
        train_01_loss = []
        avg_train_01_loss = []

    # Save ll values on testing set?
    if X_test is not None and y_test is not None:
        test_ll_arr = []
        avg_test_ll_arr = []
        if classfn is not None:
            test_01_loss = []
            avg_test_01_loss = []

    # No loss function given -> use loss for a binary classifier
    if llfn is None:
        from ..validation import validation
        llfn = validation.logloss_bin

    # Dummy old super huge loss
    old_ll = 1.0e10
    scale = 1.0/n

    # While not converged, do
    iters = 0
    converged = False

    while not converged:
    	# Too many iterations! (this probably shouldn't happen)
    	if iters >= max_iter:
    		print("Maximum iteration threshold hit %d." % iters)
    		print("Returning current solution: Convergence not guarenteed.")
    		print("lambda = %.3e" % lam)
    		print("Old ll, current ll: %e, %e" % (old_ll, ll))
    		if saveloss:
    		    return w0, w_pred, np.asarray(loss_arr), np.asarray(iter_arr)
    		else:
    		    return w0, w_pred

        # Randomly permute X, y
        inds = np.random.permutation(X.shape[0])
        X_per = X[inds]
        y_per = y[inds]

        # Loop over samples (run an epoch)
        ii = 0 # Keeps track of how many samples have been used for gradient estimation
        avg_w = np.zeros_like(w_pred) # zero out epoch-averaged weight vector
        avg_w0 = np.zeros_like(w0)
        for batch in make_batches(X_per, y_per, size=n):
            X_b = batch[0].reshape((n,batch[0].shape[-1]))
            y_b = batch[1].reshape((n,batch[1].shape[-1]))

            # Transform data on the fly?
            if transform is not None:
                X_b = transform(X_b, X, alpha)

            # Precompute gradient using w^(t), w0 since this doesn't change in a given iteration
            # using either one or a batch of samples
            wgrad, w0grad = grad(X_b, y_b, w_pred, w0, sparse=sparse)

            # Update w0 (not regularized!)
            w0 = w0 - eta * scale * w0grad

            # Update features using gradient
            w_pred = w_pred - eta * scale * (lam * w_pred + wgrad)

            # Update epoch averaged weight vectors
            avg_w0 += w0*n/len(X)
            avg_w += w_pred*n/len(X)

            # If epoch is complete, compute losses for this fit
            if (ii+n) % nout == 0 or iters == 0:

                ll = 0.0
                ll_avg = 0.0
                loss_01 = 0.0
                loss_01_avg = 0.0
                for batch in make_batches(X, y, label=train_label, size=n):
                    X_b = batch[0].reshape((n,batch[0].shape[-1]))
                    y_b = batch[1].reshape((n,batch[1].shape[-1]))
                    y_l = batch[2].reshape((n,batch[2].shape[-1]))

                    # Transform data on the fly?
                    if transform is not None:
                        X_b = transform(X_b, X, alpha)

                    ll += llfn(X_b, y_b, w_pred, w0)*n/len(X)
                    ll_avg += llfn(X_b, y_b, avg_w, avg_w0)*n/len(X)

                    if classfn is not None:
                        loss_01 += loss01fn(y_l,classfn(X_b, w_pred, w0))*n/len(X)
                        loss_01_avg += loss01fn(y_l,classfn(X_b, avg_w, avg_w0))*n/len(X)

                print(ll)
                if savell:
                    ll_arr.append(ll)
                    avg_ll_arr.append(ll_avg)
                    iter_arr.append(iters)

                    if classfn is not None:
                        train_01_loss.append(loss_01)
                        avg_train_01_loss.append(loss_01_avg)

                # Compute testing set loss for this iteration using fit from training set?
                if X_test is not None and y_test is not None:
                    ll = 0.0
                    ll_avg = 0.0
                    loss_01 = 0.0
                    loss_01_avg = 0.0
                    for batch in make_batches(X_test, y_test, label=test_label, size=n):
                        X_b = batch[0].reshape((n,batch[0].shape[-1]))
                        y_b = batch[1].reshape((n,batch[1].shape[-1]))
                        y_l = batch[2].reshape((n,batch[2].shape[-1]))

                        # Transform data on the fly?
                        if transform is not None:
                            X_b = transform(X_b, X, alpha)

                        ll += llfn(X_b, y_b, w_pred, w0)*n/len(X_test)
                        ll_avg += llfn(X_b, y_b, avg_w, avg_w0)*n/len(X_test)

                        if classfn is not None:
                            loss_01 += loss01fn(y_l,classfn(X_b, w_pred, w0))*n/len(X_test)
                            loss_01_avg += loss01fn(y_l,classfn(X_b, avg_w, avg_w0))*n/len(X_test)

                    # Store test ll
                    test_ll_arr.append(ll)
                    avg_test_ll_arr.append(ll_avg)

                    if classfn is not None:
                        test_01_loss.append(loss_01)
                        avg_test_01_loss.append(loss_01_avg)

                # Using an adaptive step size?
                if adaptive:
                    scale = 1.0/(n * np.sqrt(1.0 + iters))

                # Is it converged (is loss not changing by some %?)
                if np.fabs((ll - old_ll)/old_ll) > eps:
                    converged = False
                else:
                    converged = True
                    break

                # Store old loss, iterate
                old_ll = ll
                iters += 1

            ii = ii + n

    # Uh, so what do I return? ... everything
    return w0, avg_w0, w_pred, avg_w, np.asarray(ll_arr), \
    np.asarray(avg_ll_arr), np.asarray(test_ll_arr), np.asarray(avg_test_ll_arr), \
    np.asarray(train_01_loss), np.asarray(avg_train_01_loss), \
     np.asarray(test_01_loss), np.asarray(avg_test_01_loss), np.asarray(iter_arr)
    """
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
    """
# end function


def make_batches(X, y, label=None,size=1):
    """
    Create list of data batches of a given size.  Useful for making batches to iterate over
    using gradient descent.

    Parameters
    ----------
    X : array (n x d)
        input data
    y : array (n x 1)
        labeled data
    label : array (n x 1) (optional)
        another array like y to yield.  Defaults to None
    size : int (optional)
        batch size

    Returns
    -------
    batches : list
        list of the form [(X_batch, y_batch), ...]
    """

    if label is None:
        for i in range(0, len(y), size):
            yield (X[i:i + size], y[i:i + size])
    else:
        for i in range(0, len(y), size):
            yield (X[i:i + size], y[i:i + size], label[i:i + size])
# end function
