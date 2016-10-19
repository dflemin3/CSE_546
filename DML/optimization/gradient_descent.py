# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file implements various functions and algorithms pertaining optimization via
gradient descent.
"""

from __future__ import print_function, division
import numpy as np
from ..classification import classifier_utils as cu


def gradient_descent(model, X, y, lam=1.0, eta = 1.0e-2, w = None, w0 = None, sparse = False,
                     eps = 1.0e-1, max_iter = 5000, adaptive = False):
    """
    Performs regularized batch gradient descent to optimize model with an update step:

    w_i^(t+1) <- w_i^(t) + eta * {-lambda*w_i^(t) + sum_j[x_i^j(y^j - y_hat^j)]}

    Model call is something like linear_model(X, w, w0, sparse=False)

    In practice, for a learning rate I use k*eta/n instead of eta where k is determined
    each step.  Initially, k = 1 and if we move downhill, set k = k * 2 to increase step
    size.  If we move uphill, decrease stepsize via k = k / 2

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
    	w_old = np.copy(w_pred) + 10000000. # offset
    else:
    	# Initial conditions given, use those as first w_pred
		w_pred = np.copy(w)
		w_old = np.copy(w_pred) + 10000000. # offset

    if w0 is None:
		w0 = 0.0

    # While not converged, do
    iters = 0
    converged = False

    # Precompute X transpose since it doesn't change
    XT = X.T

    # Dummy old loss
    old_loss = 1.0e10
    scale = 1.0/n

    while not converged:
    	# Too many iterations!
    	if iters >= max_iter:
    		print("Maximum iteration threshold hit %d." % iters)
    		print("Returning current solution: Convergence not guarenteed.")
    		print("lambda = %.3e" % lam)
    		print("Old loss, current loss: %.3e, %.3e" % (old_loss, loss))
    		return w0, w_pred

        # Precompute y_hat using w^(t), w0 since this doesn't change in a given iteration
        y_hat = model(X, w_pred, w0, sparse=sparse)
        arg = y - y_hat

        # Update w0 (not regularized!)
        w0 = w0 + eta * scale * np.sum(arg)

        # Loop over features to update according to gradient, learning rate
        # Do so in a vectorized manner
        w_pred = w_pred + eta * scale * (-lam * w_pred + XT.dot(arg))

        # Adapt step size: if square loss new > old, decrease stepsize, increase otherwise
        loss = np.sum(arg*arg)
        print(loss)

        if adaptive:
            if loss > old_loss: # Moving uphill
                scale /= 2.0
            else: # Moving downhill
                scale *= 1.01

        # Is it converged (is new cost not much different than old cost?)
        if np.fabs(loss - old_loss) > eps:
            converged = False
        else:
            converged = True

        # Store old_cost, iterate
        old_loss = loss
        iters += 1

    return w0, w_pred
# end function













