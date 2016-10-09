# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file implements various functions and algorithms pertaining to LASSO regression.
Namely, this code contains routines which implement the coordinate descent method to
solve the LASSO problem.
"""

from __future__ import print_function, division

import numpy as np
import scipy.sparse as sp
import regression_utils as ru

def fit_lasso(X,y,lam=1.0, sparse = True, w = None, w0 = None, max_iter = 500,
			  eps = 1.0e-10):
    """
    Implimentation of the naive (un-optimized) lasso regression
    algorithm.

    Algorithm 1: Coordinate Descent Algorithm for Lasso

    while not converged do:
        w_0 <- sum_i=1_N[y_i - sum_j[w_j X_ij]]/N
        for(k [1,d]) do:
            a_k <- 2 * sum_i=1_N[X_ik ^2]
            c_k <- 2 * sum_i=1_N[X_ik (y_i - (w_0 + sum_j!=k[w_j X_ij]))]
            w_k <- (c_k + lambda)/a_k if c_k < -lambda
                    0 if c_k is between [-lambda,lambda]
                    (c_k - lambda)/a_k if c_k > lambda
        end
    end

    Parameters
    ----------
    X : array (n x d)
    	matrix of input data
    y : array (n x 1)
    	vector of response variables
    lam : float (optional)
    	regularization tuning parameter
    sparse : bool (optional)
    	whether or not X is scipy.sparse.csc. defaults to True
    w : array (d x 1) (optional)
    	optional initial conditions for weight vector
    w0 : float (optional)
    	optional initial condition for constant offset
    max_iter : int (optional)
    	maximum number of while loop iterations
    eps : float (optional)
    	convergence tolerance

    All matrices X assumed to be sparse and of the form given by
    scipy.sparse.csc matrix

    Returns
    -------
    w : numpy array
        d x 1 weight vector
    w_0 : float
        offset
    y_hat : numpy array (n x 1)
        predictions
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
		w_pred = w
		w_old = np.copy(w_pred) + 10000000. # offset


    # While not converged, do
    iters = 0
    converged = False

    while not converged:
    	# Too many iterations!
    	if iters >= max_iter:
    		print("Maximum iteration threshold hit %d." % iters)
    		print("Returning current solution: Convergence not guarenteed.")
    		print("lambda = %.3lf" % lam)
    		return w_0, w_pred

        #Store for convergence test
        w_old = np.copy(w_pred)

        #Compute w_0
        w_0 = np.sum(y - X.dot(w_pred))/n

        #Compute a_k: d x 1 summing over columns
        a = np.zeros_like(w_pred)
        c = np.zeros_like(w_pred)

        for k in range(d):

            # Compute a (d x 1)
            # Note: different behavior whether or not you're assuming
            # X is a sparse array (via scipy implementation)
            # For sparse matricies (scipy.sparse ones), X.dot(w) is MUCH
            # faster than np.dot(X,w)...like 10^4 times faster
            if sparse:
            	a[k] = (2.0 * X[:,k].T.dot(X[:,k]))[0,0]
            else:
            	a[k] = (2.0 * X[:,k].T.dot(X[:,k]))

            # Ignore where j == k
            w_pred_tmp = np.copy(w_pred)
            w_pred_tmp[k] = 0.0
            alpha = X.dot(w_pred_tmp) + w_0

            #Compute c (d x 1)
            c[k] = 2.0*X[:,k].T.dot((y-alpha))

            #Compute w_k
            if(c[k] < -lam):
                w_pred[k] = (c[k] + lam)/a[k]
            elif(c[k] >= -lam and c[k] <= lam):
                w_pred[k] = 0.0
            elif(c[k]  > lam):
                w_pred[k] = (c[k] - lam)/a[k]
            else:
                print("Error! Shouldn't ever happen.")
        #end for

        iters += 1

        # Is it converged?
        if np.any(np.fabs(w_pred - w_old) > eps):
        	converged = False
        else:
        	converged = True

    #end while

    #Return as row array
    #y_hat = np.zeros(y.shape)
    #y_hat = X.dot(w_pred) + w_0
    return w_0, w_pred
# end function


def fast_lasso_sparse(X,y,l=10,w=-999,w_0=-999):
    """
    Implimentation of the naive (un-optimized) lasso regression
    algorithm.

    Parameters
    ----------
    X : n x d matrix of data
    X_i : the ith row of X
    y : N x 1 vector of response variables
    w : d dimensions weight vector (optional)
    w_0 : scalar offset term (optional)
    l : regularization tuning parameter

    All matrices X assumed to be sparse and of the form given by
    scipy.sparse.csc matrix

    Algorithm 1: Coordinate Descent Algorithm for Lasso

    while not converged do:
        w_0 <- sum_i=1_N[y_i - sum_j[w_j X_ij]]/N
        for(k [1,d]) do:
            a_k <- 2 * sum_i=1_N[X_ik ^2]
            c_k <- 2 * sum_i=1_N[X_ik (y_i - (w_0 + sum_j!=k[w_j X_ij]))]
            w_k <- (c_k + lambda)/a_k if c_k < -lambda
                    0 if c_k is between [-lambda,lambda]
                    (c_k - lambda)/a_k if c_k > lambda
        end
    end

    Returns
    -------
    w : numpy array
        d x 1 weight vector
    w_0 : float
        offset
    y_hat : numpy array (n x 1)
        predictions
    """
    #Define values
    N = y.shape[0]
    d = X.shape[1]
    y = y.reshape(N,1)

    #If no initial conditions, assume Gaussian
    if not hasattr(w, "__len__") and w == -999:
        w = np.random.randn(d)
    if w_0 == -999:
        w_0 = np.random.randn(1)

    #Convergence condition
    eps = 1.0e-6
    w_old = np.zeros(w.shape).reshape(d,1)
    w_pred = np.copy(w).reshape(d,1)

    while(np.sqrt((w_pred - w_old).dot((w_pred - w_old).T)[0][0]) > eps):
        #Store for convergence test
        w_old = np.copy(w_pred)

        #Compute w_0
        w_0 = np.sum(y)
        w_0 -= X.dot(w_pred).sum()
        w_0 /= N

        #Compute a_k: d x 1 summing over columns
        a = 2.0*np.asarray((X.power(2).sum(axis=0).T))
        c = np.zeros(d)

        for k in range(0,d):

            alpha = np.zeros((d,d))
            np.fill_diagonal(alpha, 1)
            alpha[k,k] = 0
            alpha = X.dot(alpha.dot(w_pred)) + w_0

            #Compute c: d x 1
            c[k] = 2.0*X[:,k].T.dot((y-alpha))

            """
            #Compute c_k: d x 1
            c_sum = 0.0
            for i in range(0,N):
                #Select not k columns
                ind = [x for x in range(0,d) if x != k]
                c_sum += X[i,k]*(y[i] - (X[i,ind].dot(w_pred[ind]) + w_0))
            c[k] = 2.0*c_sum
            """

            #Compute w_k
            if(c[k] < -l):
                w_pred[k] = (c[k] + l)/a[k]
            elif(c[k] >= -l and c[k] <= l):
                w_pred[k] = 0.0
            elif(c[k]  > l):
                w_pred[k] = (c[k] - l)/a[k]
            else:
                print("Error! Shouldn't ever happen.")
        #end for
        #print w_pred
    #end while

    #Return as row array
    y_hat = np.zeros(y.shape)
    y_hat = X.dot(w_pred) + w_0
    return w_pred.T, w_0, y_hat
# end function


def compute_max_lambda(X,y):
    """
    Compute the smallest lambda for which the solution w is entirely zero
    aka the maximum lambda in a regularization path

    Parameters
    ----------
    X : array (n x d)
    	matrix of data (scipy sparse matrix)
    y : vector (n x 1)
        vector of response variables

    Returns
    -------
    lam : float
        Smallest value of lambda for which the solution w is entirely zero
    """
    y_mean = np.mean(y)
    arg = X.T.dot(y - y_mean)
    return 2.0*np.linalg.norm(arg,ord=np.inf)
# end function


def check_solution(X,y,w_pred,w_0_pred,lam, eps = 1.0e-6):
    """
    See if the computed solution w_pred, w_0_pred for a given lambda l
    is correct.  That occurs when:

    test = 2X^T(Xw_pred + w_0_pred - y)

    and the zero indicies are lesser in magnitude than lambda

    Parameters
    ----------
    X : array (n x d)
    	 matrix of data
    y : array
    	(n x 1) vector of response variables
    w_pred : array (d x 1)
    	predicted d dimensions weight vector
    w_0_pred :  array (d x 1)
    	scalar offset term
    lam : float
    	regularization tuning parameter

    Returns
    -------
    ans : bool
        whether or not the solution passes the test criteria
    """

    test = 2.0*X.T.dot(X.dot(w_pred) + w_0_pred - y)

    #Mask values to find 0s
    mask = (np.fabs(w_pred) < eps)

    # If you predict all 0s, you're probably wrong
    if int(np.sum(mask)) == len(mask):
    	return False

    # Check1: Solution correct if |zero entries| < lam for all non-zero entries
    test1_mask = np.fabs(test[mask]) > lam

    if np.sum(test1_mask) > 0:
    	check1 = False
    else:
    	check1 =  True

    # Check2: Non-zero entries should take the value -lambda * sign(w_pred)
    # In practice, check to see if they're close to each other
    eps = eps * len(w_pred)
    test2_mask=np.fabs(-lam*ru.sign(w_pred[~mask])-test[~mask])/np.fabs(w_pred[~mask]) > 0.01

    if np.sum(test2_mask) > 0:
    	check2 = False
    else:
    	check2 = True

    return (check1 & check2)

# end function


def lasso_reg_path(X, y, w_true, scale = 10., sparse = True, max_iter = 10, max_lam = None):
	"""
	Perform a regularization path to find the proper lambda regularization penalty
	for the lasso algorithm on a particular dataset.

	Parameters
    ----------
    X : array (n x d)
    	matrix of input data
    y : array (n x 1)
    	vector of response variables
    w_true : array (d x 1) (optional)
    	true weight vector for computing precision, recall
    num : float (optional)
    	each iteration, lambda decreases by 1/scale
    sparse : bool (optional)
    	whether or not X is scipy.sparse.csc. defaults to True
    max_iter : int
    	Number of iterations
    max_lam : float (optional)
    	Maximum/start lambda for reg path.  Defaults to None

    Returns
    -------
    lam_hat : float
    	optimal regularization penalty
    lams : array
    	lams over regularization path
    recall : array
    	recall as a function of lambda
    prec : array
    	precision as a function of lambda
	"""

	# Init empty lists
	recall = []
	prec = []
	lams = []

	# Find max lambda to start reg path
	if max_lam is None:
		lam = compute_max_lambda(X,y)
	else:
		lam = max_lam

	# Assume first w, w_0 are zeros
	w_0 = 0.0
	w_pred = np.zeros((X.shape[-1],1))

	for ii in range(max_iter):
		print("Fit iteration, lambda:",ii,lam)
		# Fit model using previous fit
		w_0, w_pred = fit_lasso(X, y, w=w_pred, w0=w_0, lam=lam, sparse=sparse)

		# Store lam, prec, recall
		lams.append(lam)
		if w_true is not None:
			recall.append(ru.recall_lasso(w_true, w_pred))
			prec.append(ru.precision_lasso(w_true, w_pred))

		# Scale lam for next iteration
		lam /= scale

	# Done!
	if w_true is not None:
		return np.array(lams), np.array(recall), np.array(prec)
	else:
		return np.array(lams)
# end function


# Test it out!
if __name__ == "__main__":

    # Generate some fake data
    n = 100000
    d = 10
    k = 5
    lam = 1000.0
    sparse = False
    w, X, y = ru.generate_norm_data(n,k,d,sparse=sparse)

    # What should the maximum lambda in a regularization step be?
    print("Lambda_max:",compute_max_lambda(X,y))

    print("Performing LASSO regression...")
    w_0_pred, w_pred = fit_lasso(X,y,lam=lam,sparse=sparse)
    print("w_pred:",w_pred)
    print(w)

    # Was the predicted solution correct?
    print(check_solution(X,y,w_pred,w_0_pred,lam=lam))
