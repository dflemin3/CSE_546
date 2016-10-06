#!/usr/bin/env python2
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

def fit_lasso_sparse(X,y,lam=1.0):
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
    lam : float
    	regularization tuning parameter
    
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
        
    #Convergence condition
    eps = 1.0e-6
    w_old = np.zeros((d,1))
    w_pred = np.copy(w)
    
    first_iter = True
    
    # While not converged, do
    while first_iter or (np.sum(np.fabs(w_pred - w_old)) < eps):
    	first_iter = False
    
        #Store for convergence test 
        w_old = np.copy(w_pred)
        
        #Compute w_0
        w_0 = np.sum(y - X.dot(w_pred))/n
            
        #Compute a_k: d x 1 summing over columns
        a = np.zeros_like(w)
        c = np.zeros_like(w)
         
        for k in range(0,d):
            
            # Compute a (d x 1)
            a[k] = 2.0 * X[:,k].T.dot(X[:,k])
            
            alpha = np.zeros((d,d))
            np.fill_diagonal(alpha, 1)
            alpha[k,k] = 0 # Ignore where j != k
            alpha = X.dot(alpha.dot(w_pred)) + w_0
            
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
    #end while
    
    #Return as row array
    y_hat = np.zeros(y.shape)
    y_hat = X.dot(w_pred) + w_0
    return w_pred, w_0, y_hat
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


def check_solution(X,y,w_pred,w_0_pred,lam):
    """    
    See if the computed solution w_pred, w_0_pred for a given lambda l
    is correct.  That occurs when:
    
    test = 2X^T(Xw_pred + w_0_pred - y)
    
    and the zero indicies are lesser in magnitude than lambda
    
    Parameters
    ----------
    X : n x d matrix of data
    y : N x 1 vector of response variables
    w_pred : predicted d dimensions weight vector
    w_0_pred : scalar offset term
    l : regularization tuning parameter
    
    Returns
    -------
    ans : bool
        whether or not the solution passes the test criteria
    """
    
    eps = 1.0e-8
    test = 2.0*X.T.dot(X.dot(w_pred) + w_0_pred - y)
    
    #Mask values 
    mask = (np.fabs(w_pred) < eps) # Find 0s
    
    # Solution correct if |non-zero entries| < lam for all non-zero entries
    test_mask = np.fabs(test[~mask]) > lam
    
    if np.sum(test_mask) > 0:
    	return False
    else:
    	return True
# end function


# Test it out!
if __name__ == "__main__":
    
    # Generate some fake data
    w, X, y = ru.generate_norm_data(10000,5,10)
    
    # What's its shape?
    print(w.shape,X.shape,y.shape)
    
    print("Performing LASSO regression...")
    print(fit_lasso_sparse(X,y,lam=500.0))
    print(w)
    print(check_solution(X,y,w,0,lam=500.0))
