# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains routines to be used for unsupervised k-means clustering.  Can
also pass labels for supervised classification, if you're into that sort of
thing.
"""

from __future__ import print_function, division

import numpy as np
import random
from scipy.spatial.distance import cdist
from ..validation import validation as val


def kmeans(X, k, verbose=False, max_iters=500):
    """
    Perform k-means clustering where the EM algorithm assigns all points to once
    of k clusters.

    Parameters
    ----------
    X : array (n x d)
        training data
    k : int
        number of clusters
    verbose : bool (optional)
        whether or not to perform a ton of other error estimates and output them
        pretty much the major diagnostic flag. Defaults to False.
    max_iters : int (optional)
        maximum number of iterations to perform.  Defaults to 500.

    Returns
    -------
    y_hat : array (n x 1)
        prediction vector of cluster labels in [0,k-1]
    mu : array (k x d)
        centroids
    """

    # Get dimensions for convenience
    n = X.shape[0]
    d = X.shape[-1]

    # Initialize mean cluster vector using k random samples
    inds = random.sample(range(0, X.shape[0]-1), k)
    mu = X[inds]

    # Init label arrays
    y_hat = np.zeros(n, dtype=int)
    y_hat_old = np.zeros_like(y_hat)

    # Init error arrays if need be
    if verbose:
        rec_err = []
        iter_arr = []

    converged = False
    iters = 0
    # Run as long as labels keep changing
    while not converged:

        if iters >= max_iters:
            print("Too many iterations!.  Returning current labels.")
            return y_hat

        # Reset old array for convergence check later on
        y_hat_old = np.copy(y_hat)

        # E step: assign data point to its closest cluster center
        # z_i = argmin_k ||x_i - mu_k||^2 (l2 norm)

        # Compute distance from each point to each class mean (n x k)
        normsq = cdist(X, mu, 'sqeuclidean')
        y_hat = np.argmin(normsq, axis=1)

        # M step: Update cluster centers
        for ii in range(k):
            mask = (y_hat == ii)
            mu[ii] = np.mean(X[mask], axis=0)

        # Compute errors and such
        if verbose:
            print(iters)
            rec_err.append(val.reconstruction_error(X, y_hat, mu))
            iter_arr.append(iters)

        # If labels didn't change, done!
        if np.array_equal(y_hat, y_hat_old):
            converged = True
            break

        iters += 1

    if verbose:
        return y_hat.reshape(X.shape[0],1), mu, np.asarray(rec_err), \
               np.asarray(iter_arr)
    else:
        return y_hat.reshape(X.shape[0],1), mu
# end function


def k_mapper(y_cluster, y_label):
    """
    Create mapping array from cluster number to label.

    Parameters
    ----------
    y_cluster : array (n x 1)
        cluster labels (from running k-means)
    y_label : array (n x 1)
        actual point labels

    Returns
    -------
    classifier : array (k x 1)
        mapping array from cluster to label

    """
    # First, create mapping from cluster label to label (like MNIST digit label)
    clusters = np.unique(y_cluster)
    classifier = np.zeros(len(clusters))

    for ii in range(len(clusters)):
        a = y_label[y_cluster == clusters[ii]]
        classifier[ii] = np.argmax(np.bincount(a))

    return classifier
# end function


def k_classify(X, mu, classifier):
    """
    Get a label for each center based on most frequent digit assigned to it.
    To classify, find closest center, then use corresponding label for said
    center.

    Parameters
    ----------
    X : array (n x d)
        data to classify
    mu : array (k x d)
        centroid matrix
    classifier : array (k x 1)
        mapping array from cluster to label

    Returns
    -------
    y_hat : array (n x 1)
        predicted label vector
    """

    # For all points in X, see which cluster they belong to and translate
    # that into a label then return
    # Compute distance from each point to each class mean (n x k)
    normsq = cdist(X, mu, 'sqeuclidean')
    y_hat = np.argmin(normsq, axis=1)

    for ii in range(len(y_hat)):
        y_hat[ii] = classifier[y_hat[ii]]

    return y_hat.reshape(len(y_hat),1)
# end function
