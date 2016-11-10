"""

@author dflemin3 [David P. Fleming, University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

Nov. 2016

This file contains routines for computing kernels, like an RBF kernel.

"""

# Imports
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit, float64


def RBF(x, X, sigma=None):
    """
    Radial basis function (RBF) kernel transformation of the form:

    exp(-||x - X_j||^2 / (2 sigma^2))

    aka Gaussian aka squared exponential kernel

    Parameters
    ----------
    x : array (samples x d)
        input samples(s)
    X : array (n x d)
        input data
    sigma : float
        bandwidth parameter

    Returns
    -------
    K : array (samples x n)
        kernel transformation
    """

    # Compute distance between each sample and every other
    normsq = cdist(x, X, 'sqeuclidean')

    return np.exp(-normsq/(2.0*sigma*sigma))
# end function


def estimate_bandwidth(X, num = 100, scale = 10.0):
    """
    Estimate the bandwidth parameter for an RBF kernel via the following
    procedure:

    Randomly grab some rows from X, estimate their pairwise distance, take the
    mean of that quantity and scale it down by a factor of a few.

    Parameters
    ----------
    X : array (n x d)
        input data
    num : int (optional)
        number of rows to use for pairwise distances.  Defaults to 100.
    scale : float (optional)
        How much to scale final answer by.  Defaults to 10

    Returns
    -------
    sigma : float
        bandwidth parameter for rbf kernel
    """

    inds = np.random.permutation(X.shape[0])[:num]
    x = X[inds]

    return np.mean(cdist(x,X))/scale
# end function
