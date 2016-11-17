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
#from numba import jit, float64

def RBF(x, X, v=None, sigma=1.0):
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
    v : array (d x n)
        transformation matrix (Not used here)
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


#@jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:], float64),
#nopython=True, cache=True)
def fourier_rbf(x, X, v, sigma=1.0):
    """
    Fourier approximation to a radial basis function (RBF) kernel transformation
     of the form:

    h_j(x) = sin(vx/sigma)

    Parameters
    ----------
    x : array (samples x d)
        input samples(s)
    X : array (n x d)
        input data
    v : array (d x n)
        transformation matrix (required)
    sigma : float
        bandwidth parameter

    Returns
    -------
    K : array (samples x n)
        kernel transformation
    """
    return np.sin(np.dot(x,v)/sigma)
# end function


def generate_fourier_v(d, k):
    """
    Generate a matrix with d by k matrix where each of the k columns has all
    its coordinates independently sampled form the standard normal distribution.

    Parameters
    ----------
    d : int
        number of rows
    k : int
        number of columns

    Returns
    -------
    v : array (d x k)
        transformation matrix
    """
    return np.random.normal(size=(d,k))


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
