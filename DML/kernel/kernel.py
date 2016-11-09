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


def RBF(x, X, sigma):
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
    norm = cdist(x,X)

    return np.exp(-(norm*norm)/(2.0*sigma*sigma))
# end function
