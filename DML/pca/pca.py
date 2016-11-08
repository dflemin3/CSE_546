"""

@author dflemin3 [David P. Fleming, University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

Nov. 2016

This file contains routines for Principal Component Analysis (PCA) and related
diagnostics, like making scree plots.

"""

# Imports
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from ..data_processing import normalization as norm


def fit_pca(X, l=None, center=True):
    """
    Perform PCA by solving the SVD problem where

    X = USV^T such that X_pred = USV^T

    where only the first l components are retained.

    Note that this performs better when X has been scaled, so do that by setting
    center = True

    Parameters
    ----------
    X : array (n x d)
        centered input data array
    l : int (optional)
        number of principal components to return.  Defaults to 50
    svd : bool (optional)
        whether to instead perform SVD to solve pca.  Defaults to False

    Returns
    -------
    u : array (n x l)
    s : array (1 x l)
    v : array  (d x l)
    """

    # Center the data?
    if center:
        X_in, X_mean = norm.center(X)
    else:
        X_in = X

    # If no l is given, return all d principal components
    if l is None:
        l = X_in.shape[1]

    # Make sure shapes are correct
    assert X.shape[0] > X_in.shape[1], "This PCA inplementation is for for N > D"

    # Solve PCA using SVD approach
    # Use thin SVD as for N > D, last N-D cols of U are irrelevant
    u, s, v = np.linalg.svd(X_in, full_matrices=False)

    if center:
        return u[:,:l], s[:l].reshape((1,l)), v[:,:l], X_mean
    else:
        return u[:,:l], s[:l].reshape((1,l)), v[:,:l]
# end function


class PCA(object):
    """
    PCA object which, surprisingly, performs PCA on input data written with
    sklearn-ish style using thin SVD.
    """

    def __init__(self,l=None):
        self.u = None # SVD output
        self.s = None # SVD output
        self.v = None # SVD output
        self.l = l # Number of principal components used in calculations
        self.components = None # Principal components
        self.X_mean = None # Mean of data (works better when data is centered?)

    def fit(self,X, center=False):
        """
        Fit the pca model using pca.fit (see parameters in that function). This
        function stores the output of SVD and the mean of the data since the
        fitting process in general works better on centered data.  The mean is
        used later on to re-transform data back into physical space.
        """
        # Fit for components retaining all principal components
        if center:
            self.u, self.s, self.v, self.X_mean = fit_pca(X, l=None, center=center)
        else:
            self.u, self.s, self.v = fit_pca(X, l=None, center=center)
            self.X_mean = 0.0

        # Now set principal components
        self.components = self.v
    # end function

    def transform(self,X):
        """
        Transform to projected principal component space using first l principal
        components given by

        X_trans = US or XW (Murphy 12.58)

        for principal components W

        Parameters
        ----------
        X : array (n x d)

        Returns
        -------
        X : array (n x l)
        """
        return X.dot(self.components[:,:self.l])
    # end function


    def inverse_transform(self,X):
        """
        Transfrom from principal component space back to physical space using
        first l principal components given by

        X_pred = USV^T (Murphy 12.59)

        where US is the transformed input data (denoted here as X)

        Parameters
        ----------
        X : array (n x l)
            Transformed data (like u * s)

        Returns
        -------
        X : array (n x d)
        """
        return X.dot(self.components[:,:self.l].T) + self.X_mean
    # end function


    def reproject(self, X):
        """
        Combine transform, inverse_transform methods into one handy function.
        """
        return self.inverse_transform(self.transform(X - X.mean(axis = 0)))
    # end function


    def scree(self):
        """
        Computes the fractional reconstruction error for the first l principal
        components according to the following formula

        1 - sum(i = 1, l) lambda_i / sum(i = 1, d) lambda_i

        Parameters
        ----------

        Returns
        -------
        err : vector (1 x l)
            fractional reconstruction error
        """

        # Precompute sum(i = 1, d) lambda_i term
        denom = np.sum(self.s**2)

        return 1.0 - (np.cumsum(self.s[:,:self.l]**2)/denom)
    # end function
# end class
