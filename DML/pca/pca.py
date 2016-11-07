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
    vT : array  (l x l)
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
    assert X.shape[0] > X_in.shape[1], "This PCA inplementation fors for N > D"

    # Solve PCA using SVD approach
    # Use thin SVD as for N > D, last N-D cols of U are irrelevant
    u, s, v = np.linalg.svd(X_in, full_matrices=False)

    if center:
        return u[:,:l], s[:l].reshape((1,l)), v.T[:,:l], X_mean
    else:
        return u[:,:l], s[:l].reshape((1,l)), v.T[:,:l]
# end function


class PCA(object):
    """
    PCA wrapper object.  More docs soon (TM)
    """

    def __init__(self,l=None):
        self.u = None
        self.s = None
        self.vT = None
        self.l = l
        self.components = None
        self.X_mean = None

    def fit(self,X):
        """
        Fit the pca model using pca.fit (see parameters in that function)
        """
        # Fit for eigenvectors and so on
        self.u, self.s, self.vT, self.X_mean = fit_pca(X, l=self.l, center=True)

        # Now set principal components
        self.components = self.vT / self.s * np.sqrt(len(X))
    # end function

    def transform(self,X):
        """
        Transform to projected principal component space.

        Parameters
        ----------
        X : array (n x d)

        Returns
        -------
        X : array (n x l)
        """
        return self.u * self.s
    # end function


    def inverse_transform(self,X):
        """
        Transfrom from principal component space back to physical space.

        Parameters
        ----------
        X : array (n x l)

        Returns
        -------
        X : array (n x d)
        """
        return X.dot((self.s ** 2 * self.components / len(X)).T) + self.X_mean
    # end function
# end class
