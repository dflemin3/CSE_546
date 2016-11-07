# -*- coding: utf-8 -*-
"""
Created Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This file contains utility functions for cleaning and normalizing data
"""
from __future__ import print_function, division
import numpy as np


def normal_scale(X):
    """
    Scale input data to have 0 mean and unit standard deviation in the feature
    direction.

    I think this is broken...

    Parameters
    ----------
    X : array (n x d)
        input data

    Returns
    X_scale : array (n x d)
    """

    return (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
# end function


def center(X):
    """
    Center the data by subtracting off the mean along each feature.

    Parameters
    ----------
    X : array (n x d)
        input data

    Returns
    X_scale : array (n x d)
    X_mean : array (d x 1)
        mean of features
    """
    X_mean = np.mean(X, axis=0)
    return X - X_mean, X_mean
