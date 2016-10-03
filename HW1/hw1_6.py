#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:28:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves questions 6.1 and 6.2 of CSE 546 HW1
"""

from __future__ import print_function, division
import mnist_utils as mu
import ridge_utils as ri
import regression_utils as ru

def mnist_two_filter(y):
    """
    Takes in a MNIST label vector and sets all instances of 2 to 1 and
    everything else to 0 to train a binary classifier.
    """
    mask = (y == 2)
    y[mask] = 1
    y[~mask] = 0
    return y
#end function


def mnist_ridge_thresh(X, y, bins = 10):
    """
    This function finds the best w (dot) x criteria for labeling a MNIST
    digit as a 2.  For the purposes of this classification, 2 -> 1, while
    everything else -> 0
    """
    
    return None
    
if __name__ == "__main__":
    
    # Load in MNIST training data
    X, y = mu.load_mnist(dataset='training')
    
    # Filter for 2s for binary classifier "truth"
    y_true = mnist_two_filter(y)
    
    # Mask to check stuff out
    mask = (y == 1)
    
    # Fit using ridge regression with lambda = 1
    w0, w = ri.fit_ridge(X, y)
    
    # What's it look like?
    print(ru.linear_model(X, w, w0)[mask][:10])
    