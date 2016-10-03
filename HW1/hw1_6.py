#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:28:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves questions 6.1 and 6.2 of CSE 546 HW1
"""

from __future__ import print_function, division
import numpy as np
import mnist_utils as mu
import ridge_utils as ri
#import regression_utils as ru

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


def mnist_ridge_thresh(X, y, lam = 1, num = 20, minthresh=-1.0e0, 
                       maxthresh=2.0e0):
    """
    This function finds the best w (dot) x criteria for labeling a MNIST
    digit as a 2.  For the purposes of this classification, 2 -> 1, while
    everything else -> 0
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    
    """
    
    # Filter for 2s for binary classifier "truth"
    y_true = mnist_two_filter(y)
    
    # Mask corresponding to truth
    mask = (y_true == 1)
    
    # Make array of thresholds to testout
    thresh = np.linspace(minthresh,maxthresh,num)
    
    res_01 = np.zeros_like(thresh)
    
    # Loop over thresholds, evaluate model, see which is best
    for ii in range(len(thresh)):
        print("Threshold iteration",ii)
        w0, w, y_hat = ri.ridge_bin_class(X, y, lam=lam, thresh=thresh[ii])
        print("Number of 2s:",np.sum(y_hat))
        
        # Heuristic is (correct - false positives)/(truth)
        # Number correct minus number of false positives
        tmp = np.sum(y_hat[mask])# - np.sum(y_hat[~mask])
        res_01[ii] = tmp/len(mask)
    
    # Get best threshold, return it
    best_ind = np.argmax(res_01)  
    
    return thresh[best_ind], res_01[best_ind]
# end function

    
if __name__ == "__main__":
    
    # Flags to control functionality
    find_best_thresh = True
    
    
    # Best threshold from previous running of script
    """
    From a previous grid search:
    Best wx threshold: 0.421
    Best 0/1 error: 0.067
    """
    best_thresh = 0.421
    
    # Load in MNIST training data
    print("Loading MNIST Training data...")
    X, y = mu.load_mnist(dataset='training')
    
    if find_best_thresh:
        # Find the best threshold base on 0/1 error
        best_thresh, best_01 = mnist_ridge_thresh(X, y)
        print("Best wx threshold: %.3lf" % best_thresh)
        print("Best 0/1 error: %.3lf" % best_01)
    