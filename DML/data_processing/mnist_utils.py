# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:26:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu
"""

from __future__ import print_function, division
import os
import pickle
import numpy as np

def load_mnist(dataset="training", path="../Data"):
    """
    Loads MNIST files into numpy arrays from pickle files.

	Parameters
	----------
	dataset : str
		Which set you wish to load: training or testing
	path : str
		directory where MNIST .pkl files are located

	Returns
	-------
	images : array
		MNIST image arryas (n x d)
	labels : array
		MNIST label arrays (n x 1)
    """
    if dataset.lower() == "training":
        images = os.path.join(path, 'mnist_training_images.pkl')
        labels = os.path.join(path, 'mnist_training_labels.pkl')
    elif dataset.lower() == "testing":
        images = os.path.join(path, 'mnist_testing_images.pkl')
        labels = os.path.join(path, 'mnist_testing_labels.pkl')
    else:
    	raise ValueError("Dataset must be 'testing' or 'training'")

    # Open pickle files
    with open(images, "rb") as handle:
    	images = pickle.load(handle)
    with open(labels, "rb") as handle:
    	labels = pickle.load(handle)

    return images, labels
# end function


def mnist_filter(y, filterby=2):
    """
    Takes in a MNIST label vector and sets all instances of filterby to 1 and
    everything else to 0 to train a binary classifier.

    Parameters
    ----------
    y : array (n x 1)
    filterby : int (optional)
        Which number to set to 1 for a binary classifier.  Defaults to 2

    Returns
    -------
    y_filt : array (n x 1)
        but masked!
    """
    mask = (y == filterby)
    y_filt = np.zeros_like(y)
    y_filt[mask] = 1

    return y_filt
#end function