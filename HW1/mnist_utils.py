# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:26:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu
"""

from __future__ import print_function, division
import os
import pickle

def load_mnist(dataset="training",
               path="/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data"):
    """
    Loads MNIST files into numpy arrays

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


if __name__ == "__main__":
    # Test data loading, make sure shape is correct
    images, labels = load_mnist(dataset='training')

    print(images.shape,labels.shape)