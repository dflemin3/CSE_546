# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:26:17 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu
"""

from __future__ import print_function, division
import os, struct
from array import array as pyarray
import numpy as np
from numpy import array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10),
               path="/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data"):
    """
    Loads MNIST files into 3D numpy arrays

    Credit: http://g.sweyla.com/blog/2012/mnist-numpy/

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images.reshape(len(images),784), labels

if __name__ == "__main__":
    # Test data loading, make sure shape is correct
    images, labels = load_mnist(dataset='training')
    
    print(images.shape,labels.shape)