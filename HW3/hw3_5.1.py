# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 5.1 of CSE 546 HW3
"""

from __future__ import print_function, division
import numpy as np
import sys
import os
sys.path.append("..")
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import DML.clustering.kmeans as kmeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Flags to control functionality
show_plots = True
save_plots = False
k = 16
verbose = True

# Load in MNIST data
#print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='testing')

print("Running k-means...")
y_hat, mu_hat, rec_err, iter_arr = kmeans.kmeans(X_train, k=k, verbose=verbose)

if show_plots:
    fig, ax = plt.subplots()

    ax.plot(iter_arr,rec_err,"o-",lw=2,color="blue")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reconstruction Error")

    plt.show()

    fig, axes = plt.subplots(nrows=4,ncols=4)

    if save_plots:
        fig.savefig("k_16_rec_err.pdf")

if show_plots:
    for ii, ax in enumerate(axes.flatten()):

        # Diplay ith eigenvector
        image = mu_hat[ii,:]

        # Plot reconstructed image
        ax.imshow(image.reshape((28, 28)), cmap='binary', interpolation="nearest")
        ax.text(0.95, 0.05, 'n = %d' % (ii+1), ha='right',
                transform=ax.transAxes, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if save_plots:
        fig.savefig("k_16_means.pdf")
