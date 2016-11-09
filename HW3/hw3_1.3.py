# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 1.3 of CSE 546 HW3
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.pca.pca as pca
import DML.data_processing.mnist_utils as mu
import DML.data_processing.normalization as norm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Flags to control functionality
show_plots = True
save_plots = True
use_one_digit = False

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=12, center=True)

# Fit model
print("Fitting PCA model...")
PCA.fit(X_train)

#####################################################
#
# 1.3.1: Eigenvectors and you: Plot first 10
#
#####################################################

if show_plots:

    fig, axes = plt.subplots(nrows=4,ncols=4)

    for ii, ax in enumerate(axes.flatten()):

        # Diplay ith eigenvector
        image = PCA.components[:,ii]

        # Plot reconstructed image
        ax.imshow(image.reshape((28, 28)), cmap='binary', interpolation="nearest")
        ax.text(0.95, 0.05, 'n = %d' % (ii+1), ha='right',
                transform=ax.transAxes, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if save_plots:
        fig.savefig("eigendirections.pdf")
