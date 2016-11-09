# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 1.4 of CSE 546 HW3
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
save_plots = True
plot_digits = True
plot_recon = True

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=50, center=True)

# Fit model
print("Fitting PCA model...")
PCA.fit(X_train)

#####################################################
#
# 1.4.1: Visualize 6 digits
#
#####################################################

# Define digit indicies
dig_inds = [0,111,1112,224,2222,3333]


if plot_digits:

    fig, axes = plt.subplots(nrows=3,ncols=2)

    for ii, ax in enumerate(axes.flatten()):

        # Diplay ith eigenvector
        image = X_train[dig_inds[ii]]

        # Plot reconstructed image
        ax.imshow(image.reshape((28, 28)), cmap='binary', interpolation="nearest")
        ax.text(0.95, 0.05, 'n = %d' % y_train.squeeze()[dig_inds[ii]], ha='right',
                transform=ax.transAxes, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if save_plots:
        fig.savefig("mnist_digits.pdf")

if plot_recon:

    # Now loop over allll those digits and plot their reconstructions
    k = [2,5,10,20,50,100]
    for ii in range(len(dig_inds)):
        fig, axes = plt.subplots(nrows=3,ncols=2)

        # Now loop number of components per reconstruction
        for jj, ax in enumerate(axes.flatten()):

            # Select first k[jj] components for reconstruction
            PCA.l = k[jj]

            image = PCA.reproject(X_train[dig_inds[ii]])

            # Plot reconstructed image
            ax.imshow(image.reshape((28, 28)), cmap='binary', interpolation="nearest")
            ax.text(0.95, 0.05, 'k = %d' % k[jj], ha='right',
                    transform=ax.transAxes, color='red')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()

        if save_plots:
            name = str(y_train.squeeze()[dig_inds[ii]]) + "_recon_mnist.pdf"
            fig.savefig(name)
