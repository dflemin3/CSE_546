# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 1.2 of CSE 546 HW3
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

# Load in MNIST data
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')

# Center data
#X_mean = np.mean(X_test, axis=0)
#X_scaled = norm.center(X_test)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):

    # Init PCA object
    PCA = pca.PCA(l=(i+1))

    # Fit model
    PCA.fit(X_test)

    # Transform to low-dimensional space
    Xtrans = PCA.transform(X_test)

    # Transform back to "physical" space
    image = PCA.inverse_transform(Xtrans)

    # Plot for 0th sample, a 7
    ax.imshow(image[0].reshape((28, 28)), cmap='binary')
    ax.text(0.95, 0.05, 'n = {0}'.format(i + 1), ha='right',
            transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
