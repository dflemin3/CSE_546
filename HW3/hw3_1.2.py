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
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

# Pick some digit to make the computer draw
ind = 2222
print(y_train[ind])

# Init PCA object
PCA = pca.PCA(l=X_train.shape[-1])

# Fit model
PCA.fit(X_train)

# Using principal components, draw the reconstructed image
image = PCA.reproject(X_train[ind])

fig, ax = plt.subplots()

# Plot for 0th sample, a 7
ax.imshow(image.reshape((28, 28)), cmap='binary')
ax.text(0.95, 0.05, 'n = {0}'.format(100), ha='right',
        transform=ax.transAxes, color='green')
ax.set_xticks([])
ax.set_yticks([])

plt.show()

"""

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):

    # Init PCA object
    PCA = pca.PCA(l=(i+1))

    # Fit model
    PCA.fit(X_train)

    # Using principal components, draw the reconstructed image
    image = PCA.reproject(X_train[ind])

    # Plot for 0th sample, a 7
    ax.imshow(image.reshape((28, 28)), cmap='binary')
    ax.text(0.95, 0.05, 'n = {0}'.format(i + 1), ha='right',
            transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
"""
