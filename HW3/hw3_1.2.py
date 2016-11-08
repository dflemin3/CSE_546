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

# Flags to control functionality
show_plots = False

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

# Init PCA object
PCA = pca.PCA(l=50)

# Fit model
print("Fitting PCA model...")
PCA.fit(X_train, center=True)

if show_plots:
    # Pick some digit to make the computer draw
    ind = 2222
    print(y_train[ind])


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

print("Plotting scree plot...")
# Get variance explained by singular values
explained_variance = (PCA.s[:,:PCA.l] ** 2) / len(X_train)
total_var = ((PCA.s ** 2) / len(X_train)).sum()
explained_variance_ratio = explained_variance / total_var

scree = PCA.scree()
plt.plot(np.arange(PCA.l),explained_variance_ratio.squeeze(), lw=3, color="red")

# Now sklearn it
from sklearn.decomposition import PCA
skpca = PCA(n_components=50, svd_solver="full")
skpca.fit(X_train)

plt.plot(skpca.explained_variance_ratio_, lw=2,color="blue")

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
