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


# Flags to control functionality
show_plots = False
save_plots = False
k = 16
verbose = True

# Load in MNIST data
print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='testing')
X_test, y_test = mu.load_mnist(dataset='training')

print("Running k-means...")
y_cluster, mu_hat, rec_err, iter_arr = kmeans.kmeans(X_train, k=k, verbose=verbose)

# Now classifiy
classifier = kmeans.k_mapper(y_cluster, y_train)
y_hat_train = kmeans.k_classify(X_train, mu_hat, classifier)
y_hat_test = kmeans.k_classify(X_test, mu_hat, classifier)

print(val.loss_01(y_train,y_hat_train))
print(val.loss_01(y_test,y_hat_test))

# Compute number of assignments in descending order
counts = np.bincount(y_cluster.squeeze())
x_counts = np.arange(k)

# Sort in descending order
inds = np.argsort(counts)[::-1]

if show_plots:
    # Plot number of assignments for each center in descending order

    fig, ax = plt.subplots()

    ax.plot(counts[inds],"o")
    plt.locator_params(axis='x',nbins=k)
    #ax.xaxis.set_ticks(x_counts[inds])
    ax.set_xticklabels(x_counts[inds])

    ax.set_xlabel("Cluster Number")
    ax.set_ylabel("Number of Assignments")

    plt.show()

    if save_plots:
        fig.savefig("k_16_num_assignments.pdf")

if show_plots:
    # Plot the reconstruction error vs iteration
    fig, ax = plt.subplots()

    ax.plot(iter_arr,rec_err,"o-",lw=2,color="blue")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reconstruction Error")

    plt.show()

    if save_plots:
        fig.savefig("k_16_rec_err.pdf")

if show_plots:
    fig, axes = plt.subplots(nrows=4,ncols=4)

    # Plot the centroids!
    for ii, ax in enumerate(axes.flatten()):

        # Diplay ith centroid
        image = mu_hat[inds[ii],:]

        # Plot reconstructed image
        ax.imshow(image.reshape((28, 28)), cmap='binary', interpolation="nearest")
        ax.text(0.95, 0.05, 'n = %d' % counts[inds[ii]], ha='right',
                transform=ax.transAxes, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if save_plots:
        fig.savefig("k_16_means.pdf")
