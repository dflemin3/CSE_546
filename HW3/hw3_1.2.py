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
show_plots = True
save_plots = True
use_one_digit = False

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

# Training using only one digit?
if use_one_digit:
    mask = y_train.squeeze() == 5
    X_train = X_train[mask]
    print(X_train.shape)
    y_train = y_train[mask]

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=50, center=True)

# Fit model
print("Fitting PCA model...")
PCA.fit(X_train)

#####################################################
#
# 1.2.1: Eigenvalues and you
#
#####################################################

eval_list = np.asarray([1, 2, 10, 30, 50]) - 1 # my 1st eigenvalue is evals[0]
for ev in eval_list:
    print("%d eigenvalue: %lf" % ((ev+1),PCA.evals[ev]))
print("Sum of eigenvalues: %lf" % np.sum(PCA.evals))

#####################################################
#
# 1.2.2: Fractional reconstruction error from 1-50
# eigenvalues
#
#####################################################

frac_err_50 = PCA.frac_reconstruction_error()
x_arr = np.arange(len(frac_err_50)) + 1 # to shift from 0-49 -> 1-50

# Plot fractional reconstruction error for 1-50 components
if show_plots:

    fig, ax = plt.subplots()

    ax.plot(x_arr, frac_err_50, "o-", lw=3, color="blue")

    # Format plot
    ax.set_xlabel("k")
    ax.set_ylabel("Fractional Reconstruction Error")
    ax.set_ylim((0,1))
    fig.tight_layout()

    plt.show()

    if save_plots:
        fig.savefig("fractional_rec_error.pdf")

# Plot fractional reconstruction error for 2-50 components
if show_plots:

    fig, ax = plt.subplots()

    ax.plot(x_arr[1:], frac_err_50[1:], "o-", lw=3, color="blue")

    # Format plot
    ax.set_xlabel("k")
    ax.set_ylabel("Fractional Reconstruction Error")
    ax.set_ylim((0,1))
    fig.tight_layout()

    plt.show()

    if save_plots:
        fig.savefig("fractional_rec_error_sans_mean.pdf")
