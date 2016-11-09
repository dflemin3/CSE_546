# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2 of CSE 546 HW3
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.pca.pca as pca
import DML.data_processing.mnist_utils as mu
import DML.data_processing.normalization as norm
import DML.classification.classifier_utils as cu
import DML.regression.regression_utils as ru
import DML.optimization.gradient_descent as gd
import DML.validation.validation as val
import matplotlib as mpl
import matplotlib.pyplot as plt

# Flags to control functionality
save_plots = False

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='testing')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=100, center=True)

print("Training the model...")
PCA.fit(X_train)

# Reproject training data using PCA approximation
X_train = PCA.reproject(X_train)

best_eta = 1.0e-7
eps = 1.0e-4
batchsize = 100
sparse = False
Nclass = 10

print("Running minibatch SGD...")
w0, w, ll_train, train_01, iter_train = \
gd.stochastic_gradient_descent(ru.linear_grad, X_train, y_train_true,
                               lam=0.0, eta=best_eta, sparse=sparse,
                               savell=True, adaptive=True, eps=eps,
                               multi=Nclass, llfn=val.MSE_multi,
                               train_label=y_train,
                               classfn=cu.multi_linear_classifier,
                               loss01fn=val.loss_01, batchsize=batchsize)

# Plot Training, testing ll vs iteration number
fig, ax = plt.subplots()

# Plot 0/1 loss
ax.plot(iter_train, train_01, lw=2, color="green", label=r"Train")

# Format plot
ax.legend(loc="upper right")
ax.set_xlabel("Iteration")
ax.set_ylabel("0/1 Loss")
fig.tight_layout()
if save_plots:
        fig.savefig("sgd_nn_mnist_multi_train_test_01.pdf")

plt.show()
