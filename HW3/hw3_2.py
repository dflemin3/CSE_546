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
import DML.kernel.kernel as kernel
import matplotlib as mpl
import matplotlib.pyplot as plt

# Flags to control functionality
save_plots = False

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='testing')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()

#print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=50, center=True)

print("Training the model...")
PCA.fit(X_train)
X_train = PCA.transform(X_train)
X_test = PCA.transform(X_test)

cut = 2000

X_train = X_train[:cut]
y_train = y_train[:cut]
y_train_true = y_train_true[:cut]
X_test = X_test[:cut]
y_test = y_test[:cut]
y_test_true = y_test_true[:cut]

# Estimate kernel bandwidth
sigma = kernel.estimate_bandwidth(X_train, num = 100, scale = 10.0)
print("Estimted kernel bandwidth: %lf" % sigma)

best_eta = 1.0e-3
eps = 1.0e-3
batchsize = 100
sparse = False
Nclass = 10

print("Running minibatch SGD...")
w0, avg_w0, w_pred, avg_w, train_ll, avg_train_ll, test_ll, avg_test_ll, \
train_01, avg_train_01, test_01, avg_test_01, iter_arr = \
gd.SGD_chunks(ru.linear_grad, X_train, y_train_true,
                               X_test=X_test, y_test=y_test_true,
                               lam=0.0, eta=best_eta, sparse=sparse,
                               savell=True, adaptive=True, eps=eps,
                               multi=Nclass, llfn=val.MSE_multi,
                               batchsize=batchsize,
                               train_label=y_train, test_label=y_test,
                               classfn = cu.multi_linear_classifier,
                               nout=X_train.shape[0], loss01fn=val.loss_01,
                               transform=kernel.RBF, alpha=sigma)


# Plot Training, testing ll vs iteration number
fig, ax = plt.subplots()

# Plot 0/1 loss
ax.plot(iter_arr, train_ll, lw=2, color="green", label=r"Train")
ax.plot(iter_arr, test_ll, lw=2, color="blue", label=r"Test")
ax.plot(iter_arr, avg_train_ll, lw=2, color="green", ls="--", label=r"Avg Train")
ax.plot(iter_arr, avg_test_ll, lw=2, color="blue", ls="--", label=r"Avg Test")

# Format plot
ax.legend(loc="upper right")
ax.set_xlabel("Iteration")
ax.set_ylabel("0/1 Loss")
fig.tight_layout()
if save_plots:
        fig.savefig("sgd_nn_mnist_multi_train_test_01.pdf")

plt.show()
