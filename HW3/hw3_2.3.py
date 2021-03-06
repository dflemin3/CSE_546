# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.3 of CSE 546 HW3 [Extra Credit!]
of using softmax classification on PCA-reduced and neural network
transformend MNIST data.
"""

from __future__ import print_function, division
import numpy as np
import sys
import os
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
show_plots = True
save_plots = True

# Load in MNIST data
print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='training')
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

"""
cut = 10000
X_train = X_train[:cut]
y_train = y_train[:cut]
y_train_true = y_train_true[:cut]
X_test = X_test[:cut]
y_test = y_test[:cut]
y_test_true = y_test_true[:cut]
"""

# Generate naive neural network transformation matrix v
v = ru.neural_net_v(X_train.shape[-1], X_train.shape[0])

# Set SGD parameters
best_eta = 1.0e-9
eps = 1.0e-4
batchsize = 100
sparse = False
Nclass = 10
cache_name = "nn_best.npz"


# Only run if cache doesn't exist
if not os.path.exists(cache_name):
    print("Running minibatch SGD...")
    w0, avg_w0, w, avg_w, train_ll, avg_train_ll, test_ll, avg_test_ll, \
    train_01, avg_train_01, test_01, avg_test_01, iter_arr = \
    gd.SGD_chunks(cu.multi_logistic_grad, X_train, y_train_true,
                                   X_test=X_test, y_test=y_test_true,
                                   lam=0.0, eta=best_eta, sparse=sparse,
                                   savell=True, adaptive=True, eps=eps,
                                   multi=Nclass, llfn=val.logloss_multi,
                                   batchsize=batchsize,
                                   train_label=y_train, test_label=y_test,
                                   classfn = cu.multi_logistic_classifier,
                                   nout=X_train.shape[0], loss01fn=val.loss_01,
                                   transform=ru.neural_net, sigma=1.0,
                                   v=v)

    # Cache results...
    np.savez(cache_name, w0=w0, avg_w0=avg_w0, w=w, avg_w=avg_w, train_ll=train_ll,
             avg_train_ll=avg_train_ll, test_ll=test_ll, avg_test_ll=avg_test_ll,
             train_01=train_01, avg_train_01=avg_train_01, test_01=test_01,
             avg_test_01=avg_test_01, iter_arr=iter_arr,sigma=sigma,best_eta=best_eta,
             eps=eps)
# Cache exists, load it
else:
    res = np.load(cache_name)
    iter_arr = res["iter_arr"]
    train_ll = res["train_ll"]
    avg_train_ll = res["avg_train_ll"]
    test_ll = res["test_ll"]
    avg_test_ll = res["test_ll"]
    train_01 = res["train_01"]
    avg_train_01 = res["avg_train_01"]
    test_01 = res["test_01"]
    avg_test_01 = res["avg_test_01"]

if show_plots:

    # Plot Training, testing ll vs iteration number
    fig, ax = plt.subplots()

    # Plot 0/1 loss
    ax.plot(iter_arr, train_ll, lw=2, color="green", label=r"Train")
    ax.plot(iter_arr, test_ll, lw=2, color="blue", label=r"Test")
    ax.plot(iter_arr, avg_train_ll, lw=2, color="green", ls="--", label=r"Avg Train")
    ax.plot(iter_arr, avg_test_ll, lw=2, color="blue", ls="--", label=r"Avg Test")

    # Format plot
    ax.legend(loc="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Square Loss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("nn_square_loss.pdf")

    plt.show()

    # Plot Training, testing 0/1 loss vs iteration number
    fig, ax = plt.subplots()

    # Plot 0/1 loss
    ax.plot(iter_arr, train_01, lw=2, color="green", label=r"Train")
    ax.plot(iter_arr, test_01, lw=2, color="blue", label=r"Test")
    ax.plot(iter_arr, avg_train_01, lw=2, color="green", ls="--", label=r"Avg Train")
    ax.plot(iter_arr, avg_test_01, lw=2, color="blue", ls="--", label=r"Avg Test")

    # Format plot
    ax.legend(loc="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("0/1 Loss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("nn_01_loss.pdf")

    plt.show()

    # Plot Training, testing 0/1 loss vs iteration number once 0/1 loss < 0.05
    fig, ax = plt.subplots()

    mask = test_01 <= 0.05

    # Plot 0/1 loss
    ax.plot(iter_arr[mask], train_01[mask], lw=2, color="green", label=r"Train")
    ax.plot(iter_arr[mask], test_01[mask], lw=2, color="blue", label=r"Test")
    ax.plot(iter_arr[mask], avg_train_01[mask], lw=2, color="green", ls="--", label=r"Avg Train")
    ax.plot(iter_arr[mask], avg_test_01[mask], lw=2, color="blue", ls="--", label=r"Avg Test")

    # Format plot
    ax.legend(loc="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("0/1 Loss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("nn_01_loss_masked.pdf")

    plt.show()

#######################################################
#
# Compute, output final losses
#
#######################################################
print("Best fit losses:")
print("Training square loss, 0/1 loss:",train_ll[-1],train_01[-1])
print("Testing square loss, 0/1 loss:",test_ll[-1],test_01[-1])

print("Best fit losses averaged over last epoch:")
print("Training square loss, 0/1 loss:",avg_train_ll[-1],avg_train_01[-1])
print("Testing square loss, 0/1 loss:",avg_test_ll[-1],avg_test_01[-1])
