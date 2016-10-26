# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.4 of CSE 546 HW2
"""

from __future__ import print_function, division
import numpy as np
import os
import sys
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.regression.regression_utils as ru
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import matplotlib as mpl
import matplotlib.pyplot as plt

# Performance
# Training, testing 0-1 loss: 0.079, 0.083
# Training, testing logloss: 0.291, 0.300
# best_lambda = 28140.0
# best_eta = 1.0e-8
# eps = 5.0e-3
# batchsize = 100

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

# Flags to control functionality

# Display, save plots?
show_plots = True
save_plots = True
run_minibatch_sgd = True

# Define constants


# Classifier parameters
eps = 1.0e-3
seed = 42
sparse = False
Nclass = 10
k = 10000
nn_cache = "../Data/hw2-data/MNIST_nn_cache.npz"

if not os.path.exists(nn_cache):
    RuntimeError("Transformation matrix v doesn't exist!")
else:
    print("Reading from cache...")
    res = np.load(nn_cache)
    v = res["v"]

# Load in MNIST training data then transform it
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
X_train = ru.naive_nn_layer(X_train, k=k, v=v)
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()
print("Estimated best lambda: %.3lf" % val.estimate_lambda(X_train, scale=1.0e-2))

# Load in MNIST training data then transform it
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
X_test = ru.naive_nn_layer(X_test, k=k, v=v)
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

#######################################################
#
# Run minibatch with bestfit?
#
#######################################################

if run_minibatch_sgd:
    # Performance

    best_lambda = 28140.0
    best_eta = 1.0e-8
    eps = 5.0e-3
    batchsize = 100

    print("Running minibatch SGD...")
    w0, w, ll_train, ll_test, train_01, test_01, iter_train = \
    gd.stochastic_gradient_descent(cu.multi_logistic_grad, X_train, y_train_true,
                                   lam=best_lambda, eta=best_eta, sparse=sparse,
                                   savell=True, adaptive=True, eps=eps,
                                   multi=Nclass, llfn=val.logloss_multi,
                                   X_test=X_test, y_test=y_test_true,
                                   train_label=y_train,
                                   test_label=y_test, classfn=cu.multi_logistic_classifier,
                                   loss01fn=val.loss_01, batchsize=batchsize)


    if show_plots:

        # Plot Training, testing ll vs iteration number
        fig, ax = plt.subplots()

        # Plot log loss
        ax.plot(iter_train, ll_train, lw=2, color="green", label=r"Train")
        ax.plot(iter_train, ll_test, lw=2, color="blue", label=r"Test")

        # Plot BGD best fit on the testing set from before
        plt.axhline(y=0.419, xmin=-1, xmax=100, ls="--", linewidth=2, color = 'k',
                    label="BGD Testing Best Fit")

        # Format plot
        ax.legend(loc="upper right")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LogLoss")
        fig.tight_layout()
        if save_plots:
                fig.savefig("sgd_nn_mnist_multi_train_test_ll.pdf")

        plt.show()

        # Plot Training, testing ll vs iteration number
        fig, ax = plt.subplots()

        # Plot 0/1 loss
        ax.plot(iter_train, train_01, lw=2, color="green", label=r"Train")
        ax.plot(iter_train, test_01, lw=2, color="blue", label=r"Test")

        # Plot BGD best fit on the testing set from before
        plt.axhline(y=0.106, xmin=-1, xmax=100, ls="--", linewidth=2, color = 'k',
                    label="BGD Testing Best Fit")

        # Format plot
        ax.legend(loc="upper right")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("0/1 Loss")
        fig.tight_layout()
        if save_plots:
                fig.savefig("sgd_nn_mnist_multi_train_test_01.pdf")

        plt.show()

    #######################################################
    #
    # Compute, output final losses
    #
    #######################################################
    val.summarize_loss(X_train, y_train_true, X_test, y_test_true, w, w0,
                       y_train_label=y_train, y_test_label=y_test,
                       classfn=cu.multi_logistic_classifier, lossfn=val.logloss_multi)