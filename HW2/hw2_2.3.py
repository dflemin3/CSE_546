# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.3 of CSE 546 HW2
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

# Flags to control functionality

# Display, save plots?
find_best_lambda = True
show_plots = True
save_plots = True
run_sgd = False
run_minibatch_sgd = True

# Define constants


# Classifier parameters
eps = 1.0e-3
seed = 42
sparse = False
Nclass = 10

# Load in MNIST training data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()
print("Estimated best lambda: %.3lf" % val.estimate_lambda(X_train))

# Load in MNIST training data
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

#######################################################
#
# Run minibatch with bestfit?
#
#######################################################

# Search for the best regularization constant?
if find_best_lambda:
    error_val, error_train, np.array(lams) = logistic_reg_path(X_train, y_train, X_val, y_val,
                         y_train_label = None, y_val_label = None,
                         grad=cu.bin_logistic_grad, lammax=1000.,
                         scale=2.0, num=10, error_func=loss_01, thresh=0.5, best_w=False,
                         eta = 1.0e-4, sparse=False, eps=5.0e-3, max_iter=1000,
                         adaptive=True, llfn=logloss_bin, savell=False, batchsize=100,
                         classfn=cu.logistic_classifier, **kwargs)

if run_minibatch_sgd:
    # Performance
    # Training, testing 0-1 loss: 0.095, 0.090
    # Training, testing logloss: 0.341, 0.326
    # lam = 1.0, eta = 5.0e-6, eps = 5.0e-3

    best_lambda = 1.0
    best_thresh = 0.5
    best_eta = 5.0e-6
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
                fig.savefig("sgd_mini_mnist_multi_train_test_ll.pdf")

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
                fig.savefig("sgd_mini_mnist_multi_train_test_01.pdf")

        plt.show()

    #######################################################
    #
    # Compute, output final losses
    #
    #######################################################
    val.summarize_loss(X_train, y_train_true, X_test, y_test_true, w, w0,
                       y_train_label=y_train, y_test_label=y_test,
                       classfn=cu.multi_logistic_classifier, lossfn=val.logloss_multi)

#######################################################
#
# Run SGD with best fit?
#
#######################################################
if run_sgd:

    # Performance
    # Training, testing 0-1 loss: 0.081, 0.090
    # Training, testing logloss: 0.379, 0.453
    # lambda: 1.0, eps = 5.0e-3, eta = 5.0e-6

    # SGD params
    best_lambda = 1.0
    best_thresh = 0.5
    best_eta = 5.0e-6
    eps = 5.0e-3
    batchsize = None

    print("Running SGD...")
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

        # Plot -(log likelikehood) to get logloss
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
                fig.savefig("sgd_mnist_multi_train_test_ll.pdf")

        plt.show()

        # Plot Training, testing ll vs iteration number
        fig, ax = plt.subplots()

        # Plot -(log likelikehood) to get logloss
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
                fig.savefig("sgd_mnist_multi_train_test_01.pdf")

        plt.show()

    #######################################################
    #
    # Compute, output final losses
    #
    #######################################################
    val.summarize_loss(X_train, y_train_true, X_test, y_test_true, w, w0,
                       y_train_label=y_train, y_test_label=y_test,
                       classfn=cu.multi_logistic_classifier, lossfn=val.logloss_multi)