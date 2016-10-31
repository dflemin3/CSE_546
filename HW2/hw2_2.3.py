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
run_sgd = True
run_minibatch_sgd = True

# Classifier parameters
best_lambda = 0.1 # Found via reg path
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

# Search for the best regularization constant using a regularization path?
# Use minibatch SGD to find it for speed/convergence reasons
if find_best_lambda:
    lammax = 1.0e8
    eta = 5.0e-6
    scale = 10.
    num = 10
    eps = 5.0e-3
    frac = 0.1 # Fraction of training data to split into validation set
    kwargs = {}

    # Split training data into subtraining set, validation set
    X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

    # Filter y values to 0, 1 labels
    y_tr_true = np.asarray(y_tr[:, None] == np.arange(max(y_tr)+1),dtype=int).squeeze()
    y_val_true = np.asarray(y_val[:, None] == np.arange(max(y_val)+1),dtype=int).squeeze()

    # Run reg path!
    err_val, err_train, lams = val.logistic_reg_path(X_tr, y_tr_true, \
                         X_val, y_val_true,
                         y_train_label = y_tr, y_val_label = y_val,
                         grad=cu.bin_logistic_grad, lammax=lammax,
                         scale=scale, num=num, error_func=val.loss_01, thresh=0.5, best_w=False,
                         eta = eta, sparse=False, eps=eps, max_iter=1000,
                         adaptive=True, llfn=val.logloss_multi, savell=False, batchsize=100,
                         classfn=cu.multi_logistic_classifier, **kwargs)

    # Find minimum lambda from minimum validation error
    # Mask infs
    print(err_val)
    err_val[np.isinf(err_val)] = np.nan
    ind_l = np.nanargmin(err_val)
    best_lambda = lams[ind_l]
    print("Best lambda:",best_lambda)

    if show_plots:
        fig, ax = plt.subplots()

        ax.plot(lams,err_val,lw=3,color="blue",label=r"Validation 0/1 Loss")
        ax.plot(lams,err_train,lw=3,color="green",label=r"Training 0/1 Loss")
        ax.axvline(x=best_lambda,lw=3,ls="--",color="black",label=r"Best $\lambda$")

        ax.set_xlabel(r"Regularization Constant $\lambda$")
        ax.set_ylabel("0/1 Loss")
        ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()

        plt.show()

        if save_plots:
            fig.savefig("sgd_reg_path.pdf")

if run_minibatch_sgd:
    # Performance
    # Training, testing 0-1 loss: 0.094, 0.088
    # Training, testing logloss: 0.336, 0.321
    # lam = 0.1, eta = 5.0e-6, eps = 5.0e-3

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
    # Training, testing 0-1 loss: 0.082, 0.091
    # Training, testing logloss: 0.467, 0.570
    # lambda: 0.1, eps = 5.0e-3, eta = 5.0e-6

    # SGD params
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