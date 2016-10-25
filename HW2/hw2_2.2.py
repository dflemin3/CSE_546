# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.2 of CSE 546 HW2
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

# Perform regularization path over lambda and eta?
find_best_lam = False

# Display, save plots?
show_plots = True
save_plots = True

# Best Performance:
# Training, testing 0-1 loss: 0.112, 0.106
# Training, testing logloss: 0.439, 0.419
# eta: 1.0e-4
# eps: 5.0e-3

# Define constants

# Regularization path parameters
lammax = 1.0e3
scale = 10
num = 5
frac = 0.1

# Best fits from reg path
best_lambda = 1000.0
best_thresh = 0.5
best_eta = 1.0e-4

# Classifier parameters
eps = 5.0e-3
seed = 42
sparse = False
Nclass = 10

# Load in MNIST training data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()

# Load in MNIST training data
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

#######################################################
#
# Run reg path over lambda, eta?
#
#######################################################
if find_best_lam:
    print("Running regularization path to optmize lambda, eta...")
    print("Estimated best lambda: %.3lf" % val.estimate_lambda(X_train, scale=1.0e-3))

    # Split training data into subtraining set, validation set
    X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

    # Filter y values to 0, 1 labels
    y_tr_true = np.asarray(y_tr[:, None] == np.arange(max(y_tr)+1),dtype=int).squeeze()
    y_val_true = np.asarray(y_val[:, None] == np.arange(max(y_val)+1),dtype=int).squeeze()

    # Define arrays
    eta_arr = np.logspace(-7,-3,num)
    err_val = np.zeros((num,num))
    err_train = np.zeros((num,num))

    for ii in range(num):
        print("Eta:",eta_arr[ii])
        err_val[ii,:], err_train[ii,:], lams = \
        val.logistic_reg_path(X_tr, y_tr_true, X_val, y_val_true, grad=cu.multi_logistic_grad,
        y_train_label=y_tr, y_val_label=y_val, classfn=cu.multi_logistic_classifier,
        lammax=lammax,scale=scale, num=num, error_func=val.loss_01, thresh=best_thresh, best_w=False,
        eta=eta_arr[ii], adaptive=True, llfn=val.logloss_multi, savell=False, eps=1.0e-2)

    # Find minimum threshold, lambda from minimum validation error
    # Mask infs
    err_val[np.isinf(err_val)] = np.nan
    ind_e,ind_l = np.unravel_index(np.nanargmin(err_val), err_val.shape)
    best_lambda = lams[ind_l]
    best_eta = eta_arr[ind_e]
    print("Best lambda:",best_lambda)
    print("Best eta:",best_eta)

#######################################################
#
# With a best fit lambda, threshold, refit, plot(?)
#
#######################################################
w0, w, ll_train, ll_test, train_01, test_01, iter_train = \
gd.gradient_descent(cu.multi_logistic_grad, X_train, y_train_true,
                               lam=best_lambda, eta=best_eta, sparse=sparse,
                               savell=True, adaptive=True, eps=eps,
                               multi=Nclass, llfn=val.logloss_multi,
                               X_test=X_test, y_test=y_test_true,
                               train_label=y_train,
                               test_label=y_test, classfn=cu.multi_logistic_classifier,
                               loss01fn=val.loss_01)


if show_plots:

    # Plot Training, testing ll vs iteration number
    fig, ax = plt.subplots()

    # Plot -(log likelikehood) to get logloss
    ax.plot(iter_train, ll_train, lw=2, color="green", label=r"Train")
    ax.plot(iter_train, ll_test, lw=2, color="blue", label=r"Test")

    # Format plot
    ax.legend(loc="upper right")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LogLoss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("bgd_mnist_multi_train_test_ll.pdf")

    plt.show()

    # Plot Training, testing ll vs iteration number
    fig, ax = plt.subplots()

    # Plot -(log likelikehood) to get logloss
    ax.plot(iter_train, train_01, lw=2, color="green", label=r"Train")
    ax.plot(iter_train, test_01, lw=2, color="blue", label=r"Test")

    # Format plot
    ax.legend(loc="upper right")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("0/1 Loss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("bgd_mnist_multi_train_test_01.pdf")

    plt.show()

#######################################################
#
# Compute, output final losses
#
#######################################################
val.summarize_loss(X_train, y_train_true, X_test, y_test_true, w, w0,
                       y_train_label=y_train, y_test_label=y_test,
                       classfn=cu.multi_logistic_classifier, lossfn=val.logloss_multi)