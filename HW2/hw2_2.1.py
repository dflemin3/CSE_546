# -*- coding: utf-8 -*-
"""
Created on Mon Oct  19

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.1 of CSE 546 HW2
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
find_best_lam = False
show_plots = True
save_plots = False

# Performance:
# Training, testing predicted number of twos: 5693, 956
# Training, testing 0-1 loss: 0.051, 0.046
# Training, testing logloss: 0.540, 0.540

# Define constants
best_lambda = 1000.
best_thresh = 0.5
best_eta = 0.000251188643151
eps = 1.0e-2

seed = 42
frac = 0.1
num = 6
lammax = 1000.0
scale = 10.0
sparse = False

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

y_train_true = mu.mnist_filter(y_train, filterby=2)
print("True number of twos in training set:",np.sum(y_train_true))

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = mu.mnist_filter(y_test, filterby=2)

# Build regularized binary classifier by minimizing 01 loss
# Will need to optimize over lambda and eta via a regularization path
if find_best_lam:
    print("Running regularization path to optmize lambda, eta...")

    # Split training data into subtraining set, validation set
    X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

    # Filter y values to 0, 1 labels
    y_tr_true = mu.mnist_filter(y_tr, filterby=2)
    y_val_true = mu.mnist_filter(y_val, filterby=2)

    # Define arrays
    eta_arr = np.logspace(-6,0,num)
    err_val = np.zeros((num,num))
    err_train = np.zeros((num,num))

    for ii in range(num):
        print("Eta:",eta_arr[ii])
        err_val[ii,:], err_train[ii,:], lams = \
        val.binlogistic_reg_path(X_tr, y_tr_true, X_val, y_val_true, grad=cu.bin_logistic_grad,
        lammax=lammax,scale=scale, num=num, error_func=val.loss_01, thresh=best_thresh, best_w=False,
        eta=eta_arr[ii], adaptive=True, llfn=val.loglike_bin, savell=False)

    # Find minimum threshold, lambda from minimum validation error
    # Mask infs
    err_val[np.isinf(err_val)] = np.nan
    ind_e,ind_l = np.unravel_index(np.nanargmin(err_val), err_val.shape)
    best_lambda = lams[ind_l]
    best_eta = eta_arr[ind_e]
    print("Best lambda:",best_lambda)
    print("Best eta:",best_eta)

# With a best fit lambda, threshold, refit
w0, w, ll_train, ll_test, iter_train = gd.gradient_ascent(cu.bin_logistic_grad, X_train, y_train_true,
                                                    lam=best_lambda, eta=best_eta, savell=True,
                                                    X_test=X_test, y_test=y_test_true,
                                                    adaptive=True, eps=eps)

if show_plots:

    # Plot Training, testing ll vs iteration number
    fig, ax = plt.subplots()

    # Plot -(log likelikehood) to get logloss
    ax.plot(iter_train, -ll_train, lw=2, color="green", label=r"Train")
    ax.plot(iter_train, -ll_test, lw=2, color="blue", label=r"Test")

    # Format plot
    ax.legend(loc="upper right")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LogLoss")
    fig.tight_layout()
    if save_plots:
            fig.savefig("mnist_train_test_ll.pdf")

    plt.show()

# Now compute, output 0-1 loss for training and testing sets
y_hat_train = cu.logistic_classifier(X_train, w, w0, thresh=best_thresh, sparse=sparse)
y_hat_test = cu.logistic_classifier(X_test, w, w0, thresh=best_thresh, sparse=sparse)
print("Training, testing predicted number of twos: %d, %d" % (np.sum(y_hat_train),np.sum(y_hat_test)))
error_train = val.loss_01(y_train_true, y_hat_train)/len(y_train)
error_test = val.loss_01(y_test_true, y_hat_test)/len(y_test)
print("Training, testing 0-1 loss: %.3lf, %.3lf" % (error_train, error_test))

# Now compute, output logloss for training and testing sets
y_hat_train = cu.logistic_model(X_train, w, w0, sparse=sparse)
y_hat_test = cu.logistic_model(X_test, w, w0, sparse=sparse)
ll_train = -val.loglike_bin(y_train, y_hat_train)/len(y_hat_train)
ll_test = -val.loglike_bin(y_test, y_hat_test)/len(y_hat_test)
print("Training, testing logloss: %.3lf, %.3lf" % (ll_train, ll_test))