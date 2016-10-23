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
from sklearn.feature_extraction import DictVectorizer

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true')

# Flags to control functionality
find_best_lam = False
show_plots = False
save_plots = False

# Performance:
# Training, testing predicted number of twos: 265675, 44228
# Training, testing 0-1 loss: 0.129, 0.119
# Training, testing logloss: -0.687, -0.697

# Define constants
best_lambda = 1000.
best_thresh = 0.5
best_eta = 0.000251188643151
eta = 1.0e-5
eps = 5.0e-4

seed = 42
frac = 0.1
num = 6
lammax = 1000.0
scale = 10.0
sparse = False
Nclass = 10

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

# With a best fit lambda, threshold, refit
w0, w, ll_train, ll_test, iter_train = gd.gradient_ascent(cu.multi_logistic_grad, X_train, y_train_true,
                                                    lam=best_lambda, eta=eta, sparse=sparse,
                                                    savell=True, adaptive=True, eps=eps,
                                                    multi=Nclass, llfn=val.loglike_multi,
                                                    X_test=X_test, y_test=y_test_true)


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
            fig.savefig("mnist_multi_train_test_ll.pdf")

    plt.show()

y_hat_test = cu.multi_logistic_classifier(X_test, w, w0)

# Now compute, output 0-1 loss for training and testing sets
y_hat_train = cu.multi_logistic_classifier(X_train, w, w0)
y_hat_test = cu.multi_logistic_classifier(X_test, w, w0)
print("Training, testing predicted number of twos: %d, %d" % (np.sum(y_hat_train),np.sum(y_hat_test)))
error_train = val.loss_01(y_train, y_hat_train)/len(y_train)
error_test = val.loss_01(y_test, y_hat_test)/len(y_test)
print("Training, testing 0-1 loss: %.3lf, %.3lf" % (error_train, error_test))

# Now compute, output logloss for training and testing sets
logl_train = -val.loglike_multi(X_train, y_train_true, w, w0)/len(y_train)
logl_test = -val.loglike_multi(X_test, y_test_true, w, w0)/len(y_test)
print("Training, testing logloss: %.3lf, %.3lf" % (logl_train, logl_test))