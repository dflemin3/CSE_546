# -*- coding: utf-8 -*-
"""
Created on Oct 20 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 1.1 of CSE 546 HW2

Best fit:
Training 01 Loss: 0.140733333333
Testing 01 Loss: 0.1382
Training Square Loss: 2.48708333333
Testing Square Loss: 2.4174
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import DML.regression.ridge_utils as ri

# Performance
# Training 01 Loss: 0.142333333333
# Testing 01 Loss: 0.1395
# Training Square Loss: 264.793533333
# Testing Square Loss: 263.6672

# Flags to control functionality
find_best_lam = False

# Define constants
best_lambda = 1.0e4
best_thresh = 0.4
seed = 1
frac = 0.1
num = 5
kwargs = {}
lammax = 1.0e4
scale = 10.0
Nclass = 10 # classes are 0 - 9

# Load in MNIST training data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()

# Load in MNIST training data
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()
# Estimate the best lambda?
if find_best_lam:

    kwargs = {}

    # Fit for 4s
    # Perform grid search to find best regularization constant and threshold?
    if find_best_lam:
        print("Finding optimal lambda and threshold via regularization path.")

        thresh_arr = np.linspace(-0.2,1.,num)
        err_val = np.zeros((num,num))
        err_train = np.zeros((num,num))

        # Split training data into subtraining set, validation set
        X_tr, y_tr, X_val, y_val = val.split_data(X_train, y_train, frac=frac, seed=seed)

        # Filter y values to 0, 1 labels
        y_tr_true = mu.mnist_filter(y_tr,filterby=3)
        y_val_true = mu.mnist_filter(y_val,filterby=3)

        # Loop over thresholds
        for i in range(num):
            # Internally loop over lambdas in regularization path
            err_val[i,:], err_train[i,:], lams = val.linear_reg_path(X_tr, y_tr_true, X_val,
            														 y_val_true, ri.fit_ridge,
            														 lammax=lammax,
            														 scale=scale,
																	 num=num, error_func=val.loss_01,
																	 thresh=thresh_arr[i], **kwargs)

        # Find minimum threshold, lambda from minimum validation error
        ind_t,ind_l = np.unravel_index(err_val.argmin(), err_val.shape)
        best_lambda = lams[ind_l]
        best_thresh = thresh_arr[ind_t]
        print("Best lambda:",best_lambda)
        print("Best threshold:",best_thresh)

# Fit for the class prediction regression coefficients
w0, w = ri.fit_ridge(X_train, y_train_true, lam=best_lambda)

print(w.shape,w0.shape)

# Using fit on training set, predict labels for train, test data by selecting whichever
# prediction is the largest (one vs all classification)
y_hat_train = cu.multi_linear_classifier(X_train, w, w0)
y_hat_test = cu.multi_linear_classifier(X_test, w, w0)

# Compute 01 Loss!
print("Training 01 Loss:",val.loss_01(y_train,y_hat_train))
print("Testing 01 Loss:",val.loss_01(y_test,y_hat_test))

# Compute square loss!
print("Training Square Loss:",val.square_loss(y_train_true,y_hat_train))
print("Testing Square Loss:",val.square_loss(y_test_true,y_hat_test))