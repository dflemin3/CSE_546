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
#print("Loading MNIST Training data...")
#X_train, y_train = mu.load_mnist(dataset='training')

# Load in MNIST training data, set 2s -> 1, others -> 0
print("Loading MNIST Testing data...")
X_test, y_test = mu.load_mnist(dataset='testing')

y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

# With a best fit lambda, threshold, refit
w0, w, ll_train, iter_train = gd.gradient_ascent(cu.multi_logistic_grad, X_test, y_test_true,
                                                    lam=best_lambda, eta=eta, sparse=sparse,
                                                    savell=True, adaptive=True, eps=eps,
                                                    multi=Nclass, llfn=val.loglike_multi)


if show_plots:
    plt.plot(iter_train,ll_train)
    plt.show()

y_hat_test = cu.multi_logistic_classifier(X_test, w, w0)

#for ii in range(len(y_hat_test)):
#    print(y_test[ii],y_hat_test[ii])

error_test = val.loss_01(y_test, y_hat_test)/len(y_test)
print("01 Loss:",error_test)