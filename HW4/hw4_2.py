# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2 of CSE 546 HW4
"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.pca.pca as pca
import DML.data_processing.mnist_utils as mu
import DML.deep_learning.deep_utils as deep
import DML.validation.validation as val

"""import matplotlib.pyplot as plt
import matplotlib as mpl
#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)"""

# Flags to control functionality
show_plots = True
save_plots = True
use_one_digit = False

# Parameters
eps = 2.0e-3
eta = 1.0e-4

# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='testing')
y_train = mu.mnist_filter(y_train)

cut = 2000
X_train = X_train[:cut]
y_train = y_train[:cut]

print("Training neural network...")
activators = [deep.relu,deep.sigmoid]
activators_prime = [deep.relu_prime,deep.sigmoid_prime]
y_hat = deep.neural_net(X_train, y_train, nodes=500, activators=activators, activators_prime=activators_prime,
               scale=1.0, eps=eps, eta=eta)
mask = y_hat > 0.5
y_hat[mask] = 1
y_hat[~mask] = 0

print(y_hat,np.sum(y_hat))

print("0/1 Loss: %lf" % val.loss_01(y_train,y_hat))
