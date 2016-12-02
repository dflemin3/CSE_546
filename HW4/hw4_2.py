# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2 of CSE 546 HW4

ReLu, linear:

# Parameters
eps = 2.5e-3
eta = 5.0e-9
k = 50
scale = 0.1
lam = 0.0
batchsize = 10
nout = 30000
nclass = 10
nodes = 500


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

# Parameters
eps = 2.5e-3
eta = 5.0e-3
k = 50
scale = 0.1
lam = 0.0
batchsize = 10
nout = 30000
nclass = 10
nodes = 500

# Load in MNIST data
print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=k, center=True)

print("Performing PCA with k = %d components..." % k)
PCA.fit(X_train)
X_train = PCA.transform(X_train)
X_test = PCA.transform(X_test)

"""
cut = 5000
X_train = X_train[:cut]
y_train = y_train[:cut]
"""

print("Training neural network...")
activators = [deep.tanh,deep.sigmoid]
activators_prime = [deep.tanh_prime,deep.sigmoid_prime]
y_hat_train, w_1, w_2 = deep.neural_net(X_train, y_train_true, nodes=nodes, activators=activators, activators_prime=activators_prime,
               scale=scale, eps=eps, eta=eta, lam=lam, adaptive=True, batchsize=batchsize, nout=nout, nclass=nclass)

# Compute y_hat
a_hidden = activators[0](X_test.dot(w_1))
y_hat_test = activators[1](a_hidden.dot(w_2))

y_hat_train = np.argmax(y_hat_train, axis=1).reshape((X_train.shape[0],1))
y_hat_test = np.argmax(y_hat_test, axis=1).reshape((X_test.shape[0],1))

print("Training 0/1 Loss: %lf" % val.loss_01(y_train,y_hat_train))
print("Testing 0/1 Loss: %lf" % val.loss_01(y_test,y_hat_test))
