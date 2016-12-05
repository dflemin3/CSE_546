# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2.1 of CSE 546 HW4

tanh, linear

Training square Loss: 0.055638
Testing square Loss: 0.073848
Training 0/1 Loss: 0.013117
Testing 0/1 Loss: 0.027800

"""

from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.pca.pca as pca
import DML.data_processing.mnist_utils as mu
import DML.deep_learning.deep_utils as deep
import DML.validation.validation as val
import random

# Parameters for tanh, linear layers
eps = 7.5e-4
eta = 1.0e-3
k = 50
scale = 0.001
lam = 0.0
batchsize = 10
nout = 30000
nclass = 10
nodes = 500
seed = 42

# Seed RNG
np.random.seed(seed=seed)

# Flags to control functionality
show_plots = True
save_plots = True

# Load in MNIST data, process it for multiclass classification
print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='training')
y_train_true = np.asarray(y_train[:, None] == np.arange(max(y_train)+1),dtype=int).squeeze()
X_test, y_test = mu.load_mnist(dataset='testing')
y_test_true = np.asarray(y_test[:, None] == np.arange(max(y_test)+1),dtype=int).squeeze()

# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
print("Performing PCA with k = %d components..." % k)
PCA = pca.PCA(l=k, center=False)
PCA.fit(X_train)
X_train = PCA.transform(X_train)
X_test = PCA.transform(X_test)

print("Training neural network...")
activators = [deep.tanh,deep.linear]
activators_prime = [deep.tanh_prime,deep.linear_prime]

y_hat, w_1, w_2, b_1, b_2, iter_arr, train_sq_loss, train_01_loss, test_sq_loss, \
test_01_loss = \
                deep.neural_net(X_train, y_train_true, nodes=nodes,
                               activators=activators,
                               activators_prime=activators_prime, scale=scale,
                               eps=eps, eta=eta, lam=lam, adaptive=True,
                               batchsize=batchsize, nout=nout, nclass=nclass,
                               X_test = X_test, y_test = y_test_true,
                               train_label=y_train, test_label=y_test,
                               verbose=True)


# Output final losses
print("Training square Loss: %lf" % train_sq_loss[-1])
print("Testing square Loss: %lf" % test_sq_loss[-1])
print("Training 0/1 Loss: %lf" % train_01_loss[-1])
print("Testing 0/1 Loss: %lf" % test_01_loss[-1])

# Visualize losses?
if show_plots:
    print("Plotting...")
    #Typical plot parameters that make for pretty plots
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (9,8)
    mpl.rcParams['font.size'] = 20.0
    mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
    mpl.rc('text', usetex=True)

    # Plot square loss vs epoch
    fig, ax = plt.subplots()

    ax.plot(iter_arr, train_sq_loss, lw=2, color="green", label="Train")
    ax.plot(iter_arr, test_sq_loss, lw=2, color="blue", label="Test")

    ax.set_ylabel("Mean Square Loss")
    ax.set_xlabel("Half Epoch")
    ax.legend(loc="best")

    plt.show()

    if save_plots:
        fig.savefig("tanh_linear_sq.pdf")

    # Plot 0/1 loss vs epoch onces it's below 7% on testing set
    fig, ax = plt.subplots()

    mask = test_01_loss < 0.07

    ax.plot(iter_arr[mask], train_01_loss[mask], lw=2, color="green", label="Train")
    ax.plot(iter_arr[mask], test_01_loss[mask], lw=2, color="blue", label="Test")

    ax.set_ylabel("0/1 Loss")
    ax.set_xlabel("Half Epoch")
    ax.legend(loc="best")

    plt.show()

    if save_plots:
        fig.savefig("tanh_linear_01.pdf")

    # Plot learned hidden layer weights in "physical space"
    # 50 x 500, 500 x 10
    hidden = w_1.T[random.sample(range(0, len(w_1)), 10)]

    fig, axes = plt.subplots(nrows=5,ncols=2)

    for ii, ax in enumerate(axes.flatten()):

        image = PCA.inverse_transform(hidden[ii,:])

        # Plot reconstructed image
        ax.imshow(image.reshape((28, 28)), cmap='binary',
        interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if save_plots:
        fig.savefig("tanh_linear_hidden.pdf")
