# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 5.2 of CSE 546 HW3

0/1 Loss from k-means fit with 16 clusters...
Training: 0.332833333333
Testing: 0.3265

0/1 Loss from k-means fit with 250 clusters...
Training: 0.0890833333333
Testing: 0.084
"""

from __future__ import print_function, division
import sys
sys.path.append("..")
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import DML.clustering.kmeans as kmeans


# Load in MNIST data
print("Loading MNIST data...")
X_train, y_train = mu.load_mnist(dataset='training')
X_test, y_test = mu.load_mnist(dataset='testing')

# Fit with 16 clusters
k = 16
verbose = True

print("Running k-means with %d clusters..." % k)
y_cluster, mu_hat, rec_err, iter_arr = kmeans.kmeans(X_train, k=k, verbose=verbose)

# Now classifiy
classifier = kmeans.k_mapper(y_cluster, y_train)
y_hat_train = kmeans.k_classify(X_train, mu_hat, classifier)
y_hat_test = kmeans.k_classify(X_test, mu_hat, classifier)

print("0/1 Loss from k-means fit with %d clusters..." % k)
print("Training:",val.loss_01(y_train,y_hat_train))
print("Testing:",val.loss_01(y_test,y_hat_test))

# Fit with 250 clusters
k = 250
verbose = True

print("Running k-means with %d clusters..." % k)
y_cluster, mu_hat, rec_err, iter_arr = kmeans.kmeans(X_train, k=k, verbose=verbose)

# Now classifiy
classifier = kmeans.k_mapper(y_cluster, y_train)
y_hat_train = kmeans.k_classify(X_train, mu_hat, classifier)
y_hat_test = kmeans.k_classify(X_test, mu_hat, classifier)

print("0/1 Loss from k-means fit with %d clusters..." % k)
print("Training:",val.loss_01(y_train,y_hat_train))
print("Testing:",val.loss_01(y_test,y_hat_test))
