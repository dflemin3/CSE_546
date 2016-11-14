# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 5.1 of CSE 546 HW3
"""

from __future__ import print_function, division
import numpy as np
import sys
import os
sys.path.append("..")
import DML.data_processing.mnist_utils as mu
import DML.validation.validation as val
import DML.clustering.kmeans as kmeans
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Flags to control functionality
show_plots = False
save_plots = False
k = 5

# Load in MNIST data
#print("Loading MNIST data...")
#X_train, y_train = mu.load_mnist(dataset='testing')
X, y = make_blobs(n_samples=1000, n_features=k, centers=5, shuffle=True)

skmeans = KMeans(n_clusters=k, precompute_distances=False, algorithm="full",
                 init="random").fit(X)

print("Running k-means...")
y_hat = kmeans.kmeans(X, k=k)
y_hat_skmeans = skmeans.labels_
print("Done!")

print("0/1 loss:",val.loss_01(y,y_hat))
print("0/1 loss sklearn:",val.loss_01(y,y_hat_skmeans))
