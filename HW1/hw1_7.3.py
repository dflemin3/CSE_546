#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script answers questions 7.3.1 and 7.3.2 from CSE 546 HW1
"""

import regression_utils as ru
import lasso_utils as lu
import numpy as np

#############################################
#
# 7.3.1.
#
#############################################

# Define parameters
n = 50
d = 75
k = 5
sigma = 1
sparse = True

# Generate synthetic data (sparse for the lasso function)
w_true, X_train, y_true = ru.generate_norm_data(n,k,d,sigma,sparse=sparse)

print("lam_max:",lu.compute_max_lambda(X_train,y_true))

# Train lasso
lam = 100.0 
w_0_pred, w_pred = lu.fit_lasso(X_train,y_true,lam=lam, sparse=sparse)

# Make Prediction
y_hat = ru.linear_model(X_train, w_pred, w_0_pred, sparse=sparse)

# Test precision, recall
print("Precision:",ru.precision_lasso(w_true,w_pred))
print("Recall:",ru.recall_lasso(w_true,w_pred))