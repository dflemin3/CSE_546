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

# Generate synthetic data (sparse for the lasso function)
w, X, y = ru.generate_norm_data(n,k,d,sigma,sparse=True)

print("lam_max:",lu.compute_max_lambda(X,y))
