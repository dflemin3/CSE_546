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
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['font.size'] = 20.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)

#############################################
#
# 7.3.1.
#
#############################################

# Define parameters
n = 50
d = 75
k = 5
max_lam = 5000.0
sigma = 1
sparse = True
max_iter = 10
show_plots = True

# Generate synthetic data (sparse for the lasso function)
print("Generating synthetic data for sigma = %.1lf" % sigma)
w_true, X_train, y_train = ru.generate_norm_data(n,k,d,sigma,sparse=sparse)

print("Running lasso regularization path for sigma = %.1lf" % sigma)
lams_1, recall_1, prec_1 = lu.lasso_reg_path(X_train, y_train, w_true, sparse = sparse, 
									   scale = 2., max_iter = max_iter, max_lam=max_lam)

# Generate synthetic data (sparse for the lasso function)
print("Generating synthetic data for sigma = %.1lf" % (10*sigma))
w_true, X_train, y_train = ru.generate_norm_data(n,k,d,(10*sigma),sparse=sparse)

print("Running lasso regularization path for sigma = %.1lf" % sigma * 10)
lams_10, recall_10, prec_10 = lu.lasso_reg_path(X_train, y_train, w_true, sparse = sparse, 
									   scale = 2., max_iter = max_iter, max_lam=max_lam)
# Sigma = 1, 10 plot
if show_plots:
	print("Plotting precission, recall...")
	fig, ax = plt.subplots()
	
	# Sigma = 1 curves
	ax.plot(lams_1,recall_1, lw=3, color="blue", label="Recall")
	ax.plot(lams_1,prec_1, lw=3, color="red", label="Precision")
	
	# Sigma = 10 curves
	ax.plot(lams_10,recall_10, lw=3, color="blue", ls="--")
	ax.plot(lams_10,prec_10, lw=3, color="red", ls="--")
	
	# Format
	ax.set_xlabel(r"Regularization Parameter $\lambda$")
	ax.set_ylabel("Recall, Precision")
	
	ax.set_ylim(0.,1.05)
	ax.set_xlim(8,(max_lam+0.1*max_lam))
	ax.set_xscale("log")
	leg = ax.legend(loc=8)
	leg.get_frame().set_alpha(0.0)
	
	fig.tight_layout()
	fig.savefig("synthetic_prec_rec.pdf")
	
	plt.show()






