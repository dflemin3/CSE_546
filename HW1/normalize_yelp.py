# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script normalizes the HW1 Yelp data such that each column/feature (n x 1) has a l2
norm of 1.  To do this, I simply divided every element in a column by the column's l2 norm.
"""

import pandas as pd
import numpy as np
import scipy.io as io
import scipy.sparse

# Normalize upvote data
df = pd.read_csv("/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw1-data/upvote_data.csv",header=None)

def func(x):
    # Normalize vector by its l2 norm
    return x/np.linalg.norm(x,ord=2)
df = df.apply(func,axis=0);

# Write to file
df.to_csv("/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw1-data/upvote_data.csv",header=False,index=False)

# Normalize star data
df = pd.DataFrame(data=io.mmread("/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw1-data/star_data.mtx").todense())

def func(x):
    # Normalize vector by its l2 norm
    return x/np.linalg.norm(x,ord=2)
df = df.apply(func,axis=0);

# Write to matrix market format
io.mmwrite("/Users/dflemin3/Desktop/Career/Grad_Classes/CSE_546/Data/hw1-data/star_data", scipy.sparse.csr_matrix(df))