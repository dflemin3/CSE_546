# -*- coding: utf-8 -*-
"""
Created on Nov 2016

@author: dflemin3 [David P Fleming, University of Washington]

@email: dflemin3 (at) uw (dot) edu

This script solves question 2 of CSE 546 HW3 of using one-against-all linear
regression classification on PCA-reduced and RBF fourier kernel transformend
MNIST data.
"""

from __future__ import (print_function, division, absolute_import,
                       unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)

# Define points
x1 = [0, -1]
x2 = [np.sqrt(2), 1]

# Define f(x)
x_arr = np.linspace(-2,2,100)
fx = 0.5*x_arr**2 + np.sqrt(2.0)*x_arr/2.0 - 1.0

fig, ax = plt.subplots()

# Plot f(x), points
ax.plot(x_arr,fx,lw=2,color="black", label="f(x)")
ax.scatter(0, -1, s=50, color="red",label="X1", zorder=20)
ax.scatter(np.sqrt(2), 1, s=50, color="green",label="X2", zorder=20)

# Format plot
ax.set_xlabel("x")
ax.set_ylabel("f(x), y")
ax.set_xlim(-2,2)
ax.legend(loc="upper left")

plt.show()

fig.savefig("svm_by_hand.pdf")
