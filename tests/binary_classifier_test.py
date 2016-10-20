# Test functions using my logistic regressor and sklearn's implementation
import sys
import numpy as np
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu
import DML.regression.regression_utils as ru
import DML.validation.validation as val

# Fake classifier parameters
n = 1000
k = 5
d = 10
w0 = 0
seed = 42

w, X, y = cu.generate_binary_data(n, k, d, w0=w0, seed=seed)
#w, X, y = ru.generate_norm_data(n, k, d, w0=w0, seed=seed)

# Normalize data
#X = val.normalize(X)
#X = X - np.mean(X,axis=0)

print(w)

# Now try with my gradient descent implementation using adaptive step sizes
model = cu.logistic_model
#model = ru.linear_model
w0, w_pred = gd.gradient_ascent(model, X, y, lam=1.0e-1, eta = 1.0e0, max_iter = 500, adaptive=True)

print(w_pred)