# Test functions using my logistic regressor and sklearn's implementation
import sys
import numpy as np
sys.path.append("..")
import DML.classification.classifier_utils as cu
import DML.optimization.gradient_descent as gd
import DML.data_processing.mnist_utils as mu

# Do sklearn test?
do_sklearn_fit = False

# Fake classifier parameters
n = 1000
k = 5
d = 10
w0 = 0
seed = 42

w, X, y = cu.generate_binary_data(n, k, d, w0=w0, seed=seed)

# Center data
X = X - np.mean(X,axis=0)
y = y - np.mean(y)

print(w)

# Test model using sklearn logistic regression?
if do_sklearn_fit:
    from sklearn.linear_model import LogisticRegression

    # Fit with Logistic Regressor
    lr = LogisticRegression(C=2) # Hand-tunes regularization for this problem
    lr.fit(X,y)
    print(lr.coef_,lr.coef_.shape)

# Now try with my gradient descent implementation using adaptive step sizes
model = cu.logistic_model
w0, w_pred = gd.gradient_descent(model, X, y, lam=20.0, eta = 100, max_iter = 500, adaptive=True)

print(w_pred)