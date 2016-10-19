# Test functions using my logistic regressor and sklearn's implementation
import sys
sys.path.append("..")
import DML.classification.classifier_utils as cu
from sklearn.linear_model import LogisticRegression

# Fake classifier parameters
n = 10000
k = 5
d = 10
w0 = 0
seed = 42

w, X, y = cu.generate_binary_data(n, k, d, w0=w0, seed=seed)
print(w)

# Fit with Logistic Regressor
lr = LogisticRegression(C=2) # Hand-tunes regularization for this problem
lr.fit(X,y)
print(lr.coef_,lr.coef_.shape)
