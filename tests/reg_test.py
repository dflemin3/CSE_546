# Test functions using my logistic regressor and sklearn's implementation
import sys
sys.path.append("..")
import DML.regression.ridge_utils as ri
import DML.regression.lasso_utils as lu
import DML.regression.regression_utils as ru
import time

print("Testing ridge regression...")
seed = 1
sparse = False
w, X, y = ru.generate_norm_data(10000,7,10,sparse=sparse,seed=seed)

print(w.shape,X.shape,y.shape)

print("Performing ridge regression...")
print(ri.fit_ridge(X,y,lam=10.0))
print(w)

print("Testing lasso regression...")
# Generate some fake data
n = 10000
d = 75
k = 5
lam = 500.0
sparse = True
seed = 1
w, X, y = ru.generate_norm_data(n,k,d,sigma=1,sparse=sparse,seed=seed)

# What should the maximum lambda in a regularization step be?
print("Lambda_max:",lu.compute_max_lambda(X,y))

print("Performing LASSO regression...")
start = time.time()
w_0_pred, w_pred = lu.fit_lasso_fast(X,y,lam=lam,sparse=sparse)
end = time.time()
print("fast:",end-start)

print("w_pred:",w_pred)
print(w)

# Was the predicted solution correct?
print(lu.check_solution(X,y,w_pred,w_0_pred,lam=lam))