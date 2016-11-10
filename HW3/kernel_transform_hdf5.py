from __future__ import print_function, division
import numpy as np
import sys
sys.path.append("..")
import DML.pca.pca as pca
import DML.data_processing.mnist_utils as mu
import DML.optimization.gradient_descent as gd
import DML.kernel.kernel as kernel
import h5py
import time


"""
# Load in MNIST data
print("Loading MNIST Training data...")
X_train, y_train = mu.load_mnist(dataset='training')

#print("Loading MNIST Testing data...")
#X_test, y_test = mu.load_mnist(dataset='testing')

# Init PCA object
# Solve for all principal components but do calculations using only 50
# Can reset l later if need be as all principal components are retained
PCA = pca.PCA(l=50, center=True)

print("Training the model...")
PCA.fit(X_train)
X_train = PCA.transform(X_train)

# Estimate kernel bandwidth
sigma = kernel.estimate_bandwidth(X_train, num = chunk_size, scale = 10.0)
print("Estimted kernel bandwidth: %lf" % sigma)

# Open hdf5 file
ii = 0
with h5py.File('kernel_train.hdf5', 'w') as outfile:
    dset = outfile.create_dataset("train", (60000, 60000), chunks=(chunk_size, 60000))

    for batch in gd.make_batches(X_train, y_train ,size=chunk_size):
        print(ii)
        X_b = batch[0].reshape((chunk_size,batch[0].shape[-1]))

        X_b = kernel.RBF(X_b, X_train, sigma=sigma)

        dset[ii:ii + chunk_size] = X_b

        ii += chunk_size
print("Done!")
"""
