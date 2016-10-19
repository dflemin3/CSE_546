# Test mnist data importing functions
import sys
sys.path.append("..")
import DML.data_processing.mnist_utils as mu

# Test data loading, make sure shape is correct
images, labels = mu.load_mnist(dataset='training')

print(images.shape,labels.shape)
