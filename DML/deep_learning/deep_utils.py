"""

@author: dflemin3 [David P. Fleming, University of Washington, Seattle]

@email: dflemin3 (at) uw (dot) edu

This file contains routines for deep learning, specifically for running a
neural network with l layers.  Some of the helper functions are redefined
numpy functions located here for convienence, doc purposes.  Stupid, I know, but
future confused me might thank me later.

"""

from __future__ import print_function, division
import numpy as np
from ..optimization import gradient_descent as gd


def linear(x):
    """
    Trivial mapping for convienence

    Parameters
    ----------
    x : float, array

    Returns
    -------
    x : float, array
    """
    return x
# end function


def linear_prime(x):
    """
    Derivative of trivial mapping

    Parameters
    ----------
    x : float, array

    Returns
    -------
    x' : float, array
    """
    return 1.0
# end function


def sigmoid(x):
    """
    Compute the sigmoid function 1.0 / (1.0 e^-x)

    Parameters
    ----------
    x : float, array

    Returns
    -------
    sigmoid(x) : float, array
    """
    return 1.0 / (1.0 + np.exp(-x))
# end function


def sigmoid_prime(x):
    """
    Derivative of the sigmoid function.

    Parameters
    ----------
    x : float, array

    Returns
    -------
    sigmoid'(x) : float, array
    """
    return sigmoid(x) * (1.0 - sigmoid(x))
# end function


def relu(x):
    """
    Compute the rectified linear function, aka max(0,x).  Not sure why people
    call it that but whatever.

    Parameters
    ----------
    x : float, array

    Returns
    -------
    relu(x) : float, array
    """
    return np.maximum(0.0,x)
# end function


def relu_prime(x):
    """
    Derivative of the rectified linear function, aka max(0,x).

    Parameters
    ----------
    x : float, array

    Returns
    -------
    relu'(x) : float, array
    """
    if x > 0:
        return 1.0
    else:
        return 0.0
relu_prime = np.vectorize(relu_prime) # Vectorize!
# end function


def tanh(x):
    """
    Compute the hyperbolic tangent of x.

    Parameters
    ----------
    x : float, array

    Returns
    -------
    tanh(x) : float, array
    """
    return np.tanh(x)
# end function


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent of x.

    Parameters
    ----------
    x : float, array

    Returns
    -------
    tanh'(x) : float, array
    """
    return 1.0 - tanh(x)**2
# end function


def initialize_weights(X, d, nodes, activator=None, scale=1.0, input_layer=False):
    """
    Compute the initial weight matrix for a neural network hidden layer
    depending on which activation function is used.  Weights are initialized
    according to some sort of normal distribution to break symmetry and to
    ensure that W dot z ~ 1 for each layer.

    Parameters
    ----------
    X : array (n x d)
        input data
    d : int
        number of features
    nodes : int
        number of nodes for hidden layer
    activator : str
        Activation functions for each layer.  Currently support sigmoid, tanh,
        relu, and linear.
    scale : float (optional)
        Amount to scale weights.  Defaults to 1.0.
    input_layer : bool (optional)
        whether or not this layer is an input layer.  Defaults to False

    Returns
    -------
    W : weight matrix
    """

    # If none given, default to relu as it works well
    if activator is None:
        raise RuntimeError("No activator specified!")

    # Init weights depending on activation function
    if activator is sigmoid or activator is tanh:
        print("Using sigmoid/tanh layer...")
        if input_layer:
            sigma = scale/(np.mean(X)**2) # Assuming mean(X) ~ E[X]
            W = np.random.normal(loc=0.0, scale=sigma, size=(d,nodes))
        else:
            sigma = scale/np.sqrt(d)
            W = np.random.normal(loc=0.0, scale=sigma, size=(d,nodes))
    elif activator is relu or activator is linear:
        print("Using relu/linear layer...")
        sigma = scale
        W = np.random.normal(loc=0.0,scale=sigma,size=(d,nodes))
    else:
        raise RuntimeError("No such activator: %s" % activator)

    return W
# end function


def simple_scale(y_hat, w_1, w_2, scale=1.0):
    """
    Perform a simple y_hat scaling so E[y_hat] ~ E[y].  Here I don't touch the
    weight vectors but keep them for compatibility

    Parameters
    ----------
    y_hat : array
        array to scale
    w_1 : array
        weight vector I don't use
    w_2 : array
        other weight vector I also don't use
    scale : float (optional)
        extra scaling factor because why not? Defaults to 1.0

    Returns
    -------
    y_hat : array
        scaled y_hat
    """
    return scale*y_hat/y_hat.shape[0], w_1, w_2
# end function


def trivial_scale(y_hat, w_1, w_2, scale=1.0):
    """
    Trivial scaling function that doesn't do anything.
    """
    return y_hat, w_1, w_2
# end function


def neural_net(X, y, nodes=50, activators=None, activators_prime=None,
               scale=1.0, eps=1.0e-3, eta=1.0e-3, lam=0.0, scaler=None,
               adaptive=False, nout=None, batchsize=10, nclass=1):
    """
    Train a 1 hidden layer neural net classifier optimized using SGD

    Parameters
    ----------
    X : array (n x d)
        training array
    y : array (n x 1)
        training labels
    nodes : int (optional)
        number of nodes for hidden layer
    activators : list
        list of activation functions
    activators_prime : list
        list of derivatives of activation functions
    scale : float (optional)
        amount by which initial weight vectors are scaled.  Defaults to 1.0
    eps : float (optional)
        loss convergence criterion.  Defaults to 1.0e-3.
    eta : float (optional)
        learning rate
    lam : float (optional)
        l2 regularization constant
    adaptive : bool (optional)
        whether or not to use and adaptive learning rate
    nclass : int (optional)
        number of classes.  Defaults to binary classification (nclass = 1)

    Returns
    -------
    ?
    """

    #Define values
    if batchsize is None:
        n = 10 # Default to minibatch SGD
    else:
        n = batchsize
    d = X.shape[-1]

    # Set up scaling function
    if scaler is None:
        scaler = trivial_scale

    # Default epoch size is n_training_samples
    if nout is None:
        nout = X.shape[0]

    # Set up activation functions and their derivatives?
    if activators is None:
        activators = [tanh,linear]
    if activators_prime is None:
        activators_prime = [tanh_prime, linear_prime]

    # Initialize weight matrices for each layer-layer interface
    # Input -> Hidden layer
    w_1 = initialize_weights(X, d, nodes, activator=activators[0], scale=scale) # d x nodes

    # Hidden layer -> output
    w_2 = initialize_weights(X, nodes, nclass, activator=activators[1], scale=scale) # nodes x nclass

    # Init bias terms
    b_1 = np.zeros((1,nodes))
    b_2 = np.zeros((1,nclass))

    # SGD params setup
    converged = False
    old_loss = 1.0e10
    loss = old_loss
    iters = 0
    scale = 1.0/n

    # Main loop
    while not converged:

        # Randomly permute X, y
        inds = np.random.permutation(X.shape[0])
        X_per = X[inds]
        y_per = y[inds]

        # Loop over batches
        ii = 0
        for batch in gd.make_batches(X_per, y_per, size=n):
            X_b = batch[0].reshape((n,batch[0].shape[-1]))
            y_b = batch[1].reshape((n,batch[1].shape[-1]))

            # 1: Feed forward pass
            # Inputs -> hidden layer
            z_1 = np.dot(X_b,w_1) + b_1
            a_hidden = activators[0](z_1) # n x nodes

            # Hidden layer -> output layer
            z_2 = np.dot(a_hidden,w_2) + b_2
            a_out = activators[1](z_2) # n x nclass

            # 2: Compute delta for output layer, hidden layer
            # change z->a?
            delta_out = -(y_b - a_out)*activators_prime[1](z_2) # n x nclass
            delta_hidden = np.dot(delta_out,w_2.T)*activators_prime[0](z_1) # nodes x nclass

            # 3: Compute gradients
            grad_w2 = np.dot(a_hidden.T,delta_out)
            grad_w1 = np.dot(X_b.T,delta_hidden) # d x nodes

            grad_b2 = np.sum(delta_out, axis=0, keepdims=True)
            grad_b1 = np.sum(delta_hidden, axis=0, keepdims=True)

            # 4: Update parameters!
            w_1 = w_1 - scale*eta*(w_1*lam + grad_w1)
            b_1 = b_1 - scale*eta*grad_b1

            w_2 = w_2 - scale*eta*(w_2*lam + grad_w2)
            b_2 = b_2 - scale*eta*grad_b2

            # If epoch is complete (or 1st iter), compute losses for this fit
            if (ii+n) % nout == 0 or iters == 0:

                # Compute y_hat over entire training set
                a_hidden = activators[0](np.dot(X,w_1) + b_1)
                y_hat = activators[1](np.dot(a_hidden,w_2) + b_2)
                loss = np.sum(np.power(y - y_hat,2))/len(y) # Square loss
                loss_01 = np.sum(np.argmax(y_hat, axis=1) != np.argmax(y, axis=1))/y.shape[0]
                print(loss, loss_01)

                # Using an adaptive step size?
                if adaptive:
                    scale = 1.0/(n * np.sqrt(1.0 + iters))

                # Is it converged (is loss not changing by some %?)
                if np.fabs(loss - old_loss)/np.fabs(old_loss) > eps:
                    converged = False
                else:
                    converged = True
                    break

                # Store old loss, iterate
                old_loss = loss
                iters += 1
            ii = ii + n

    return y_hat, w_1, w_2, b_1, b_2
# end function
