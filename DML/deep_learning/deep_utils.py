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
    return np.fmax(0.0,x)
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


def initialize_weights(X, d, nodes, activator=None, scale=1.0):
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

    Returns
    -------
    W : weight matrix
    """

    # If none given, default to relu as it works well
    if activator is None:
        activator = "relu"

    # Init weights depending on activation function
    if activator is sigmoid or activator is tanh:
        sigma = scale/(np.mean(X)**2) # Assuming mean(X) ~ E[X]
        W = np.random.normal(loc=0.0, scale=sigma, size=(d,nodes))
    elif activator is relu:
        sigma = scale/np.sqrt(nodes)
        W = np.random.normal(loc=0.0, scale=sigma, size=(d,nodes))
    elif activator is linear:
        W = np.random.normal(size=(d,nodes))
    else:
        raise RuntimeError("No such activator: %s" % activator)

    return W
# end function


def neural_net(X, y, nodes=50, activators=None, activators_prime=None,
               scale=1.0, eps=1.0e-3, eta=1.0e-3):
    """
    Train a neural net classifier that is optimized using SGD with one hidden
    layer.  For now, I don't include an offset term

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

    Returns
    -------
    ?
    """

    # Inits
    n = X.shape[0]
    d = X.shape[-1]

    # Set up activation functions and their derivatives?
    if activators is None:
        activators = [relu,linear]
    if activators_prime is None:
        activators_prime = [relu_prime, linear_prime]

    # Initialize weight matrices for each layer-layer interface

    # Input -> Hidden layer
    w_1 = initialize_weights(X, d, nodes, activator=activators[0], scale=scale)

    # Hidden layer -> output
    w_2 = initialize_weights(X, nodes, 1, activator=activators[1], scale=scale)

    converged = False
    old_loss = 1.0e10
    loss = old_loss

    # Main loop
    while not converged:
        old_loss = loss

        # 1: Feed forward pass

        # Inputs -> hidden layer
        a_hidden = activators[0](X.dot(w_1)) # n x nodes

        # Hidden layer -> output layer
        a_out = activators[1](a_hidden.dot(w_2)) # n x 1

        # 2: Compute delta for output layer, hidden layer
        delta_out = -(y - a_out)*activators_prime[1](a_hidden.dot(w_2)) # n x 1
        delta_hidden = delta_out.dot(w_2.T)*activators_prime[0](X.dot(w_1)) # nodes x 1

        # 3: Compute gradients
        grad_w1 = X.T.dot(delta_hidden)
        grad_w2 = a_out.T.dot(delta_out)

        # 4: Update parameters!
        w_1 = w_1 - eta*grad_w1/n
        w_2 = w_2 - eta*grad_w2/n

        # Compute losses

        # Compute y_hat
        a_hidden = activators[0](X.dot(w_1))
        y_hat = activators[1](a_hidden.dot(w_2))
        loss = np.sum(np.power(y - y_hat,2))/len(y)

        print(loss)

        # Has the solution converged?
        if np.fabs((loss - old_loss)/old_loss) > eps:
            converged = False
        else:
            converged = True
            break

    return y_hat
# end function
