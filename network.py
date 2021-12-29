"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Standard library
import random

# Third-party libraries
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def cost_derivative(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return output_activations - y


def cost_derivative_cross_entropy(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    # same derivative as quadratic function
    return output_activations - y


class Network(object):

    def __init__(self, sizes, is_binary_classification=False):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.is_binary_classification = is_binary_classification

        # weights and biases here are lists on numpy arrays
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        # Shape of training data:
        # The image vector of the digit: training_data[-1][0].shape: (784, 1)
        # The digit it self            : training_data[-1][1].shape: (10, 1)
        # We nee to learn the "representation" of the digits

        # n: the number of total training samples (digits)
        n = len(training_data)

        n_test = 0
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            # shuffle data before each epoch
            random.shuffle(training_data)

            # mini_batches: list with parts of training data as batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # for each mini batch do learn weights
            for mini_batch in mini_batches:
                # Below is the CORE LEARNING function
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                if self.is_binary_classification:
                    acc, f1 = self.evaluate(test_data)
                    print("Epoch {} : test accuracy: {} test AUC: {} / # of test samples: {}".
                          format(j, acc, f1, n_test))
                else:
                    acc, f1 = self.evaluate(test_data)
                    print("Epoch {} : test accuracy: {} test F1 (macro): {} / # of test samples: {}".
                          format(j, acc, f1, n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        # these are 2 arrays just to keep gradient sums across the mini batch samples
        # for weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For every training sample (x, y) in the mini batch
        for x, y in mini_batch:
            # Find the gradient deltas for weights and biases
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Append the gradient deltas for weights and biases gradient sums
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Adjust (UPDATE) weights and biases
        # use the averages of weights and biases gradients
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # create the nablas, just set a number, for our case zero
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # steps 1. and 2
        # feed-forward pass for the training example (x,y)
        activation = x

        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # z is vector of scalars after dot product
            # with the dimension of teh i-yh hidden layer (how many nodes)
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # steps 3. and 4.
        # backward pass

        # last (output) layer calculations: list[-1]
        # cost_derivative: dCost / dsigmoid  = out - target ( = output_activations-y)
        # sigmoid_prime  : dsigmoid / dzeta  = sigmoid(zeta)*(1-sigmoid(zeta))

        #                : dzeta / dw        = activation

        # biases
        # below is the delta of the last layer
        # Note: nablas are the partial derivatives of the cost function with respect to weights and biases
        if self.is_binary_classification:
            delta = cost_derivative_cross_entropy(activations[-1], y)
        else:
            delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta

        # weights
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        '''
        Existing evaluation implementation
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

        '''
        if self.is_binary_classification:
            nn_test_scores = [(np.asscalar(self.feedforward(x)))
                              for (x, y) in test_data]
        else:
            nn_test_scores = [(np.argmax(self.feedforward(x)))
                              for (x, y) in test_data]

        y_true = [y for (x, y) in test_data]

        if self.is_binary_classification:
            return roc_auc_score(y_true, nn_test_scores), roc_auc_score(y_true, nn_test_scores)
        else:
            return accuracy_score(y_true, nn_test_scores), f1_score(y_true, nn_test_scores, average='macro')


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    # See: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    return sigmoid(z) * (1 - sigmoid(z))
