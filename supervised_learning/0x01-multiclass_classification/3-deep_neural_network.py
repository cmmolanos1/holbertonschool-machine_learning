#!/usr/bin/env python3
"""First Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor.

        Args:
            nx (int): number of inputs (Xi).
            layers (list): the number of nodes in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        arr_layers = np.array(layers)
        len_pos = arr_layers[arr_layers >= 1].shape[0]
        if len(layers) == 0 or len_pos != len(layers) or \
                isinstance(arr_layers[0], np.integer) is False:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for la in range(self.__L):
            # Layer 0 = X, Layers = [L1_size, L2_size, ..., Ln_size]
            key_w = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)

            if la == 0:
                # Layer 0 = X, Size Layer 0 = nx
                weight = np.random.randn(layers[la], nx) * np.sqrt(2 / nx)
                self.__weights[key_w] = weight
            else:
                weight = np.random.randn(layers[la], layers[la - 1]) * \
                         np.sqrt(2 / layers[la - 1])
                self.__weights[key_w] = weight

            biases = np.zeros((layers[la], 1))
            self.__weights[key_b] = biases

    @property
    def L(self):
        """ A attribute getter.

        Returns:
            int: Private A.
        """
        return self.__L

    @property
    def cache(self):
        """ cache attribute getter.

        Returns:
            int: Private A.
        """
        return self.__cache

    @property
    def weights(self):
        """ A attribute getter.

        Returns:
            int: Private A.
        """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates forward propagation of the neural network.

        Args:
            X (ndarray): input data (number X, number examples).

        Returns:
            ndarray: vector with activation results using sigmoid.
            dict: dictionary with all activation vectors.
        """
        self.__cache['A0'] = X

        for la in range(self.__L):
            key_W = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)
            key_A = "A{}".format(la)
            key_newA = "A{}".format(la + 1)

            W = self.__weights[key_W]
            A = self.__cache[key_A]
            b = self.__weights[key_b]
            z = np.matmul(W, A) + b

            if la < self.__L - 1:
                # Sigmoid
                activation = 1 / (1 + np.exp(-z))
            else:
                # Softmax
                t = np.exp(z)
                activation = t / np.sum(t, axis=0, keepdims=True)

            self.__cache[key_newA] = activation

        return activation, self.__cache

    def cost(self, Y, A):
        """Calculates the function cost of the neuron.
        It was applied the Logistic Regression Cost Function.

        Args:
            Y (ndarray): the actual or correct values for the input data.
            A (ndarray): activated output of the neuron for each example.

        Returns:
            float: calculated cost, better if cost -> 0 .
        """
        constant = -1 / A.shape[1]
        summation = Y * np.log(A)
        return constant * summation.sum()

    def evaluate(self, X, Y):
        """Evaluates neural network predictions.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.

        Returns:
            ndarray: activated output of neuron, where if A[i] >= 0.5
                     the output is 1, otherwise 0.
            float: the neuron's cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A == np.amax(A, axis=0), 1, 0)
        return (A, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the network.

        Args:
            Y (ndarray): the actual or correct values for the input data.
            cache (dict): contents all activation units for each layer.
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dz = {}
        dW = {}
        db = {}

        for la in reversed(range(1, self.__L + 1)):
            A = cache["A{}".format(la)]
            A_prev = cache["A{}".format(la - 1)]

            # 3
            if la == self.__L:
                kdz = "dz{}".format(la)
                kdW = "dW{}".format(la)
                kdb = "db{}".format(la)

                dz[kdz] = A - Y
                dW[kdW] = np.matmul(dz[kdz], A_prev.T) / m
                db[kdb] = dz[kdz].sum(axis=1, keepdims=True) / m
            else:
                # 2 - 1
                kdz_n = "dz{}".format(la + 1)
                kdz_c = "dz{}".format(la)
                kdW_n = "dW{}".format(la + 1)
                kdW = "dW{}".format(la)
                kdb_n = "db{}".format(la + 1)
                kdb = "db{}".format(la)
                kW = 'W{}'.format(la + 1)
                kb = 'b{}'.format(la + 1)

                W = self.__weights[kW]
                dz[kdz_c] = np.matmul(W.T, dz[kdz_n]) * (A * (1 - A))
                dW[kdW] = np.matmul(dz[kdz_c], A_prev.T) / m
                db[kdb] = dz[kdz_c].sum(axis=1, keepdims=True) / m

                self.__weights[kW] -= alpha * dW[kdW_n]
                self.__weights[kb] -= alpha * db[kdb_n]

                if la == 1:
                    self.__weights['W1'] -= alpha * dW['dW1']
                    self.__weights['b1'] -= alpha * db['db1']

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron, performing gradient_descent iterations times.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.
            iterations (int): number of times the function will train the
                              neuron.
            alpha (float): learning rate.
            verbose (bool): If true, prints the cost after step iterations.
            graph (bool): If true, prints the plot iterations vs cost.
            step (int): number of iteration to print message and graph.

        Returns:
            ndarray: activated output of neuron at the end of training,
                     where if A[i] >= 0.5 the output is 1, otherwise 0.
            float: the neuron's cost at the end of training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        print_step = 0
        cost_axis = np.zeros(iterations + 1, )

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if step and (i == print_step or i == iterations):
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                    print_step += step

            if graph is True:
                cost_axis[i] = cost

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost_axis)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format.

        Args:
            filename (str): filename for the new file.

        Returns:

        """
        file_Name = filename
        if '.pkl' not in file_Name:
            file_Name = file_Name + '.pkl'

        fileObject = open(file_Name, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()

    @staticmethod
    def load(filename):
        """Loads picker file.

        Args:
            filename (str): name of picker file.

        Returns:
            obj: the object inside the file.
        """
        try:
            fileObject = open(filename, 'rb')
            neural = pickle.load(fileObject)
            fileObject.close()
            return neural
        except FileNotFoundError:
            return None
