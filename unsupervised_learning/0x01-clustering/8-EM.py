#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """ performs the expectation maximization for a GMM:

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return (None, None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) != float or tol <= 0:
        return (None, None, None, None, None)
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    l_past = 0

    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if (verbose is True):
            if (i % 10 == 0) or (i == 0):
                print("Log Likelihood after {} iterations: {}".format(
                    i, log_l))
            if (i == iterations - 1):
                print("Log Likelihood after {} iterations: {}".format(
                    i, log_l))
            if abs(log_l - l_past) <= tol:
                print("Log Likelihood after {} iterations: {}".format(
                    i, log_l))
                break

        if abs(log_l - l_past) <= tol:
            break

        l_past = log_l

    return (pi, m, S, g, log_l)
