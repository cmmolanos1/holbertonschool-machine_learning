#!/usr/bin/env python3
"""
Pooling backward
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network.

    Args:
        dA (np.ndarray): matrix of shape (m, h_new, w_new, c_new) containing
                         the partial derivatives with respect to the output
                         of the pooling layer
        A_prev (np.ndarray): matrix of shape (m, h_prev, w_prev, c) containing
                             the output of the previous layer.
        kernel_shape (tuple): (kh, kw) containing the size of the kernel for
                              the pooling.
        stride (tuple): (sh, sw) containing the strides for the pooling.
        mode (str): max or avg, indicating whether to perform maximum or
                    average pooling, respectively.

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev).
    """

    m, h_prev, w_prev, _ = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = h * sh + kh
                    horiz_start = w * sw
                    horiz_end = w * sw + kw

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:
                                              vert_end, horiz_start:
                                              horiz_end, c]
                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i, vert_start: vert_end,
                                horiz_start: horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        average = da / (kh * kw)
                        distribute = np.ones(kernel_shape) * average
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += distribute

    return dA_prev
