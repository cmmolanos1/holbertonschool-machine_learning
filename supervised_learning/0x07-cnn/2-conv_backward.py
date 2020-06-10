#!/usr/bin/env python3
"""
Pooling forward
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural
    network.

    Args:
        dZ (np.ndarray): matrix of shape (m, h_new, w_new, c_new) containing
                         the partial derivatives with respect to the
                         unactivated output of the convolutional layer.
        A_prev (np.ndarray): matrix of shape (m, h_prev, w_prev, c_prev)
                             containing the output of the previous layer.
        W (np.ndarray): matrix of shape (kh, kw, c_prev, c_new) containing the
                        kernels for the convolution.
        b (np.ndarray): matrix of shape (1, 1, 1, c_new) containing the biases
                        applied to the convolution.
        padding (str): same or valid, indicating the type of padding used.
        stride (tuple): (sh, sw) containing the strides for the convolution.

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant')
    dA_prev = np.zeros_like(A_prev_pad)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = h * sh + kh
                    horiz_start = w * sw
                    horiz_end = w * sw + kw

                    dA_prev[i,
                            vert_start:vert_end,
                            horiz_start:horiz_end:,
                            :] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += A_prev_pad[i,
                                                 vert_start:vert_end,
                                                 horiz_start:horiz_end,
                                                 :] * dZ[i, h, w, c]

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev

    return dA_prev, dW, db
