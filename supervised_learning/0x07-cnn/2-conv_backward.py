#!/usr/bin/env python3
"""
Pooling forward
"""

import numpy as np

z
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

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2))

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant')

    dA_prev_pad = np.pad(dA_prev,
                         ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         'constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = h * sh + kh
                    horiz_start = w * sw
                    horiz_end = w * sw + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end,
                                         :]

                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad[:, :, :]

    return dA_prev, dW, db
