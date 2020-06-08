#!/usr/bin/env python3
"""
Pooling forward
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a convolutional layer of a neural
    network.

    Args:
        A_prev (np.ndarray): matrix of shape (m, h_prev, w_prev, c_prev)
                             containing the output of the previous layer.
        kernel_shape (tuple): (kh, kw) = kernel hight, kernel weight.
        stride (tuple): shape (sh, sw) containing the strides for the
                        convolution.
        mode (str): 'max' for maxpool, 'avg' for avgpool.

    Returns:
        The output of the pooling layer.

    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    poolh = int((h_prev - kh) / sh) + 1
    poolw = int((w_prev - kw) / sw) + 1

    pooled = np.zeros((m, poolh, poolw, c_prev))

    for j in range(poolh):
        for k in range(poolw):
            images_slide = A_prev[:,
                                  j * sh:j * sh + kh,
                                  k * sw:k * sw + kw]
            if mode == 'max':
                pooled[:, j, k] = np.max(images_slide, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, j, k] = np.mean(images_slide, axis=(1, 2))

    return pooled
