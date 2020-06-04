#!/usr/bin/env python3
"""
Grayscale
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    Args:
        images (np.ndarray): matrix of shape (m, h, w) containing multiple
                             grayscale images.
        kernel (np.ndarray): matrix of shape (kh, kw) containing the kernel
                             for the convolution.
        padding: if same: performs same padding,
                 if valid: valid padding.
                 if tuple (ph, pw) = padding for the height,
                                     padding for the weight.

    Returns:
        np.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if type(padding) == tuple:
        ph, pw = padding

    padded_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    ch = int(((h + 2 * ph - kh) / sh) + 1)
    cw = int(((w + 2 * pw - kw) / sw) + 1)

    convolved = np.zeros((m, ch, cw))

    for j in range(ch):
        for k in range(cw):
            convolved[:, j, k] = (np.sum(padded_img[:,
                                                    j*sh:(kh+(j*sh)),
                                                    k*sw:(kw+(k*sw))] *
                                         kernel, axis=(1, 2)))
    return convolved
