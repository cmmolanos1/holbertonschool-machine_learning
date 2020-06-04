#!/usr/bin/env python3
"""
Multikernels
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images with multiple kernels.

    Args:
        images (np.ndarray): matrix of shape (m, h, w, c) containing multiple
                             grayscale images.
        kernels (np.ndarray): matrix of shape (kh, kw, c, nc) containing the
                              kernel for the convolution. nc stands for number
                              of kernels.
        padding: if same: performs same padding,
                 if valid: valid padding.
                 if tuple (ph, pw) = padding for the height,
                                     padding for the weight.

    Returns:
        np.ndarray: The convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if type(padding) == tuple:
        ph, pw = padding

    padded_img = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant')

    ch = int(((h + 2 * ph - kh) / sh) + 1)
    cw = int(((w + 2 * pw - kw) / sw) + 1)

    convolved = np.zeros((m, ch, cw, c))

    for j in range(ch):
        for k in range(cw):
            for n in range(nc):

                images_slide = padded_img[:,
                                          j * sh:j * sh + kh,
                                          k * sw:k * sw + kw]
                kernel = kernels[:, :, :, n]
                elem_mul = np.multiply(images_slide, kernel)
                convolved[:, j, k, n] = elem_mul.sum(axis=1).sum(axis=1).\
                    sum(axis=1)

    return convolved
