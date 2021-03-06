#!/usr/bin/env python3
"""
Same
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images (np.ndarray): matrix of shape (m, h, w) containing multiple
                             grayscale images.
        kernel (np.ndarray): matrix of shape (kh, kw) containing the kernel
                             for the convolution.

    Returns:
        np.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(kh / 2)
    pw = int(kw / 2)
    padded_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    convolved = np.zeros((m, h, w))

    for j in range(h):
        for k in range(w):
            images_slide = padded_img[:, j:j + kh, k:k + kw]
            elem_mul = np.multiply(images_slide, kernel)
            convolved[:, j, k] = elem_mul.sum(axis=1).sum(axis=1)

    return convolved
