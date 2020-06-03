#!/usr/bin/env python3

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

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

    new_kernel = np.array([kernel] * m)

    ch = h - kh + 1
    cw = w - kw + 1
    convolved = np.zeros((m, ch, cw))

    for j in range(ch):
        for k in range(cw):
            images_slide = images[:, k:k + kh, j:j + kw]
            elem_mul = np.multiply(images_slide, new_kernel)
            convolved[:, k, j] = elem_mul.sum(axis=1).sum(axis=1)

    return convolved
