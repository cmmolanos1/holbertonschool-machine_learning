#!/usr/bin/env python3

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

    ch = h - kh + 1
    cw = w - kw + 1

    if ch != h or cw != w:
        ph = int(kh / 2)
        pw = int(kw / 2)
        padded_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    convolved = np.zeros((m, h, w))

    for j in range(ch):
        for k in range(cw):
            images_slide = padded_img[:, k:k + kh, j:j + kw]
            elem_mul = np.multiply(images_slide, kernel)
            convolved[:, k, j] = elem_mul.sum(axis=1).sum(axis=1)

    return convolved
