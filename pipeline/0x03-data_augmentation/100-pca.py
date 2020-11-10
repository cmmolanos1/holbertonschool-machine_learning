#!/usr/bin/env python3
"""
rotate
"""

import numpy as np
import tensorflow as tf


def pca_color(image, alphas):
    """that performs PCA color augmentation as described in the
    AlexNet paper.

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.
        delta (tuple): length 3 containing the amount that each channel should
                       change.

    Returns:
         the augmented image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    orig_img = img.astype(float).copy()

    img = img / 255.0

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alphas)

    # broad cast to speed things up
    m2[:, 0] = alphas * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img
