#!/usr/bin/env python3
"""
Flip
"""

import tensorflow as tf


def crop_image(image, size):
    """performs a random crop of an image.

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.
        size (tuple): the size of the crop.

    Returns:
        The cropped image.
    """
    return tf.image.random_crop(image, size)
