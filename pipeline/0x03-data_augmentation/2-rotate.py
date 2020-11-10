#!/usr/bin/env python3
"""
rotate
"""

import tensorflow as tf


def rotate_image(image):
    """rotates an image by 90 degrees counter-clockwise

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.

    Returns:
        The rotated image.
    """
    return tf.image.rot90(image, k=1)
