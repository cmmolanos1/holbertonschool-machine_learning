#!/usr/bin/env python3
"""
Flip
"""

import tensorflow as tf


def flip_image(image):
    """flips an image horizontally.

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.

    Returns:
        The flipped image.
    """
    return tf.image.flip_left_right(image)
