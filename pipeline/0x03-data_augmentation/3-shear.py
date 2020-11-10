#!/usr/bin/env python3
"""
rotate
"""

import tensorflow as tf


def shear_image(image, intensity):
    """randomly shears an image.

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.
        intensity: intensity with which the image should be sheared.

    Returns:
        the sheared image.
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity)
