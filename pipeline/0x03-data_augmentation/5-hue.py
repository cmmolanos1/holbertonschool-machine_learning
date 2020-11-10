#!/usr/bin/env python3
"""
rotate
"""

import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.
        delta: the amount the hue should change.

    Returns:
         the altered image
    """
    return tf.image.adjust_hue(image, delta)
