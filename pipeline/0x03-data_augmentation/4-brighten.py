#!/usr/bin/env python3
"""
rotate
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """randomly changes the brightness of an image

    Args:
        image (tf.tensor): 3D tensor containing the image to flip.
        max_delta: the maximum amount the image should be brightened
                   (or darkened).

    Returns:
         the altered image.
    """
    return tf.image.adjust_brightness(image, max_delta)
