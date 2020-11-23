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
    #return tf.keras.preprocessing.image.random_shear(image, intensity)
    image = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(image,
                                                        intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)

    return tf.keras.preprocessing.image.array_to_img(sheared)