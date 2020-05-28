#!/usr/bin/env python3
""" regularization l2 tensorflow"""

import tensorflow as tf


def l2_reg_cost(cost):
    """ regularization l2 tensorflow"""
    return cost + tf.losses.get_regularization_losses()
