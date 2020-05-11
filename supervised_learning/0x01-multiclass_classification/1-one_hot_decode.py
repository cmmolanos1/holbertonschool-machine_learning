#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    if isinstance(one_hot, np.ndarray) is False:
        return 0

    return np.argmax(one_hot, axis=0)
