#!/usr/bin/env python3
"""
From numpy
"""

import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray.

    Args:
        array (np.ndarray): which you should create the pd.DataFrame.

    Returns:
        the newly created pd.DataFrame
    """
    col_label = [chr(65 + i) for i in range(array.shape[1])]

    return pd.DataFrame(data=array,
                        columns=col_label)
