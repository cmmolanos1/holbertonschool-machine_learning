#!/usr/bin/env python3
"""
From file
"""

import pandas as pd


def from_file(filename, delimiter):
    """loads data from a file as a pd.DataFrame

    Args:
        filename (str): the file to load from.
        delimiter (char): the column separator.

    Returns:
        the loaded pd.DataFrame.
    """
    return pd.read_csv(filename, sep=delimiter)
