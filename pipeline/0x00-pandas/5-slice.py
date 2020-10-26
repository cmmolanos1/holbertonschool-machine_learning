#!/usr/bin/env python3

import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

columns = [df.columns.get_loc(c) for c in
           ['High', 'Low', 'Close', 'Volume_(BTC)']]

df = df.iloc[::60, columns]

print(df.tail())
