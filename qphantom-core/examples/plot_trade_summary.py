# %%

from matplotlib import pyplot as plt
import pandas as pd
import os
from QPhantom.core.metrics import Metrics

from datetime import datetime

M = Metrics(size=(32, 16), fontsize=4, title_size=12, text_size=11, dpi=200)

value_df = pd.read_csv('data/value_data.csv')

M.plot_trade_summary(pd.to_datetime(value_df['date']), value_df['value'], value_df['benchmark'])
