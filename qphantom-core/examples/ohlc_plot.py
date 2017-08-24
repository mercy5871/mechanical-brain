# %%

from matplotlib import pyplot as plt
import pandas as pd
import os
from QPhantom.core.metrics import Metrics

from datetime import datetime

M = Metrics(size=(16, 16), fontsize=12, title_size=14, text_size=13, dpi=200)

value_df = pd.read_csv('data/601607.csv')

def plot_score(ax, times):
    ax.set_ylabel("Score", fontsize=M.fontsize)
    ax.grid(False)
    ax.grid(axis='y')
    ax.set_ylim([0, 1])
    ax.plot(times, value_df['score'], color='0.5', alpha=0.5)
    ax.legend(loc="upper left")

M.plot_ohlc_summary("601607.SH",
    times=value_df["trade_date"],
    opens=value_df["w_open"],
    highs=value_df["w_high"],
    lows=value_df["w_low"],
    closes=value_df["w_close"],
    volumes=value_df["w_vol"] / 10000,
    major_unit="month",
    minor_unit="day",
    extra_plots=[plot_score]
)
