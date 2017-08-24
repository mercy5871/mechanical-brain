# %%

import unittest
import pandas as pd
import numpy as np
import bottleneck as bn
from QPhantom.core.quant import Builder
from QPhantom.core.utils import measureTime
from QPhantom.core.metrics import Metrics

def equal(a, b, eps=1e-8):
    return abs(a - b) < eps

stock_data = pd.read_hdf('data/stock_sample.hdf5', key='A')
unweighted = ['open', 'high', 'low', 'close']
w_cols = ['w_{col}'.format(col=c) for c in unweighted]
#stock_data[w_cols]
stock_data[w_cols] = stock_data[unweighted].multiply(stock_data['factor'], axis=0)
stock_data['w_volume'] = stock_data['volume'] / stock_data['factor']
stock_data['w_avg'] = stock_data['amount'] / stock_data['w_volume']

base_cols = ['w_open', 'w_high', 'w_low', 'w_close', 'w_avg', 'w_volume', 'amount', 'iclose']
log_cols = [f'log_{w}' for w in base_cols]
log_r_cols = [f'logr_{w}' for w in base_cols]
df = pd.DataFrame(np.log10(stock_data[base_cols].values))
stock_data[log_cols] = df
stock_data[log_r_cols] = df - df.shift(1)

translate_cols = [f'translate_log_{w}' for w in base_cols]
stock_data[translate_cols] = pd.DataFrame(bn.move_mean(-np.log10(stock_data[base_cols].values), 100, axis=0))

builder = Builder(
    code_col='stock',
    time_col='date',
    base_df=stock_data,
    n_thread=4,
    dtype=np.float32,
    max_feature_window=105,
    max_label_window=60
)

builder['month'] = builder['date'].dt.month
builder['weekday'] = builder['date'].dt.weekday
builder['yestoday_w_close'] = builder['w_close'].shift(1)
builder['yestoday_w_avg'] = builder['w_avg'].shift(1)

builder.split_by_time(['2012-06-01', '2014-01-01'])

# %%

from QPhantom.core.metrics import Metrics
from QPhantom.core.quant.test import back_test


def gen_label(builder, min_period, triggers):
    builder.do_init()
    buy_at = 1.005
    extra_cost_rate = 0.0025
    builder.label("trigger", {
        "buy_at": buy_at,
        "buy_cond": (builder["w_low"] < builder["w_high"]) & (builder["low"] < builder["w_open"]), #没有涨停
        "extra_cost_rate": extra_cost_rate,
        "base_col": "w_open",
        "high": "w_high",
        "low": "w_low",
        "fallback": "w_close",
        "min_period": min_period,
        "least_time": 12,
        "trigger": triggers,
        "trade_cond": builder["amount"] > 1e7
    }, key="trigger")

    builder.eval()
    return [v["trigger"] for v in builder.get_label(do_init=True)]


def do_test(base_df, base_y, window_size=1, score_col=None, score_threshold=0.5, unit_max_k=5):
    with measureTime("test total"):
        trade_log = back_test(
            col_code=base_df["stock"],
            col_time=base_df["date"],
            col_price=base_df["w_close"],
            col_period=base_y["period"],
            col_buy_flag=base_y["buy_flag"],
            col_buy_price=base_y["buy_price"],
            col_sell_price=base_y["sell_price"],
            col_benchmark=base_df["iclose"],
            col_score=score_col,
            funds=500000,
            top_k=3,
            score_threshold=score_threshold,
            max_k=4,
            unit_max_k=unit_max_k,
            min_cost=5000,
            skip_rate=0.6
        )
    M = Metrics(size=(10, 10), fontsize=12)
    M.plot_trade_log(trade_log, log_scale=False, window_size=window_size)
    return trade_log

def limit(arr, start, end):
    return arr[(arr >= start) & (arr <= end)]

import matplotlib.pylab as plt
import seaborn as sns

def plot_dist(dfs, conds=[True, True, True], names=["train", "val", "test"]):
    plt.figure(figsize=(12, 12))
    x_range = (-1, 1)
    for df, cond, name in zip(dfs, conds, names):
        base_cond = np.ones(df.shape[0], dtype=np.bool)
        col = limit(df['rate'][base_cond & cond], *x_range)
        mean = df['rate'][base_cond & cond].mean()
        sns.distplot(col, bins=200, label=f"{name}@mean: {mean}")
    plt.xlim(x_range)
    plt.legend(loc="upper left", fontsize=15)
    plt.show()

def test_on_random(df_y):
    plot_dist(df_y, [True, True, True])
    df_y_train, df_y_val, df_y_test=df_y
    df_train, df_val, df_test=builder.get_df()

    a = do_test(base_df = df_train, base_y = df_y_train, unit_max_k=unit_max_k, score_threshold=2.0)
    b = do_test(base_df = df_val, base_y = df_y_val, unit_max_k=unit_max_k)
    c = do_test(base_df = df_test, base_y = df_y_test, unit_max_k=unit_max_k)
    return a, b, c

# %%

min_period = 22
unit_max_k = 2

# %%

df_y = gen_label(builder, min_period=min_period, triggers=[
    {
        "sell_at": 1.05,
        "flag": True,
        "sell_on": "w_open"
    }
])

# %%

# 测试随机选股

ta, tb, tc = test_on_random(df_y)

# %%

tc[1]["rate"] = tc[1]["close"] / tc[1]["open"]

# %%

tc[1]
