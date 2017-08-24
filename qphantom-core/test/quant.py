# %%

import unittest
import pandas as pd
import numpy as np
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
stock_data['year'] = stock_data['date'].dt.year
stock_data['weekday'] = stock_data['date'].dt.weekday
stock_data['month'] = stock_data['date'].dt.month
stock_data['yestoday_w_close'] = stock_data['w_close'].shift(1)
stock_data['rate_close'] = stock_data['w_close'] / stock_data['yestoday_w_close']

builder = Builder(
    code_col='stock',
    time_col='date',
    base_df=stock_data,
    max_label_window=2
    # target_date='2017-05-12'
)

base_cols = ['w_open', 'w_high', 'w_low', 'w_close', 'w_volume', 'amount']
raw_cols = builder.feature("raw", {
    "col": ['month', 'weekday']
}, key="raw")
one_hots = builder.feature("onehot", {
    "col": ['weekday']
}, key="onehot")
print("raw: ", raw_cols)
rate_on_close_cols = builder.feature("change_rate", {
    "col": ["w_open", "w_high", "w_low", "w_close"],
    "base_col": "w_close",
    "period": [-1, -7]
}, key="rate_on_close")
print("rate_on_close: ", rate_on_close_cols)
rate_cols = builder.feature("change_rate", {
    "col": ["w_open", "w_high", "w_low", "w_close", "w_avg", "amount", "w_volume"],
    "period": [-1, -7]
}, key="rate")
print("rate: ", rate_cols)
window_cols = builder.feature("window", {
    'window': 5,
    'col': base_cols
}, key="window")
print("window: ", window_cols)
move_window_cols = builder.feature("move_window", {
    "window": [64, 32, 16, 8, 4, 3, 2],
    "col": base_cols,
    "type": ['mean', 'std', 'min', 'max', 'rank', 'ema']
}, key="move_window")
print("move_window: ", move_window_cols)

move_window_alpha = builder.feature("alpha", {
    "open": "w_open",
    "high": "w_high",
    "low": "w_low",
    "close": "w_close",
    "window": 30,
    "type": ['alpha53','kdj', 'wr', 'cci', 'arbr']
}, key="alpha")
print("alhpa: ", move_window_alpha)

rank_feature_names = builder.feature("rank", {
    "col": ["rate_close"],
    "inverse": False
}, key="rank")
print("rank: ", rank_feature_names)

label_fix_window = builder.label("fix_window", {
    "buy_col": "w_close",
    "buy_at": -1,
    "sell_col": "w_open",
    "sell_at": 1
}, key="fix_window")
builder.label("rolling_fix_price", {
    "buy_at": 1.001,
    "sell_at": 1.098,
    "extra_cost_rate": 0.0025,
    "base_col": "yestoday_w_close",
    "high": "w_high",
    "low": "w_low",
    "fallback": "w_close"
}, key="rolling")
builder.label("trigger", {
    "buy_at": 1.001,
    "extra_cost_rate": 0.0025,
    "base_col": "yestoday_w_close",
    "high": "w_high",
    "low": "w_low",
    "fallback": "w_close",
    "min_period": 1,
    "least_time": 1,
    "trigger": [
        {
            "sell_at": 1.098,
            "flag": True, #bn.move_sum(((builder["w_close"] > buidler["yestoday_w_close"] * 1.05) & ).astype(np.int32), 4, min_count=1) == 2
            "sell_on": "yestoday_w_close"
        },
        {
            "sell_at": 0.99,
            "flag": True,
            "sell_on": "w_avg"
        }
    ]
}, key="trigger")
# builder.split_by_time('2013-01-01')
# builder.eval()

# X = builder.get_feature(using_df=True, flatten=False)
# y = builder.get_label()
X, y = builder.get_Xy(feature_using_df=True)
c_time = builder.get_time()
c_code = builder.get_code()


from QPhantom.core.quant.test import back_test

base_df = builder.get_df()

with measureTime("test total"):
    trade_log = back_test(
        funds=50000,
        col_code=builder.get_code(),
        col_time=builder.get_time(),
        col_score=None,
        col_price=base_df["w_close"],
        col_period=y["trigger"]["period"],
        col_buy_flag=y["trigger"]["buy_flag"],
        col_buy_price=y["trigger"]["buy_price"],
        col_sell_price=y["trigger"]["sell_price"],
        col_benchmark=base_df["iclose"],
        top_k=50,
        score_threshold=0.5,
        max_k=30,
        unit_max_k=3,
        min_cost=5000,
        skip_rate=0.5
    )


# M.plot_trade_summary(trade_log[0]["index"], trade_log[0]["position"] + trade_log[0]["funds"], trade_log[0]["benchmark"])


# class ImgPlotter(object):
#     def image(self, img):
#         with open('/Users/earthson/Desktop/test.png', 'wb') as f:
#             f.write(img)

# M = Metrics(size=(8, 8), fontsize=12, plotter=ImgPlotter())
# M.plot_trade_log(trade_log[-50:], window_size=1)

# %%

class QuantTest(unittest.TestCase):
    def test_move_window(self):
        def test_i(i):
            t_s = c_time.iloc[i - 16]
            t_e = c_time.iloc[i - 1]
            c = c_code.iloc[i]
            feature_v = X["move_window"]["w_low"]["mean"][16].iloc[i]
            df = stock_data
            df = df[(df['stock'] == c) & (df['date'] >= t_s) & (df['date'] <= t_e)]
            actual_v = df["w_low"].mean()
            assert equal(actual_v, feature_v), "move_window not match"

        for s, i in zip(builder.start_indices, [100, 109]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [16, 17]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [30, 47]):
            test_i(s + i)

    def test_change_rate(self):
        def test_i(i):
            t_s = c_time.iloc[i - 2]
            t_e = c_time.iloc[i - 1]
            c = c_code.iloc[i]
            feature_v = X["rate"]["w_close"][-1].iloc[i]
            df = stock_data
            base_close = df[(df['stock'] == c) & (df['date'] == t_s)]["w_close"].iloc[0]
            target_close = df[(df['stock'] == c)  & (df['date'] == t_e)]["w_close"].iloc[0]
            actual_v = target_close / base_close - 1.0
            assert equal(actual_v, feature_v), "change_rate not match"

        for s, i in zip(builder.start_indices, [100, 109]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [16, 17]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [30, 47]):
            test_i(s + i)

    def test_window(self):
        def test_i(i):
            t_s = c_time.iloc[i - 5]
            t_e = c_time.iloc[i - 1]
            c = c_code.iloc[i]
            feature_v_s = X["window"]["w_close"][0].iloc[i]
            feature_v_e = X["window"]["w_close"][4].iloc[i]
            df = stock_data
            actual_v_s = df[(df['stock'] == c) & (df['date'] == t_s)]['w_close'].iloc[0]
            actual_v_e = df[(df['stock'] == c) & (df['date'] == t_e)]['w_close'].iloc[0]
            assert equal(actual_v_s, feature_v_s), "window not match"
            assert equal(actual_v_e, feature_v_e), "window not match"

        for s, i in zip(builder.start_indices, [100, 109]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [10, 9]):
            test_i(s + i)
        for s, i in zip(builder.start_indices, [30, 47]):
            test_i(s + i)

    def test_raw(self):
        def test_i(i):
            t = c_time.iloc[i - 1]
            c = c_code.iloc[i]
            feature_v = X["raw"]["weekday"].iloc[i]
            df = stock_data
            actual_v = df[(df['stock'] == c) & (df['date'] == t)]['weekday'].iloc[0]
            assert actual_v == feature_v, "raw feature not match"

        for i in [10, 20, 1000, 10000]:
            test_i(i)

    def test_label_rolling(self):
        basic_df = builder.get_df()[["stock", "date", "open", "high", "low", "close", "factor"]].reset_index(drop=True)
        basic_df[["open", "high", "low", "close"]] = np.array(basic_df[["open", "high", "low", "close"]]) * basic_df["factor"][:, None]
        label_df = y["rolling"].reset_index(drop=True)
        basic_df[label_df.columns] = label_df
        basic_df

    def test_factor(self):
        assert (stock_data['w_open'] == stock_data['open'] * stock_data['factor']).sum() == stock_data.shape[0]

    def test_trigger(self):
        assert (y["trigger"]["rate"] != y["rolling"]["rate"]).sum() == 0, "trigger_rolling and rolling should be same"

    def test_fix_window(self):
        col_fix_window = np.array(y["fix_window"])
        dd = builder.get_df()
        dd["label"] = col_fix_window
        cc = dd["w_close"].iloc[3]
        oo = dd["w_open"].iloc[5]
        assert dd["label"].iloc[4] == oo / cc - 1.0, "fix_window should calc correct"

    def test_clean(self):
        assert len(builder.col_space) == 0, "builder col space should be clear"


if __name__ == "__main__":
    unittest.main()
