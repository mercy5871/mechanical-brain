from QPhantom.core.quant import Builder, ColumnBase
from QPhantom.core.utils import fill_zeros_with_last

import numpy as np

@Builder.register_handler("rolling_fix_price")
class RollingFixPrice(ColumnBase):
    def init(self):
        self.buy_at = self.param["buy_at"]
        self.sell_at = self.param["sell_at"]
        self.base_col = self.param["base_col"]
        self.high_col = self.param["high"]
        self.low_col = self.param["low"]
        self.fallback_col = self.param["fallback"]
        self.extra_cost_rate = self.param["extra_cost_rate"]

    def names(self):
        ns = ["rate", "time_keep", "unit_rate", "buy_flag", "sell_flag", "prefer_sell_price", "prefer_time_keep"]
        return ns

    def eval(self):
        base_col = self.base[self.base_col]
        sell_flag = base_col * self.sell_at < self.base[self.high_col]
        buy_flag = base_col * self.buy_at > self.base[self.low_col]
        end_indices = self.base.end_indices
        start_indices = self.base.start_indices
        end_flag = np.zeros(end_indices[-1] + 1, dtype=np.bool)
        end_flag[end_indices] = True
        check_flag = sell_flag | end_flag
        value_col = np.where(sell_flag, base_col * self.sell_at, 0.0)
        value_col = np.where(end_flag, self.base[self.fallback_col], value_col)
        value_col = fill_zeros_with_last(value_col[::-1])[::-1]

        arr = np.array(check_flag)[::-1]
        days_keep = np.arange(len(arr)) + 1
        days_keep = days_keep - fill_zeros_with_last(np.where(arr, days_keep, 0))
        days_keep = days_keep[::-1]

        value_col = np.roll(value_col, -1)
        return_rate = value_col * (1 -  self.extra_cost_rate) / (base_col * self.buy_at * (1.0 + self.extra_cost_rate)) - 1.0
        actual_rate = np.where(
            buy_flag,
            return_rate,
            0.0
        )
        actual_days_keep = np.where(buy_flag, days_keep + 1, 1)
        unit_expect_rate = np.power(1.0 + actual_rate, 1.0 / actual_days_keep) - 1.0
        return [actual_rate, actual_days_keep, unit_expect_rate, buy_flag, sell_flag | end_flag, value_col, days_keep]
