from QPhantom.core.quant import Builder, ColumnBase
from QPhantom.core.utils import fill_zeros_with_last, index2flag

import numpy as np
import bottleneck as bn

@Builder.register_handler("trigger")
class Trigger(ColumnBase):
    def init(self):
        self.buy_at = self.param["buy_at"]
        self.buy_cond = self.param.get("buy_cond", True)
        self.base_col = self.param["base_col"]
        self.high_col = self.param["high"]
        self.low_col = self.param["low"]
        self.fallback_col = self.param["fallback"]
        self.extra_cost_rate = self.param["extra_cost_rate"]
        self.least_time = self.param["least_time"]
        self.min_period = self.param["min_period"]
        self.triggers = self.param["trigger"]
        self.trade_cond = self.param.get("trade_cond", True)

    def names(self):
        ns = [
            "rate",
            "period",
            "unit_rate",
            "buy_flag",
            "sell_flag",
            "buy_price",
            "sell_price",
            "price_hold",
            "period_hold"
        ]
        return ns

    def eval_subtrigger(self, param):
        sell_col = param.get("sell_on", self.base_col)
        sell_at = param["sell_at"]
        trigger_flag = param["flag"]
        sell_flag = np.where(self.flag_mask, np.zeros(self.base.size, dtype=np.bool), trigger_flag)
        target_price = self.base[sell_col] * sell_at
        sold_flag = (self.base[self.high_col] > target_price) & self.trade_cond
        return sell_flag, sold_flag, target_price

    def eval(self):
        start_flag = self.base.start_flag
        end_flag = self.base.end_flag
        self.flag_mask = bn.move_sum(start_flag.astype(np.int32), self.least_time, min_count=1) >= 1
        end_mask = index2flag([i - j for i in self.base.end_indices for j in range(self.min_period)], self.base.size)

        trigger_res = [self.eval_subtrigger(p) for p in self.triggers][::-1]

        fallback_price = np.where(end_flag, self.base[self.fallback_col], 0.0)
        sold_price = fallback_price
        price_flag = end_flag
        for sell_flag, sold_flag, target_price in trigger_res:
            sold_price = np.where(
                sell_flag & sold_flag,
                target_price,
                np.where(
                    sell_flag,
                    fallback_price,
                    sold_price
                )
            )
            price_flag = (sell_flag & sold_flag) | ((~sell_flag) & price_flag) | end_flag

        # 获取需要持有的天数
        arr = price_flag[::-1]
        days_keep = np.arange(len(arr)) + 1
        days_keep = days_keep - fill_zeros_with_last(np.where(arr, days_keep, 0))
        days_keep = days_keep[::-1]

        # 计算卖出位置的索引
        sell_indices = np.arange(self.base.size)
        sell_indices = np.where(
            # 已经持有的至少min_period或者股票已经在结束区间内
            end_mask | (days_keep > self.min_period),
            #当前索引 + 持有天数
            sell_indices + days_keep,
            # min_period天后的持有天数
            sell_indices + self.min_period + days_keep[np.roll(sell_indices, -self.min_period)]
        )

        # 假设可以马上卖，按照策略能够最后卖出的价格
        prefer_sold_price = sold_price[np.arange(self.base.size) + days_keep]

        final_price_hold = np.roll(prefer_sold_price, -1) * (1 - self.extra_cost_rate)
        final_price_hold[-1] = prefer_sold_price[-1] * (1 - self.extra_cost_rate)
        period_hold = days_keep + 2.0
        period_hold[-1] = 1.0

        # 最后的卖出价格
        limit_sold_price = sold_price[sell_indices]
        # 假设持有, 到卖出的持有天数
        limit_days_keep = sell_indices - np.arange(self.base.size)

        # 实际的收益率，考虑买入成功与否，买入不成功，实际收益为0
        buy_price = self.base[self.base_col] * self.buy_at
        buy_flag = (self.base[self.low_col] < buy_price) & self.buy_cond & self.trade_cond
        buy_price_real = buy_price * (1 + self.extra_cost_rate)
        sold_price_real = limit_sold_price * (1 -  self.extra_cost_rate)
        actual_rate = np.where(
            start_flag,
            1.0,
            np.where(
                # 以buy_at价格买入成功
                buy_flag,
                sold_price_real / buy_price_real,
                1.0
            )
        ) - 1.0
        prefer_rate = prefer_sold_price * (1 - self.extra_cost_rate) / self.base[self.base_col] - 1.0
        # 实际占有资金的天数，如果买入失败，资金相当于以1的比例持有1天
        actual_days_keep = np.where(buy_flag, limit_days_keep + 1, 1)

        # 资金的平均日收益率
        unit_rate = np.power(1.0 + actual_rate, 1.0 / actual_days_keep) - 1.0

        return [
            actual_rate,
            actual_days_keep,
            unit_rate,
            buy_flag,
            price_flag,
            buy_price_real,
            sold_price_real,
            # prefer_sold_price,
            # prefer_rate,
            # days_keep + 1.0, #当天占用资金，相当于需要消耗一天的资金
            final_price_hold,
            period_hold
        ]
