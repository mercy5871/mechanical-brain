import numpy as np
from QPhantom.core.quant import Builder, ColumnBase
from QPhantom.core.utils import pct_change

@Builder.register_handler("fix_window")
class FixWindow(ColumnBase):
    """
    以某个价格买入，固定持有一定时间，强制卖出
    """

    def init(self):
        """
        buy_col: 买入价格列
        buy_at: 以哪天的价格买入，-1表示前一天的价格
        sell_col: 卖出价格列
        sell_at: 以哪天的价格卖出，1表示1天后
        """
        self.buy_col = self.param["buy_col"]
        self.buy_at = self.param["buy_at"]
        self.sell_col = self.param["sell_col"]
        self.sell_at = self.param["sell_at"]

    def names(self):
        return ["fix_window@{buy_col}:{buy_at}_sell@{sell_col}:{sell_at}".format(
            buy_col=self.buy_col,
            buy_at=self.buy_at,
            sell_col=self.sell_col,
            sell_at=self.sell_at
        )]

    def eval(self):
        window_change_rate = pct_change(
            self.base[self.sell_col],
            base_col=np.roll(self.base[self.buy_col], -self.buy_at),
            period=self.sell_at
        )
        return [window_change_rate]
