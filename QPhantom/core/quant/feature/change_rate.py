from QPhantom.core.quant import Builder, ColumnBase


@Builder.register_handler('change_rate')
class ChangeRateFeatureGenerate(ColumnBase):

    def init(self):
        self.cols = self.param['col']
        self.periods = self.param['period']
        self.base_col = self.param.get('base_col')

    def names(self):
        """
        'rate_{col}_{symbol}{periods}'

            symbol: 'p'表示到目前为止变化了多少, 'n'表示未来会变化多少
        """
        fmt = 'rate_{col}_{symbol}{period}{on_base_col}'
        return [
            (c, p)
            for c in self.cols
            for p in self.periods
            ]

    def eval(self):
        cols = self.base.map(
            lambda x: self.base.feature_change_rate(
                self.base[x[0]],
                base_col=self.base[self.base_col] if self.base_col is not None else None,
                period=x[1]
            ),
            [(c, p)
            for c in self.cols
            for p in self.periods
                ]
        )
        return cols
