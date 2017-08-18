from sklearn.preprocessing import OneHotEncoder
from QPhantom.core.quant import Builder, ColumnBase

@Builder.register_handler('rank')
class RankFeatureGenerate(ColumnBase):
    def init(self):
        self.cols = self.param['col']
        # inverse order
        self.inverse = self.param.get('inverse', False)

    def names(self):
        return self.cols

    def eval(self):
        tcol = self.base.time_col
        df = self.base[[tcol] + self.cols].groupby(tcol).rank(pct=True, ascending=not self.inverse)
        return [df[c] for c in self.cols]
