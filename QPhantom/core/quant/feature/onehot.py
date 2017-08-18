from sklearn.preprocessing import OneHotEncoder
from QPhantom.core.quant import Builder, ColumnBase
import numpy as np

@Builder.register_handler('onehot')
class WindowFeatureGenerate(ColumnBase):
    def init(self):
        self.cols = self.param['col']
        self.on_label = self.param.get('on_label', False)

    def names(self):
        return self.cols

    def eval(self):
        def trans_col(c):
            col = np.roll(self.base[c], -1) if self.on_label else self.base[c]
            return OneHotEncoder(sparse=False).fit_transform(col[:, None])
        cols = self.base.map(
            trans_col,
            self.cols
        )
        return cols
