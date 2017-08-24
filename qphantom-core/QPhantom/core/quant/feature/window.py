from QPhantom.core.quant import Builder, ColumnBase


@Builder.register_handler('window')
class WindowFeatureGenerate(ColumnBase):
    def init(self):
        self.window = self.param['window']
        self.cols = self.param['col']

    def names(self):
        return self.cols

    def eval(self):
        cols = self.base.map(
            lambda c: self.base.feature_window_extract(self.base[c], self.window),
            self.cols
        )
        return cols
