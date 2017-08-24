import bottleneck as bn
from QPhantom.core.quant import Builder, ColumnBase
import QPhantom.core.utils as F


@Builder.register_handler('move_window')
class MoveWindowFeatureGenerate(ColumnBase):
    def init(self):
        self.windows = self.param['window']
        self.cols = self.param['col']
        self.types = self.param['type']
        self.translation_cols = self.param.get('translation')
        self.scale_cols = self.param.get('scale')
        self.move_window_mapping = {
            "mean": lambda c, s, t, w: bn.move_mean(c, w) * s + t,
            "std": lambda c, s, t, w: bn.move_std(c, w) * s,
            "var": lambda c, s, t, w: bn.move_var(c, w) * s * s,
            "min": lambda c, s, t, w: bn.move_min(c, w) * s + t,
            "max": lambda c, s, t, w: bn.move_max(c, w) * s + t,
            "rank": lambda c, s, t, w: bn.move_rank(c, w),
            "sum": lambda c, s, t, w: bn.move_sum(c, w) * s + t * w,
            "ema": lambda c, s, t, w: F.ema(c, 2.0 / (w + 1), start_indices=self.base.start_indices) * s + t,
            "rsi": lambda c, s, t, w: F.rsi(c, w, start_indices=self.base.start_indices),
            "psy": lambda c, s, t, w: F.psy(c, w, start_indices=self.base.start_indices),
            "bias": lambda c, s, t, w: F.bias(c, w, start_indices=self.base.start_indices)
        }

    def feature_move_window(self, col, key, scale, translation, window):
        return self.move_window_mapping[key](col, scale, translation, window)

    def names(self):
        return [
            (c, t, w)
            for c in self.cols
            for t in self.types
            for w in self.windows
        ]

    def eval(self):
        b_cols = [self.base[c] for c in self.cols]
        t_cols = [self.base[c] for c in self.translation_cols] if self.translation_cols is not None else [0.0 for c in self.cols]
        s_cols = [self.base[c] for c in self.scale_cols] if self.scale_cols is not None else [1.0 for c in self.cols]
        cols = self.base.map(
            lambda x: self.feature_move_window(*x),
            [(c, t, scale, trans, w)
                for c, scale, trans in zip(b_cols, s_cols, t_cols)
                for t in self.types
                for w in self.windows]
        )
        return cols
