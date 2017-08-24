import h5py
import numpy as np
import numexpr as ne
import bottleneck as bn

class AutoScaler(object):
    def __init__(self, offset=1.0, threshold=20.0, do_void_scale=True):
        self.offset = offset
        self.threshold = threshold
        self.do_void_scale = do_void_scale
        self.fitted = False
        self.sum0 = 0.0
        self.sum1 = 0.0
        self.sum2 = 0.0

    def fit(self, X, axis=None):
        '''
        accumulated fit of scaler
        :param X:
        :param axis:
        :return:
        '''
        self.axis = axis if axis != None else tuple(range(len(X.shape) - 1))
        if isinstance(self.axis, int):
            self.axis = (self.axis,)
        aX = AutoScaler.void_scale(X, self.offset, self.threshold) if self.do_void_scale == True else X
        self.sum0 = (~np.isnan(aX)).sum(axis=self.axis) + self.sum0
        self.sum1 = (bn.nansum(aX, axis=self.axis[0]) if len(self.axis) == 1 and aX.dtype == np.float64 else np.nansum(aX, axis=self.axis)) + self.sum1
        aX = aX * aX
        self.sum2 = (bn.nansum(aX.astype(np.float64), axis=self.axis[0]) if len(self.axis) == 1 and aX.dtype == np.float64 else np.nansum(aX, axis=self.axis)) + self.sum2
        self.mean = self.sum1 / self.sum0
        self.std = np.sqrt(self.sum2 / self.sum0 - self.mean * self.mean)
        self.fitted = True
        return self

    def transform(self, X, use_threshold=True):
        aX = AutoScaler.void_scale(X, self.offset, self.threshold) if self.do_void_scale == True else X
        if self.fitted == False:
            return aX
        else:
            mean = self.mean
            std = self.std
            th = self.threshold
            if use_threshold == True:
                return ne.evaluate("tanh((aX - mean) / (std * th)) * th")
            else:
                return ne.evaluate("(aX - mean) / std")

    def save(self, path):
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("offset", data=self.offset)
            hf.create_dataset("threshold", data=self.threshold)
            hf.create_dataset("fitted", data=self.fitted)
            hf.create_dataset("do_void_scale", data=self.do_void_scale)
            hf.create_dataset('sum0', data=self.sum0)
            hf.create_dataset('sum1', data=self.sum1)
            hf.create_dataset('sum2', data=self.sum2)
            if self.fitted == True:
                hf.create_dataset("axis", data=self.axis)
                hf.create_dataset("mean", data=self.mean)
                hf.create_dataset("std", data=self.std)

    @staticmethod
    def load(path):
        with h5py.File(path, 'r') as hf:
            scaler = AutoScaler(
                offset=hf['offset'].value,
                threshold=hf['threshold'].value,
                do_void_scale=hf['do_void_scale'].value if 'do_void_scale' in hf else True
            )
            scaler.fitted = hf['fitted'].value
            if scaler.fitted == True:
                scaler.axis = tuple(hf['axis'].value)
                scaler.mean = hf['mean'].value
                scaler.std = hf['std'].value
            return scaler

    @staticmethod
    def void_scale(X, offset=1.0, threshold=20.0):
        ldata = ne.evaluate("log10(abs(X) + offset)")
        return ne.evaluate("tanh(where(X > 0, ldata, -ldata) / threshold) * threshold")
