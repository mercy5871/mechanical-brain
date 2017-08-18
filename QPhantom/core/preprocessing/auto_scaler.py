import numpy as np
import h5py

class AutoScaler(object):

    def __init__(self, offset=1.0, threshold=5.0):
        self.offset = offset
        self.threshold = threshold
        self.fitted = False

    def fit(self, X, axis=None):
        self.axis = axis if axis is not None else tuple(range(len(X.shape)))
        aX = AutoScaler.void_scale(X, self.offset, self.threshold)
        self.mean = np.nanmean(aX, axis=self.axis)
        self.std = np.nanstd(aX, axis=self.axis)
        self.fitted = True
        return self

    def transform(self, X):
        aX = AutoScaler.void_scale(X, self.offset, self.threshold)
        if self.fitted == False:
            return aX
        else:
            return (aX - self.mean) / (3 * self.std)


    def save(self, path):
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("offset", data=self.offset)
            hf.create_dataset("threshold", data=self.threshold)
            hf.create_dataset("fitted", data=self.fitted)
            if self.fitted == True:
                hf.create_dataset("axis", data=self.axis)
                hf.create_dataset("mean", data=self.mean)
                hf.create_dataset("std", data=self.std)


    @staticmethod
    def load(path):
        with h5py.File(path, 'r') as hf:
            scaler = AutoScaler(
                offset=hf['offset'].value,
                threshold=hf['threshold'].value
            )
            scaler.fitted = hf['fitted'].value
            if scaler.fitted == True:
                scaler.axis = tuple(hf['axis'].value)
                scaler.mean = hf['mean'].value
                scaler.std = hf['std'].value
            return scaler

    @staticmethod
    def void_scale(X, offset=1.0, threshold=5.0):
        ldata = np.log10(np.abs(X) + offset)
        return np.tanh(np.where(X > 0, ldata, -ldata) / threshold) * threshold