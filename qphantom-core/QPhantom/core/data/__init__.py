import numpy as np
from QPhantom.core.data.dataset import DataSet
from QPhantom.core.data.data_queue import DataQueue
from QPhantom.core.data.hdf_dataset import HDFDataSet
from QPhantom.core.data.bagging import ResultBagging

def batch(data, batch_size=128, padding=False):
    keys = list(data.keys())
    assert len(keys) > 0
    size = data[keys[0]].shape[0]
    assert all([data[k].shape[0] == size for k in keys])
    for i in range(0, size, batch_size):
        ans = {k: v[i:i + batch_size] for k, v in data.items()}
        if i + batch_size > size and padding == True:
            ans = { k:np.pad(c, [(0, batch_size - c.shape[0])] + [(0, 0) for i in c.shape[1:]], mode="edge") for k, c in ans.items() }
        yield ans


__all__ = [DataSet, DataQueue, HDFDataSet, batch, ResultBagging]
