import time
import numpy as np
from QPhantom.core.utils import measureTime

class DataSet(object):
    def __init__(self, data, keys=None, debug=False):
        """
        Args:
            data: list of ndarray or ndarray
        """
        self.data = data if isinstance(data, list) else [data]
        self.n_rows = self.data[0].shape[0]
        self.keys = keys
        assert all([d.shape[0] == self.n_rows for d in self.data]), "all ndarray should have same rows"
        self.debug = debug
        self.batch_time_stack = []
        self.dtypes = [d.dtype for d in self.data]
        self.shapes = [d.shape[1:] for d in self.data]

    def stat(self, clear=True):
        print(f"BATCH PREPROCESS AVG: {round(np.array(self.batch_time_stack).mean() * 1000, 1)}ms")
        print(f"BATCH PREPROCESS SUM: {round(np.array(self.batch_time_stack).sum(), 2)}s")
        self.batch_time_stack = list()

    def shuffle_batch(self, batch_size, n_batch=None, mask=None, rebalance_on=None):
        '''
        返回一个函数，它能够生成一个generator，用于返回需要的batch

        Args:
            n_batch: 能够返回的batch的数量，如果是None，表示无穷
            mask: 只返回mask之后的数据, 比如，可以把过于靠前或者靠后的样本排除，也可以追加条件，只对满足条件的样本训练
            rebalance_on: 是否基于某一列重采样，输入为数据在data中的索引号
        '''
        data = self.data if mask is None else [d[mask] for d in self.data]
        n_rows = data[0].shape[0]
        if rebalance_on is not None:
            rebalance_col = data[rebalance_on]
            uniqw, inverse, uniq_cnts = np.unique(rebalance_col, return_inverse=True, return_counts=True)
            indices = np.argsort(inverse)
            ends = np.add.accumulate(uniq_cnts)
            starts = np.roll(ends, 1)
            starts[0] = 0
            n_label = uniqw.shape[0]
            if n_label > 1000:
                raise Exception("unique label > 1000")
            if n_label > 1:
                def rebalance_batcher(batch_size, n_batch=None):
                    n_each_label = np.zeros(n_label, dtype=np.int32)
                    n_each_label[:] = batch_size // n_label
                    n_extra = batch_size - n_each_label.sum()
                    replace = batch_size > uniq_cnts.min()
                    i = 0
                    while n_batch is None or i < n_batch:
                        t_st = time.time()
                        random_each = np.zeros(n_label, dtype=np.int32)
                        random_extra = np.random.randint(0, high=n_label, size=n_extra)
                        uniq_v, idx_v, extra_cnts = np.unique(random_extra, return_index=True, return_counts=True)
                        random_each[random_extra[idx_v]] = extra_cnts
                        res_idx = np.concatenate([
                            indices[starts[idx]:ends[idx]][
                                np.random.randint(0, high=uniq_cnts[idx], size=n_each_label[idx] + random_each[idx])
                            ]
                            for idx in range(n_label)
                        ])
                        np.random.shuffle(res_idx)
                        values = [d[res_idx] for d in data]
                        ans = values if self.keys is None else {k:v for k, v in zip(self.keys, values)}
                        t_ed = time.time()
                        if self.debug == True:
                            self.batch_time_stack.append(t_ed - t_st)
                        yield ans
                        i += 1
                return rebalance_batcher(batch_size, n_batch)
        indices = np.random.permutation(n_rows)
        def batcher(batch_size, n_batch=None):
            i, j = 0, 0
            while n_batch is None or i < n_batch:
                t_st = time.time()
                if (j + 1) * batch_size > n_rows:
                    j = 0
                    np.random.shuffle(indices)
                cur = j * batch_size
                values = [d[indices[cur:cur + batch_size]] for d in data]
                ans = values if self.keys is None else {k:v for k, v in zip(self.keys, values)}
                t_ed = time.time()
                if self.debug == True:
                    self.batch_time_stack.append(t_ed - t_st)
                yield ans
                i += 1
                j += 1
        return batcher(batch_size, n_batch)

    def batch(self, batch_size, mask=None):
        """
        the last batch may have lengh < batch_size
        """
        if mask is None:
            mask = np.ones(self.data[0].shape[0], dtype=np.bool)
        masked_data = [d[mask] for d in self.data]
        for i in range(0, self.n_rows, batch_size):
            values = [d[i:i+batch_size] for d in masked_data]
            yield values if self.keys is None else {k:v for k, v in zip(self.keys, values)}

if __name__ == "__main__":
    from collections import Counter
    ds = DataSet([np.arange(5000000), np.arange(5000000) < 10])
    v = Counter(v for d in ds.shuffle_batch(batch_size=137, n_batch=20000, rebalance_on=1) for v in d[1])
    assert(max(v[True], v[False]) - min(v[True], v[False]) < 500)
