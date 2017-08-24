import unittest
import numpy as np

class Test(unittest.TestCase):
    def test_minibatch(self):
        from QPhantom.core.data import DataSet
        X = np.arange(30, dtype=np.float32).reshape([10, 3])
        y = np.arange(100, 110)
        dataset = DataSet(data=[X, y])
        shuffle_batches = list(dataset.shuffle_batch(batch_size=3, n_batch=10))
        assert len(shuffle_batches) == 10
        assert all([d.shape[0] == 3 for ds in shuffle_batches for d in ds])
        named_batcher = DataSet(data=[X, y], keys=("feature", "label"))
        shuffle_named_batches = list(named_batcher.shuffle_batch(batch_size=3, n_batch=10))
        for v in shuffle_named_batches:
            assert v["feature"].shape[0] == 3
            assert v["label"].shape[0] == 3

    def test_autoscaler(self):
        from QPhantom.core.preprocessing import AutoScaler
        for X in [np.arange(100).reshape((10, 5, 2)), np.arange(100).reshape((20, 5))]:
            ss = AutoScaler()
            ss.fit(X)
            res = ss.transform(X)
            assert res.shape == X.shape

    def test_hdf_dataset(self):
        from QPhantom.core.data import HDFDataSet
        import os
        print(os.getcwd())
        fpath = 'data/hdf_test.h5'
        if os.path.isfile(fpath):
            os.remove(fpath)
        ds = HDFDataSet(path=fpath, debug=True, compression="gzip")
        for i in range(10):
            data_i = {
                'a': np.arange(10000 * i, 10000 * (i+1)),
                'b': np.arange(10000 * 2 * 5 * i, 10000 * 2 * 5 * (i+1)).reshape((10000, 2, 5)),
                'c': np.arange(10000 * i, 10000 * (i+1))
            }
            ds.extend(data_i)
        print(ds['a'].shape)
        print(ds['b'].shape)
        for i, it in enumerate(ds.batch(batch_size=111)):
            assert it['a'][0] == i * 111
            assert it['b'][0][0][0] == 111 * i * 2 * 5
        for i, it in enumerate(ds.batch(batch_size=128)):
            assert it['a'][0] == i * 128
            assert it['b'][0][0][0] == 128 * i * 2 * 5
        for i, it in enumerate(ds.batch(batch_size=111, padding=True)):
            assert it['a'].shape[0] == 111
        for i, it in enumerate(ds.shuffle_batch(batch_size=16, step=2, n_batch=1000)):
            assert it['a'].shape[0] == 16
            assert (it['a'] != it['c']).sum() < 1
        for i, it in enumerate(ds.batch(batch_size=111, padding=True, mask=ds['a'] < 1000)):
            assert (it['a'] >= 1000).sum() < 1
            assert (it['a'] != it['c']).sum() < 1
        for i, it in enumerate(ds.shuffle_batch(batch_size=128, step=16, n_batch=100, mask=ds['a'] < 5000)):
            assert (it['a'] >= 5000).sum() < 1
        ans = np.concatenate([t[-1] for t in ds.shuffle_batch(batch_size=128, keys=['a'], step=8, n_batch=5000, mode="list")], axis=0)
        unique, counts = np.unique(ans, return_counts=True)
        assert unique.shape[0] / ds['a'].shape[0] > 0.9
        assert counts.std() < 6
        assert sum([v['b'].shape[0] for v in ds.batch(batch_size=128)]) == ds['a'].shape[0]
        def trans(d):
            d['a'] = d['a'] + 1
            return d
        res = ds.get(mask=(ds['a'] > 500) & (ds['a'] < 1000), transform=trans)
        assert (res['a'] <= 501).sum() == 0
        assert (res['a'] >= 1001).sum() == 0
        print(res)
        os.remove(fpath)

    def test_data_queue(self):
        from QPhantom.core.data import DataQueue
        assert len(list(DataQueue.background(range(10000)))) == 10000
        with DataQueue(lambda: range(20)) as q:
            assert len(list(q.buffer(n=10))) == 10

    def test_batch(self):
        from QPhantom.core.data import batch
        data = {'a': np.arange(100), 'b': np.arange(100, 200)}
        batched = list(batch(data, batch_size=11))
        assert len(batched) == 100 // 11 + 1
        assert batched[-1]['a'].shape[0] == 100 % 11
        batched = list(batch(data, batch_size=11, padding=True))
        assert len(batched) == 100 // 11 + 1
        assert batched[-1]['a'].shape[0] == 11

    def test_result_bagging(self):
        from QPhantom.core.data import ResultBagging
        bg = ResultBagging(k=7)
        x = None
        for i in range(10):
            x = bg.bag(np.ones(1) * i)
            # print(bg.stack)

        assert bg.result()[0] == np.arange(10)[-7:].mean()

if __name__ == "__main__":
    unittest.main()
