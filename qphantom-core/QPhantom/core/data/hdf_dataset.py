import h5py
import numpy as np
import bottleneck as bn

class HDFDataSet(object):
    def __init__(self, path, debug=False, chunk_size=512, compression=None):
        self.path = path
        self.chunk_size = chunk_size
        self.debug = debug
        self.compression = compression

    def add(self, items):
        return self.extend(items)

    def __open(self, mode="r"):
        return h5py.File(self.path, mode=mode, libver="latest")

    def extend(self, items):
        if hasattr(items, 'items'):
            items = items.items()
        with self.__open("a") as f:
            for k, v in items:
                if k in f:
                    origin_shape = f[k].shape
                    if origin_shape[1:] != v.shape[1:]:
                        raise Exception(f"Invalid data shape for {k}: {v.shape}. expect: {(None, ) + origin_shape[1:]}")
                    st_idx = origin_shape[0]
                    d_size = v.shape[0]
                    f[k].resize(st_idx + d_size, axis=0)
                    f[k][st_idx:st_idx + d_size] = v
                    if self.debug == True:
                        print(f'{k} exists and try to append it. from shape {origin_shape} => {(st_idx+d_size, ) + origin_shape[1:]}')
                else:
                    self.__create_kv(f, k, v)
                    if self.debug == True:
                        print(f"create dataset {k} with shape {v.shape}")


    def __create_kv(self, f, key, value):
        return f.create_dataset(
            key,
            data=value,
            maxshape=(None, ) + value.shape[1:],
            chunks=(self.chunk_size,) + value.shape[1:],
            compression=self.compression
        )

    def __getitem__(self, key):
        with self.__open() as f:
            return f[key].value

    def __setitem__(self, key, value):
        with self.__open("a") as f:
            if key in f:
                del f[key]
            self.__create_kv(f, key, value)

    def keys(self):
        with self.__open() as f:
            return list(f.keys())

    def items(self):
        with self.__open() as f:
            return list(f.items())

    def dtypes(self, keys):
        with self.__open() as f:
            return [f[k].dtype for k in keys]

    def shapes(self, keys):
        with self.__open() as f:
            return [f[k].shape for k in keys]

    def row_shapes(self, keys):
        return [s[1:] for s in self.shapes(keys)]

    def get(self, keys=None, mask=None, transform=None):
        '''
        :param keys: None for all keys
        :param mask: boolean mask
        :param transform: f({key: value}): {key: value}
        :return: {key: value}
        '''
        if keys is None:
            keys = list(self.keys())
        if transform is None:
            transform = lambda x: x
        if mask is None:
            with self.__open() as f:
                return transform({key: f.get(key).value for key in keys})
        else:
            ans = None
            n_row = mask.sum()
            cur = 0
            for batch in self.batch(keys=keys, batch_size=self.chunk_size, mask=mask, callback=transform):
                if ans is None:
                    ans = {k: np.zeros((n_row, ) + batch[k].shape[1:], dtype=batch[k].dtype) for k in keys}
                for k in keys:
                    ans[k][cur:cur+batch[k].shape[0]] = batch[k]
                cur += batch[keys[0]].shape[0]
            return ans


    def __num_chunk(self, f, key):
        return (f[key].shape[0] - 1) // self.chunk_size + 1

    @property
    def num_chunk(self):
        with self.__open() as f:
            key = list(f.keys())[0]
            return self.__num_chunk(f, key)

    def __read_chunk(self, f, keys, i=None, expect_size=None, num_chunk=None, mask=None, chunk_indices=None):
        if num_chunk is None:
            num_chunk = self.__num_chunk(f, keys[0])
        if i is None:
            i = np.random.randint(num_chunk)
        if chunk_indices is None:
            chunk_indices = np.arange(num_chunk)
        vs = [f[key][i*self.chunk_size:(i+1)*self.chunk_size] for key in keys]
        if mask is not None:
            vs = [v[mask[i*self.chunk_size:(i+1)*self.chunk_size]] for v in vs]
        expect_size = self.chunk_size if expect_size is None else expect_size
        while vs[0].shape[0] < expect_size:
            extras = self.__read_chunk(
                f,
                keys,
                chunk_indices[np.random.randint(chunk_indices.shape[0])],
                expect_size=expect_size - vs[0].shape[0],
                mask=mask,
                chunk_indices=chunk_indices
            )
            vs = [np.concatenate([v, extra], axis=0) for v, extra in zip(vs, extras) ]
        return [v[:expect_size] for v in vs]

    def __chunk_mask(self, mask):
        indices = np.arange((mask.shape[0] - 1) // self.chunk_size + 1, dtype=np.int32) * self.chunk_size
        return bn.move_sum(mask.astype(np.int32)[::-1], self.chunk_size, min_count=1)[::-1][indices] > 0

    def shuffle_batch(self, batch_size, keys=None, step=None, pool_chunks=16, n_batch=None, mode="dict", callback=None, mask=None):
        if keys is None:
            keys = list(self.keys())
        if step is None:
            step = (2*self.chunk_size) // batch_size
        data_pool = dict()
        pool_size = pool_chunks * self.chunk_size
        with self.__open() as f:
            num_chunk = self.__num_chunk(f, keys[0])
            chunk_mask = np.ones(num_chunk, dtype=np.bool) if mask is None else self.__chunk_mask(mask)
            chunk_indices = np.arange(num_chunk)[chunk_mask].copy()
            if self.debug == True:
                print("CHUNK MASK:", chunk_mask)
                print("CHUNK INDICES: ", chunk_indices)
            n_chunk = chunk_indices.shape[0]
            np.random.shuffle(chunk_indices)
            init_data = [self.__read_chunk(f, keys, i=i, mask=mask, chunk_indices=chunk_indices)
                         for i in (list(chunk_indices[:pool_chunks]) + \
                         [chunk_indices[np.random.randint(n_chunk)] for j in range(n_chunk, pool_chunks)])
                         ]
            for i, k in enumerate(keys):
                data_pool[k] = np.concatenate([init_data[j][i] for j in range(pool_chunks)], axis=0)
            del init_data
            swap_chunk_idx = 0
            i_batch = 0
            start_idx = 0
            np.random.shuffle(chunk_indices)
            chunk_idx = 0
            indices = np.arange(pool_size)
            np.random.shuffle(indices)
            np.random.shuffle(indices)
            while n_batch is None or n_batch > 0:
                i_batch += 1
                if i_batch % step == 0:
                    new_data = self.__read_chunk(f, keys, i=chunk_indices[chunk_idx], mask=mask, chunk_indices=chunk_indices)
                    chunk_idx += 1
                    if chunk_idx >= n_chunk:
                        chunk_idx = 0
                        np.random.shuffle(chunk_indices)
                    for k, c in zip(keys, new_data):
                        data_pool[k][swap_chunk_idx*self.chunk_size:(swap_chunk_idx+1)*self.chunk_size] = c
                    swap_chunk_idx = (swap_chunk_idx + 1) % pool_chunks
                    np.random.shuffle(indices)
                ans = [data_pool[k][indices[start_idx:start_idx+batch_size]] for k in keys]
                ans = ({k:v for k, v in zip(keys, ans)} if mode == "dict" else ans)
                yield ans if callback is None else callback(ans)
                start_idx += batch_size
                if start_idx + batch_size > pool_size:
                    start_idx = 0
                if n_batch is not None:
                    n_batch -= 1

    def batch(self, batch_size, keys=None, mode="dict", padding=False, callback=None, mask=None):
        if keys is None:
            keys = list(self.keys())
        with self.__open() as f:
            num_chunk = self.__num_chunk(f, keys[0])
            chunk_mask = np.ones(num_chunk, dtype=np.bool) if mask is None else self.__chunk_mask(mask)
            if self.debug == True:
                print("NUM_CHUNK:", num_chunk)
            ch = None
            for m_c, i_c in zip(chunk_mask, range(num_chunk)):
                if m_c == False:
                    continue
                new_chunk = [f[k][i_c*self.chunk_size:(i_c+1)*self.chunk_size] for k in keys]
                if mask is not None:
                    new_chunk = [v[mask[i_c*self.chunk_size:(i_c+1)*self.chunk_size]] for v in new_chunk]
                if ch is None:
                    ch = new_chunk
                else:
                    ch = [np.concatenate([o_c, n_c], axis=0) for o_c, n_c in zip(ch, new_chunk)]
                current_size = ch[0].shape[0]
                i = 0
                while i + batch_size <= current_size:
                    ans = [c[i:i+batch_size] for c in ch]
                    ans = ({k:v for k, v in zip(keys, ans)} if mode == "dict" else ans)
                    yield ans if callback is None else callback(ans)
                    i += batch_size
                ch = [c[i:] for c in ch] if i < current_size else None
            if ch is not None:
                if self.debug == True:
                    print("LAST BATCH")
                if padding == True:
                    ch = [np.pad(c, [(0, batch_size - c.shape[0])] + [(0, 0) for i in c.shape[1:]], mode="edge") for c in ch]
                ans = ({k: v for k, v in zip(keys, ch)} if mode == "dict" else ch)
                yield ans if callback is None else callback(ans)
