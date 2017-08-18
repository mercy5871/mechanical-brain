from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import bottleneck as bn
import numpy as np
import pandas as pd
from numba import jit

if __name__ == '__main__':
    import QPhantom.core.utils as F
else:
    from .. import utils as F

def _index_complete(tus):
    max_t = max([len(t) for t in tus])
    return [t + tuple('' for i in range(max_t - len(t))) for t in tus]

class Builder(object):
    """
    用于生成股票特征和label
    """

    handler = dict()

    @classmethod
    def register_handler(cls, key):
        def register(func):
            cls.handler[key] = func
            return func
        return register

    def __init__(
        self,
        code_col,
        time_col,
        base_df,
        max_feature_window=67,
        max_label_window=1,
        dtype=np.float64,
        n_thread=8,
        target_date=None
    ):
        """
        max_label_window: > 0 for train and test. = 0 for eval, you can not use any properties related to label or any index
        """
        self.max_feature_window = max_feature_window
        self.max_label_window = max_label_window
        self.code_col = code_col
        self.time_col = time_col
        threshold = max_feature_window + max_label_window
        origin_counts = base_df.groupby(self.code_col)[self.time_col].count()
        mask_drop_small_size = np.concatenate([np.zeros(cnt, dtype=np.bool) if cnt < threshold else np.ones(cnt, dtype=np.bool) for cnt in origin_counts])
        self.base_df = base_df[mask_drop_small_size]
        self.codes = self.base_df[self.code_col].unique()
        if target_date is not None:
            ext_df = self.base_df[-self.codes.shape[0]:].copy()
            ext_df[self.code_col] = self.codes
            ext_df[self.time_col] = pd.to_datetime(target_date)
            self.base_df = pd.concat([self.base_df, ext_df], axis=0)
        self.base_df = self.base_df.sort_values([self.code_col, self.time_col]).reset_index(drop=True)
        self.group_sizes = self.base_df.groupby(code_col)[time_col].count()
        self.end_indices = np.array(np.add.accumulate(self.group_sizes) - 1)
        self.start_indices = np.pad(self.end_indices[:-1] + 1, (1, 0), mode='constant', constant_values=(0, np.nan))
        self.size = self.base_df.shape[0]
        self.start_flag = F.index2flag(self.start_indices, self.size)
        self.end_flag = F.index2flag(self.end_indices, self.size)
        # generate mask
        feature_mask_units = [np.concatenate([
            np.zeros(self.max_feature_window - 1, dtype=np.bool),
            np.ones(cnt - self.max_feature_window -
                    self.max_label_window + 1, dtype=np.bool),
            np.zeros(self.max_label_window, dtype=np.bool)
        ]) for cnt in self.group_sizes]
        self.feature_mask = np.concatenate(feature_mask_units)
        if self.max_label_window > 0:
            label_mask_units = [np.concatenate([
                np.zeros(self.max_feature_window, dtype=np.bool),
                np.ones(cnt - self.max_feature_window -
                        self.max_label_window + 1, dtype=np.bool),
                np.zeros(self.max_label_window - 1, dtype=np.bool)
            ]) for cnt in self.group_sizes]
            self.label_mask = np.concatenate(label_mask_units)
            self.label_time = self.base_df[time_col][self.label_mask]
        else:
            self.label_mask = self.feature_mask
        self.dtype = dtype
        self.n_thread = n_thread
        self.do_init()
        self.split_points = [None, None]
        self.split_masks = [self.time_mask(start=None, end=None)]

    def __get_col(self, name):
        try:
            ans = self.col_space.get(name)
            return ans if ans is not None \
                else self.base_df[name]
        except:
            return self.base_df[name]

    def column(self, mode, key, param, is_tmp=True):
        self.param.append({'mode': mode, 'is_temp': is_tmp, 'param': param, 'key': key})
        task = self.handler[mode](self, key, param)
        self.task_buffer.append(task)
        return task.key_names()

    def feature(self, mode, param, key=None):
        key = key if key is not None else "feature_" + str(len(self.feature_names))
        names = self.column(mode, key, param, is_tmp=False)
        self.feature_names.append(names)
        self.feature_keys.append(key)
        return names

    def label(self, mode, param, key=None):
        key = key if key is not None else "feature_" + len(self.label_names)
        names = self.column(mode, key, param, is_tmp=False)
        self.label_names.append(names)
        self.label_keys.append(key)
        return names

    def add_feature(self, key, col):
        self.feature_names.append(key)
        self.feature_keys.append('')
        self.col_space[key] = col
        return [key]

    def add_label(self, key, col):
        self.label_names.append(key)
        self.col_space[key] = col
        return [key]

    def __getitem__(self, key):
        ans = self.__get_col(key)
        if ans is None:
            ans = self.base_df[key]
        return ans

    def __setitem__(self, key, value):
        self.base_df[key] = value

    def time_mask(self, start=None, end=None):
        tm = self.base_df[self.time_col][self.label_mask]
        all_true = tm == tm
        a = (tm >= start) if start is not None else all_true
        b = (tm < end) if end is not None else all_true
        return np.array(a & b)

    def split_by_time(self, split_points, start=None, end=None):
        """
        split data as train, test or train, validation, test in [start, end]

        Args:
            split_points: list of split point,
                eg. ['2013-01-01', '2014-01-01'] means 3 dataset
                    [start, '2013-01-01'), ['2013-01-01', '2014-01-01'), ['2014-01-01', end)
            start: time start, default is None, means no limit(based on original dataset)
            end: time end, default, is None, means no limit(based on original dataset)
        """
        self.split_points = split_points if isinstance(split_points, list) else [split_points]
        self.split_points = [start] + self.split_points + [end]
        self.split_masks = [self.time_mask(st, ed) for st, ed in zip(self.split_points[:-1], self.split_points[1:])]

    def eval(self):
        for task in self.task_buffer:
            for name, col in zip(task.key_names(), task.eval()):
                if name in self.col_space:
                    del self.col_space[name]
                self.col_space[name] = col
        # self.task_buffer = list()

    def __get_2dim_col(self, col):
        x = self.col_space[col]
        if x.ndim == 1:
            return x[:, None]
        if x.ndim == 2:
            return x
        raise Exception("Unexpected feature dim: " + x.ndim)

    def __get_features(self):
        cols = [c for cs in self.feature_names for c in cs]
        data_all = [
            self.map(
                lambda c: self.__get_2dim_col(c)[self.feature_mask][tmask],
                cols
            )
            for tmask in self.split_masks
        ]
        ans = [self.__v_stack(data) for data in data_all]
        return ans

    def __get_labels(self):
        cols = [c for cs in self.label_names for c in cs]
        data_all = [
            self.map(
                lambda c: self.__get_2dim_col(c)[self.label_mask][tmask],
                cols
            )
            for tmask in self.split_masks
        ]
        ans = [self.__v_stack(data) for data in data_all]
        return ans

    def get_feature_index(self, flatten=False):
        tuples = _index_complete([c + (i,)
            for cs in self.feature_names
            for c in cs
            for i in (range(self.col_space[c].shape[1]) if self.col_space[c].ndim == 2 else [''])
        ])
        return pd.MultiIndex.from_tuples(tuples) \
            if flatten is False \
            else ["|".join(str(cs)) for cs in tuples]


    def get_label_index(self, flatten=False):
        tuples = _index_complete([c
            for cs in self.label_names
            for c in cs
        ])
        return pd.MultiIndex.from_tuples(tuples) \
            if flatten is False \
            else ["|".join(str(cs)) for cs in tuples]

    def __get_row_index(self):
        indices = [
            self.base_df[[self.code_col, self.time_col]][self.label_mask][dmask]
            for dmask in self.split_masks
        ]
        return [
            pd.MultiIndex.from_arrays([ind[self.code_col], ind[self.time_col]])
            for ind in indices
        ]

    def do_init(self):
        # print("Builder DO INIT")
        self.col_space = OrderedDict()
        self.feature_names = list()
        self.feature_keys = list()
        self.label_names = list()
        self.label_keys = list()
        self.task_buffer = list()
        self.param = list()


    def get_Xy(self, feature_using_df=False, label_using_df=True, feature_flatten=False, do_init=True):
        """
        返回feature和label，并清理内存
        """
        self.eval()
        features = self.get_feature(using_df=feature_using_df, flatten=feature_flatten, do_init=False)
        labels = self.get_label(using_df=label_using_df, flatten=False, do_init=False)
        if do_init:
            self.do_init()
        return features + labels if len(self.split_points) > 2 else (features, labels)

    def get_feature(self, using_df=False, flatten=False, do_init=False):
        """
        返回特征DataFrame, 使用code和time作为行索引，自动构建列索引

        Args:
            using_df: 是否输出dataframe，默认为False
            flatten: 是否展开列索引，默认为False，表示多级索引。如果你使用xgboost，可以设为True
        """
        data = self.__get_features()
        if using_df is False:
            if do_init:
                self.do_init()
            return data if len(data) > 1 else data[0]
        res_df = [
            pd.DataFrame(
                data=d,
                columns=self.get_feature_index(flatten),
                index=ind
            )
            for d, ind in zip(data, self.__get_row_index())
        ]
        if do_init:
            self.do_init()
        return res_df if len(res_df) > 1 else res_df[0]

    def get_label(self, using_df=True, flatten=False, do_init=False):
        """
        Args:
            using_df: 是否输出dataframe，默认True
            flatten：是否展开索引，默认False，使用多级索引
        """
        data = self.__get_labels()
        if using_df is False:
            if do_init:
                self.do_init()
            return data if len(data) > 1 else data[0]
        res_df = [
            pd.DataFrame(
                data=d,
                columns=self.get_label_index(flatten),
                index=ind
            )
            for d, ind in zip(data, self.__get_row_index())
        ]
        if do_init:
            self.do_init()
        return res_df if len(res_df) > 1 else res_df[0]

    def get_time(self):
        """
        获取每个数据分片的时间列
        """
        res = [self[self.time_col][self.label_mask][tmask] for tmask in self.split_masks]
        return res if len(res) > 1 else res[0]

    def get_code(self):
        """
        获取每个时间分片的code
        """
        res = [self[self.code_col][self.label_mask][tmask] for tmask in self.split_masks]
        return res if len(res) > 1 else res[0]

    def get_df(self):
        """
        获取每个分片对应的原始DataFrame
        """
        res = [self.base_df[self.label_mask][tmask] for tmask in self.split_masks]
        return res if len(res) > 1 else res[0]

    @staticmethod
    def feature_window_extract(col, window):
        return F.window_extract(col, window)

    @staticmethod
    def feature_change_rate(col, base_col=None, period=1):
        return F.pct_change(col, base_col, period)

    def map(self, f, vs):
        """
        并行对vs中的元素作用函数f

        Args:
            f: 应用的函数
            vs: 被应用的对象列表
        """
        if self.n_thread == 1:
            return [f(v) for v in vs]
        with ThreadPool(self.n_thread) as pool:
            return list(pool.map(f, vs))

    def __v_stack(self, data):
        assert all([d.shape[0] == data[0].shape[0] for d in data]), "all data must be same shape"
        assert all([d.ndim == 2 for d in data]), "only 2d array supported"
        N_ROW = data[0].shape[0]
        col_shapes = [d.shape[1] for d in data]
        N_COL = sum(col_shapes)
        N_data = len(data)
        acc_shapes = np.add.accumulate(np.array([0] + col_shapes))
        res = np.zeros((N_ROW, N_COL), dtype=self.dtype)
        @jit
        def assign(i):
            res[:, acc_shapes[i]:acc_shapes[i+1]] = data[i]
        self.map(assign, range(N_data))
        return res

    def unmask(self, cols, fill_value=0.0):
        '''
        把已经被builder mask输出的结果重新组装成一列，缺失位置用fill_value填充
        '''
        return F.unmask(np.concatenate(cols), self.label_mask, fill_value)

    def label_rank(self, cols, inverse=False):
        '''
        Args:
            cols: label_col split by time OR numpy ndarray
        '''
        return_list = True
        times = self.get_time()
        assert isinstance(cols, list) == isinstance(times, list), "You shall not pass! cols must be same type as self.get_time()"
        if not isinstance(cols, list):
            cols = [cols]
            times = [times]
            return_list = False
        ans = [
            pd.DataFrame({'v': np.array(c), 'g': np.array(t)}).groupby("g")["v"].rank(pct=True, ascending=not inverse)
            for c, t in zip(cols, times)
        ]
        return ans if return_list is True else ans[0]

class ColumnBase(object):
    def __init__(self, base, key, param):
        self.key = key
        self.base = base
        self.param = param
        self.init()

    def init(self):
        pass

    def key_names(self):
        return [(self.key,) + ((l,) if isinstance(l, str) else tuple(l)) for l in self.names()]

    def names(self):
        raise Exception("NoInplementation")

    def eval(self):
        raise Exception("NoInplementation")


@Builder.register_handler('raw')
class RawFeatureGenerate(ColumnBase):
    def init(self):
        self.cols = self.param['col']
        #是否是label当天的特征，比如，可以把open，当天的weekday作为当天特征
        #千万不要把当天的均价或者close作为特征
        self.on_label = self.param.get("on_label", False)

    def names(self):
        return self.cols

    def eval(self):
        cols = [self.base[c] if self.on_label is False else self.base[c].shift(-1) for c in self.cols]
        return cols

if __name__ == "__main__":
    import doctest
    doctest.testmod()
