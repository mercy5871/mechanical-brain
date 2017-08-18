import time
import talib
import pandas as pd
import numpy as np
import bottleneck as bn
from numba import jit
from contextlib import contextmanager
from sklearn.utils import shuffle


def index2flag(indices, size):
    """
    convert index to boolean array which True at each index

    Args:
        indices: indices of True of array
        size: length of array

    Examples:
    >>> index2flag([1, 2, 3], 5)
    array([False,  True,  True,  True, False], dtype=bool)
    >>> index2flag([1, 7, 11, 2], 12)
    array([False,  True,  True, False, False, False, False,  True, False,
           False, False,  True], dtype=bool)
    """
    a = np.zeros(size, dtype=np.bool)
    a[indices] = True
    return a

def flag2index(flag):
    """
    return index of non-zero

    Examples:
    >>> flag2index([True, False, True, True, False])
    array([0, 2, 3])
    """
    return np.nonzero(flag)[0]

def unmask(arr, mask, fill_value):
    '''
    undo array mask

    Examples:
    >>> unmask([1, 4, 2], [False, False, False, True, False, True, True], -1)
    array([-1, -1, -1,  1, -1,  4,  2])
    '''
    mask = np.array(mask)
    arr = np.array(arr)
    res = np.full(mask.shape[0], fill_value, np.array(fill_value).dtype)
    res[flag2index(mask)] = arr
    return res

@contextmanager
def measureTime(title, printer=print):
    '''
    评估计算过程的时间

    Args:
        title: 打印日志的标记，方便查看是哪个部分的代码的运行时间
        printer: 打印日志的方式，默认是print，可以自定义到log文件
    '''
    t1 = time.time()
    yield
    t2 = time.time()
    printer('%s: %0.2f seconds elapsed' % (title, t2 - t1))


def acc_argmax(values):
    '''
    accumulated argmax

    Args:
        values: original numeric array

    Returns:
        indices of each max
    '''
    value_cummax = np.maximum.accumulate(values)
    max_drawdowns = (value_cummax - values) / value_cummax
    cummax_indices = np.nonzero(values == value_cummax)[0]
    value_argcummax = np.zeros_like(values, dtype=np.int32)
    value_argcummax[cummax_indices] = [cummax_indices]
    return np.maximum.accumulate(value_argcummax)


def acc_argmin(values):
    '''
    accumulated argmin

    Args:
        values: original numeric array

    Returns:
        indices of each min
    '''
    return acc_argmax(-np.array(values))


def group_nlargest(df, group_by, order_by, n, sort_mode=False):
    '''
    get n largest of each group

    Args:
        df: DataFrame input
        group_by: group by key
        order_by: order by key
        n: num of largest rows
        sort_mode: whether to use sort mode, default False, sort mode is a bit slower but robust

    Returns:
        DataFrame of n largest
    '''
    if sort_mode is False:
        rm_index = list(
            range(len(group_by) if isinstance(group_by, list) else 1))
        return df.loc[df.groupby(group_by)[order_by].nlargest(n).reset_index(rm_index, drop=True).index.get_values()]
    else:
        if not isinstance(group_by, list):
            group_by = [group_by]
        if not isinstance(order_by, list):
            order_by = [order_by]
        group_ascending = [True for _ in group_by]
        order_ascending = [False for _ in order_by]
        return df.sort_values(group_by + order_by, ascending=group_ascending + order_ascending).groupby(group_by).head(n)


def group_nsmallest(df, group_by, order_by, n, sort_mode=False):
    '''
    get n smallest of each group

    Args:
        df: DataFrame input
        group_by: group by key
        order_by: order by key
        n: num of smallest rows
        sort_mode: whether to use sort mode, default False, sort mode is a bit slower but robust

    Returns:
        DataFrame of n smallest
    '''
    if sort_mode is False:
        rm_index = list(
            range(len(group_by) if isinstance(group_by, list) else 1))
        return df.loc[df.groupby(group_by)[order_by].nsmallest(n).reset_index(rm_index, drop=True).index.get_values()]
    else:
        if not isinstance(group_by, list):
            group_by = [group_by]
        if not isinstance(order_by, list):
            order_by = [order_by]
        group_ascending = [True for _ in group_by]
        order_ascending = [True for _ in order_by]
        return df.sort_values(group_by + order_by, ascending=group_ascending + order_ascending).groupby(group_by).head(n)


def window_extract(col, window):
    res = np.transpose(np.array([np.roll(col, i)
                                 for i in range(window - 1, -1, -1)]))
    res[:window] = np.nan
    return res


def pct_change(col, base_col=None, period=1, dtype=np.float64):
    """
    返回col相对于base_col延后period的比例，以base_col为基准，period表示col延迟的周期。
    如果period是负数，则把以上结果前移|period|单位
    Args:
        col: 目标列或者目标dataframe
        base_col: 基准列，如果不指定，用col本身作为基准
        period: 时间周期，默认是一个周期后的pct_change, 负周期表示period个周期已经发生的变化
    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> pct_change([1, 2, 3], period=-1)
    array([ nan,  1. ,  0.5])
    >>> pct_change([1, 2, 3], period=1)
    array([ 1. ,  0.5,  nan])
    >>> pct_change(df, period=-1)
    array([[  nan,   nan],
           [ 1.  ,  0.25],
           [ 0.5 ,  0.2 ]])
    >>> pct_change(df, period=1)
    array([[ 1.  ,  0.25],
           [ 0.5 ,  0.2 ],
           [  nan,   nan]])
    >>> pct_change(df, base_col=df['a'], period=1)
    array([[ 1. ,  4. ],
           [ 0.5,  2. ],
           [ nan,  nan]])
    >>> pct_change(df, base_col=df['a'], period=-1)
    array([[ nan,  nan],
           [ 1. ,  4. ],
           [ 0.5,  2. ]])
    """
    col = np.array(col, dtype=dtype)
    if base_col is None:
        base_col = col
    else:
        base_col = np.array(base_col, dtype=dtype)
    if period < 0:
        base_col = np.roll(base_col, -period, axis=0).astype(dtype)
        base_col[:-period] = np.nan
    else:
        col = np.roll(col, -period, axis=0).astype(dtype)
        col[-period:] = np.nan
    if len(base_col.shape) < len(col.shape):
        base_col = base_col[:, None]
    return col / base_col - 1.0


def last_nonzero_index(arr, initial=0):
    """
    get last nonzero value's index for each value

    Examples:
    >>> last_nonzero_index([0, 0, 1.7, 0, 7.0, 0, 0, 3.5, 0, 0, 0])
    array([0, 0, 2, 2, 4, 4, 4, 7, 7, 7, 7])
    """
    ind = np.nonzero(arr)[0]
    cnt = np.cumsum(np.array(arr, dtype=bool))
    return np.where(cnt, ind[cnt-1], 0)

def fill_zeros_with_last(arr, initial=0, inverse=False):
    """
    fill zero with last non-zero value before

    Examples:
    >>> fill_zeros_with_last([1, 0, 0, 2.4, 0, 0, 7.1, 0, 3, 0])
    array([ 1. ,  1. ,  1. ,  2.4,  2.4,  2.4,  7.1,  7.1,  3. ,  3. ])
    >>> fill_zeros_with_last([0.0, 0.0, 1, 0, 0, 2.4, 0, 0, 7.1, 0, 3, 0], initial=0)
    array([ 0. ,  0. ,  1. ,  1. ,  1. ,  2.4,  2.4,  2.4,  7.1,  7.1,  3. ,
            3. ])
    >>> fill_zeros_with_last([0.0, 0.0, 1, 0, 0, 2.4, 0, 0, 7.1, 0, 3, 0], initial=1)
    array([ 1. ,  1. ,  1. ,  1. ,  1. ,  2.4,  2.4,  2.4,  7.1,  7.1,  3. ,
            3. ])
    >>> fill_zeros_with_last([0.0, 0.0, 1, 0, 0, 2.4, 0, 0, 7.1, 0, 3, 0], initial=1, inverse=True)
    array([ 1. ,  1. ,  1. ,  2.4,  2.4,  2.4,  7.1,  7.1,  7.1,  3. ,  3. ,
            1. ])
    """
    arr = np.array(arr) if inverse is False else np.array(arr)[::-1]
    ind = np.nonzero(arr)[0]
    cnt = np.cumsum(np.array(arr, dtype=bool))
    ans = np.where(cnt, arr[ind[cnt-1]], initial)
    return ans if inverse is False else ans[::-1]

@jit
def _ema(arr, alpha, start_indices):
    N = arr.shape[0]
    res = np.zeros_like(arr)
    s = 0.0
    s_ = 0.0
    j = 0
    N_GROUP = start_indices.shape[0]
    for i in range(0, N):
        if j < N_GROUP and start_indices[j] == i:
            s, s_ = 0.0, 0.0
            j += 1
        s =  arr[i] + (1 - alpha) * s
        s_ = 1.0 + (1 - alpha) * s_
        res[i] = s / s_
    return res

def ema(arr, alpha, start_indices=None):
    """
    calculate EMA of specific window

    Examples:
    >>> ema(np.arange(10), 0.1, [0, 2])
    array([ 0.        ,  0.52631579,  2.        ,  2.52631579,  3.0701107 ,
            3.63128817,  4.20971405,  4.8052177 ,  5.41759326,  6.04660125])
    """
    arr = np.array(arr, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _ema(arr, alpha, start_indices)


@jit
def _rsi(arr, window, start_indices):
    N = arr.shape[0]
    res = np.zeros_like(arr)
    up   = 0.0
    down = 0.0
    pre_cnt = 0
    j = 0
    N_GROUP    = start_indices.shape[0]
    arr_dff    = arr - np.roll(arr, 1)
    for i in range(window):
        cur = arr_dff[i]
        if cur > 0:
            up += cur
        else:
            down += cur
        res[i] = np.nan
        pre_cnt += 1
    for i in range(window, N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        cur = arr_dff[i]
        if cur > 0:
            up += cur
        else:
            down += cur
        p_cur = arr_dff[i - window]
        if p_cur > 0:
            up -= p_cur
        else:
            down -= p_cur
        res[i] = np.nan if pre_cnt < window else (100 * up / (up - down) if up - down != 0.0 else 50)
        pre_cnt += 1
    return res

def rsi(arr, window, start_indices=None):
    """
    calculate rsi of specific window

    Examples:
    >>> a = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> rsi(a,2,[0,5])
    array([         nan,          nan,  42.85714286,   0.        ,
            66.66666667,  57.14285714,  70.        ,  87.5       ,   0.        ])
    """
    arr = np.array(arr, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _rsi(arr, window, start_indices)


@jit
def _psy(arr, window, start_indices):
    N = arr.shape[0]
    res = np.zeros_like(arr)
    up   = 0.0
    pre_cnt = 0
    j = 0
    N_GROUP    = start_indices.shape[0]
    arr_dff    = arr - np.roll(arr, 1)
    for i in range(window):
        cur = arr_dff[i]
        if cur > 0:
            up += 1
        res[i] = np.nan
        pre_cnt += 1
    for i in range(window, N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        cur = arr_dff[i]
        if cur > 0:
            up += 1
        p_cur = arr_dff[i - window]
        if p_cur > 0:
            up -= 1
        res[i] = np.nan if pre_cnt < window else (100 * up / window)
        pre_cnt += 1
    return res

def psy(arr, window, start_indices=None):
    """
    calculate psy of specific window

    Examples:
    >>> a = np.array([ 2.,  4.,  3.,  5.,  8.,  6.,  9.,  1.,  7.])
    >>> psy(a,4,[0,5])
    array([ nan,  nan,  nan,  nan,  75.,  50.,  75.,  50.,  50.])
    """
    arr = np.array(arr, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _psy(arr, window, start_indices)

@jit
def _bias(arr, window, start_indices):
    arr_m   = bn.move_mean(arr, window)
    res = np.where(arr_m!=0, 100*(arr - arr_m) / arr_m, 50)
    pre_cnt = 0.0
    N = arr.shape[0]
    N_GROUP = start_indices.shape[0]
    j = 0
    for i in range(N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        if pre_cnt < window:
            res[i] = np.nan
        pre_cnt += 1
    return res

def bias(arr, window, start_indices=None):
    """
    calculate bias of specific window

    Examples:
    >>> a = np.array([ 2.,  4.,  3.,  5.,  8.,  6.,  9.,  1.,  7.])
    >>> bias(a,2,[0,5])
    array([         nan,          nan, -14.28571429,  25.        ,
            23.07692308,          nan,          nan, -80.        ,  75.        ])

    """
    arr = np.array(arr, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _bias(arr, window, start_indices)


@jit
def _cci(arr_h, arr_l, arr_c,  window, start_indices):

    TP = (arr_h + arr_l + arr_c) / 3
    MA = bn.move_mean(arr_c,window)
    MD = bn.move_mean((MA - arr_c),window)
    res = np.where(MD!=0, (TP - MA) / MD /0.015, 0)

    pre_cnt = 0.0
    N = arr_c.shape[0]
    N_GROUP = start_indices.shape[0]
    j = 0
    for i in range(N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        if pre_cnt < window:
            res[i] = np.nan
        pre_cnt += 1
    return res

def cci(arr_h, arr_l, arr_c,  window, start_indices=None):
    """
    calculate cci of specific window

    Examples:
    >>> a = np.array([ 4.,  3.,  8.,  7.,  6.,  2.,  5.,  9.,  1.])
    >>> b = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> c = np.array([ 7.,  5.,  1.,  3.,  9.,  6.,  8.,  2.,  4.])
    >>> cci(a,b,c,2,[0,5])
    array([          nan,           nan,   44.44444444,  222.22222222,
            -22.22222222,           nan,           nan,   88.88888889,
             44.44444444])
    """
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    arr_c = np.array(arr_c, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _cci(arr_h, arr_l, arr_c,  window, start_indices)

@jit
def _wr(arr_h, arr_l, arr_c,  window, start_indices):
    high_n = bn.move_max(arr_h,window)
    low_n  = bn.move_min(arr_l,window)
    res = np.where(high_n - low_n!=0, 100*(high_n - arr_c) / (high_n - low_n), 50)

    pre_cnt = 0.0
    N = arr_c.shape[0]
    N_GROUP = start_indices.shape[0]
    j = 0
    for i in range(N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        if pre_cnt < window:
            res[i] = np.nan
        pre_cnt += 1
    return res

def wr(arr_h, arr_l, arr_c,  window=16, start_indices=None):
    """
    calculate wr of specific window

    Examples:
    >>> a = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> b = np.array([ 4.,  3.,  8.,  7.,  6.,  2.,  5.,  9.,  1.])
    >>> c = np.array([ 7.,  5.,  1.,  3.,  9.,  6.,  8.,  2.,  4.])
    >>> wr(a,b,c,2,[0,5])
    array([          nan,           nan,  150.        ,   -0.        ,
            400.        ,           nan,           nan,  175.        ,
             57.14285714])
    """
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    arr_c = np.array(arr_c, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _wr(arr_h, arr_l, arr_c,  window, start_indices)

@jit
def _arbr(arr_o, arr_h, arr_l, arr_c,  window, start_indices):

    numerator_ar   = bn.move_sum((arr_h - arr_o),window)
    denominator_ar = bn.move_sum((arr_o - arr_l),window)
    ar = np.where(denominator_ar!=0,100*numerator_ar / denominator_ar, 50)

    arr_y = np.roll(arr_c,1)
    numerator_br   = bn.move_sum((arr_h - arr_y),window)
    denominator_br = bn.move_sum((arr_y - arr_l),window)
    br = np.where(denominator_br!=0, 100*numerator_br / denominator_br, 50)

    pre_cnt = 0.0
    N = arr_c.shape[0]
    N_GROUP = start_indices.shape[0]
    j = 0
    for i in range(N):
        if j < N_GROUP and start_indices[j] == i:
            pre_cnt = 0
            j += 1
        if pre_cnt < window:
            ar[i] = np.nan
            br[i] = np.nan
        pre_cnt += 1
    return ar, br

def arbr(arr_o, arr_h, arr_l, arr_c,  window=26, start_indices=None):
    """
    calculate arbr of specific window

    Examples:
    >>> a = np.array([ 2.,  4.,  3.,  5.,  8.,  6.,  9.,  1.,  7.])
    >>> b = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> c = np.array([ 2.,  3.,  5.,  1.,  4.,  8.,  9.,  7.,  6.])
    >>> d = np.array([ 8.,  5.,  9.,  2.,  6.,  3.,  4.,  1.,  7.])
    >>> arbr(a,b,c,d,2,[0,5])
    (array([          nan,           nan, -300.        , -200.        ,
            -87.5       ,           nan,           nan, -116.66666667, -120.        ]), array([          nan,           nan,  -60.        , -125.        ,
            -83.33333333,           nan,           nan, -111.11111111, -112.5       ]))
    """
    arr_o = np.array(arr_o, dtype=np.float64)
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    arr_c = np.array(arr_c, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _arbr(arr_o, arr_h, arr_l, arr_c,  window, start_indices)


def group_rank(arr, groups):
    df = pd.DataFrame({'value': np.array(arr), 'group': np.array(groups)})
    return np.array(df.groupby("group")["value"].rank(pct=True))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
