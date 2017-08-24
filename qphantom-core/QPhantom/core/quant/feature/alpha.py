from QPhantom.core.quant import Builder, ColumnBase
import time
from contextlib import contextmanager
import numpy as np
import bottleneck as bn
from numba import jit
import talib
from sklearn.utils import shuffle

@jit
def _alpha53(arr_h, arr_l, arr_c,  arr_o, window, start_indices):
    p1  = np.where((arr_c - arr_l)!=0, (2*arr_c - arr_h - arr_l) / (arr_c - arr_l), 1)
    p2  = np.roll(p1,window)
    res = p1 - p2
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

def alpha53(arr_h, arr_l, arr_c,  arr_o, window=16, start_indices=None):
    """
    calculate wr of specific window

    Examples:
    >>> a = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> b = np.array([ 4.,  3.,  8.,  7.,  6.,  2.,  5.,  9.,  1.])
    >>> c = np.array([ 7.,  5.,  1.,  3.,  9.,  6.,  8.,  2.,  4.])
    >>> alpha53(a, b, c, 3, [0,5])
    array([        nan,         nan,         nan, -1.5       ,  2.33333333,
                   nan,         nan,         nan, -1.66666667])
    """
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    arr_c = np.array(arr_c, dtype=np.float64)
    arr_o = np.array(arr_o, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _alpha53(arr_h, arr_l, arr_c, arr_o, window, start_indices)


@jit
def _kdj(arr_c, arr_h, arr_l, arr_o, window, start_indices):
    N = arr_c.shape[0]
    K = np.zeros_like(arr_c)
    D = np.zeros_like(arr_c)
    RSV = np.where(arr_h-arr_l!=0,100 * (arr_c-arr_l) / (arr_h-arr_l), 50)
    N_GROUP = start_indices.shape[0]
    j = 0
    for i in range(N):
        if j < N_GROUP and start_indices[j] == i:
            K[i] = 50
            D[i] = 50
            j += 1
        else:
            K[i] =  RSV[i]*1/3 + K[i-1]*2/3
            D[i] =  K[i]*1/3   + D[i-1]*2/3
    return K, D

def kdj(arr_c, arr_h, arr_l, arr_o, window,start_indices=None):
    """
    calculate kdj of specific window

    Examples:
    >>> a = np.array([ 4.,  3.,  8.,  7.,  6.,  2.,  5.,  9.,  1.])
    >>> b = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> c = np.array([ 7.,  5.,  1.,  3.,  9.,  6.,  8.,  2.,  4.])
    >>> kdj(a,b,c,[0,5])
    (array([  50.        ,    0.        ,  116.66666667,   11.11111111,
             32.40740741,   50.        ,  -66.66666667,   -5.55555556,
            -53.7037037 ]), array([ 50.        ,  33.33333333,  61.11111111,  44.44444444,
            40.43209877,  50.        ,  11.11111111,   5.55555556, -14.19753086]))
    """
    arr_c = np.array(arr_c, dtype=np.float64)
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    arr_o = np.array(arr_o, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _kdj(arr_c, arr_h, arr_l, arr_o,window, start_indices)

@jit
def _cr(arr_o, arr_c, arr_h, arr_l, window, start_indices):
    arr_m   = (arr_o + arr_c + arr_h + arr_l)/4
    arr_p1 = arr_h - np.roll(arr_m, 1)
    arr_p2 = np.roll(arr_m, 1) - arr_l
    arr_p1 = bn.move_mean(arr_p1,window)
    arr_p2 = bn.move_mean(arr_p2,window)
    res = np.zeros_like(arr_c)
    res = np.where(arr_p2!=0, 100*arr_p1/arr_p2, 50)
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

def cr(arr_o, arr_c, arr_h, arr_l, window, start_indices=None):
    """
    calculate cr of specific window

    Examples:
    >>> a = np.array([ 4.,  3.,  8.,  7.,  6.,  2.,  5.,  9.,  1.])
    >>> b = np.array([ 4.,  7.,  3.,  1.,  5.,  2.,  9.,  8.,  6.])
    >>> c = np.array([ 7.,  5.,  1.,  3.,  9.,  6.,  8.,  2.,  4.])
    >>> d = np.array([ 1.,  9.,  2.,  4.,  8.,  6.,  3.,  7.,  5.])
    >>> cr(a,b,c,d,2,[0,5])
    array([          nan,           nan,  400.        , -157.14285714,
           -100.        ,           nan,           nan, -100.        , -900.        ])
    """
    arr_o = np.array(arr_o, dtype=np.float64)
    arr_c = np.array(arr_c, dtype=np.float64)
    arr_h = np.array(arr_h, dtype=np.float64)
    arr_l = np.array(arr_l, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _cr(arr_o, arr_c, arr_h, arr_l, window, start_indices)

@jit
def _cci(arr_h, arr_l, arr_c, arr_o, window, start_indices):

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

def cci(arr_h, arr_l, arr_c, arr_o, window, start_indices=None):
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
    arr_o = np.array(arr_o, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _cci(arr_h, arr_l, arr_c, arr_o, window, start_indices)

@jit
def _wr(arr_h, arr_l, arr_c,  arr_o, window, start_indices):
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

def wr(arr_h, arr_l, arr_c,  arr_o, window=16, start_indices=None):
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
    arr_o = np.array(arr_o, dtype=np.float64)
    start_indices = np.array(start_indices) if start_indices is not None else np.array([0])
    return _wr(arr_h, arr_l, arr_c, arr_o, window, start_indices)

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


@Builder.register_handler('alpha')
class Alpha101FeatureGenerate(ColumnBase):
    def init(self):
        self.open_col = self.param['open']
        self.high_col = self.param['high']
        self.low_col = self.param['low']
        self.close_col = self.param['close']
        self.window = self.param['window']

    def names(self):
        return ['alpha53','ar','br','wr','cci','k','d']

    #TODO: Implement
    def eval(self):
        col_o = self.base[self.open_col]
        col_h = self.base[self.high_col]
        col_l = self.base[self.low_col]
        col_c = self.base[self.close_col]

        alpha_53 = alpha53(col_h, col_l, col_c, col_o, self.window, self.base.start_indices)
        ar,br = arbr(col_o, col_h, col_l, col_c, self.window, self.base.start_indices)
        cci_one   = cci(col_h, col_l, col_c, col_o, self.window, self.base.start_indices)
        wr_one    = wr(col_h, col_l, col_c, col_o, self.window, self.base.start_indices)
        k,d   = kdj(col_c, col_h, col_l, col_o, self.window, self.base.start_indices)
        #cr_one    = cr(col_o, col_c, col_h, col_h, self.window, self.base.start_indices)

        cols = [alpha_53, ar, br, wr_one, cci_one, k, d]
        return cols
