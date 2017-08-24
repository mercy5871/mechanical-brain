# %%

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# %%

import pandas as pd
import numpy as np
import os
from QPhantom.core.metrics import Metrics

from datetime import datetime

M = Metrics(size=(12, 12), fontsize=10, title_size=12, text_size=11, dpi=200)

# %%

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 12))
mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

 # import my -difined font
from QPhantom.core.metrics import zh_font
plt.xlabel('Smarts')
# assign fontproperties
plt.ylabel('概率',fontproperties=zh_font)

plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()

# %%

import numpy as np
df = pd.read_csv('data/predict_sample.csv')

M.plot_predict_summary(df['label'], df['predict_score'], beta=1)

M.plot_predict_summary(df['label'], np.ones(df['predict_score'].shape[0])* 0, beta=1)

# %%

M.plot_precision_recall_fscore(df['label'], df['predict_score'], beta=1)

# %%

M.plot_roc_auc(df['label'], df['predict_score'])

# %%

M.plot_PR(df['label'], df['predict_score'])

# %%

value_df = pd.read_csv('data/value_data.csv')

M.plot_trade_summary(pd.to_datetime(value_df['date']), value_df['value'], value_df['benchmark'])

# %%

base_df = pd.read_hdf("data/test_data.hdf5", key="test")

# %%

base_df = base_df[base_df['date'] < datetime.strptime("20150301", "%Y%m%d")]

base_df = base_df[base_df['date'] >= datetime.strptime("20140901", "%Y%m%d")]

#base_df.to_hdf("test/data/test_data.hdf5.200", key="test")

# %%

indices, actual_label, predict_score, window_size, beta = base_df['date'], base_df['label'], base_df['predict_score'], 22, 1

M.plot_window_precision_recall_fscore(indices=indices, actual_label=actual_label, predict_score=predict_score, threshold=0.42, window_size=window_size, beta=beta)

# %%

rand_v = np.random.rand(indices.shape[0])

# %%

plt.plot(np.arange(rand_v.shape[0])[:10], rand_v[:10], 'x', color='b', markersize=10, markeredgewidth=1)
plt.show()

# %%

ans = M.score_on_topk(actual=rand_v, predict=predict_score, groups=indices, plot_move_window=20, k=50, show_score=True)


# %%

plt.figure(figsize=(8, 8))
plt.plot(ans["group"], ans["rank"], "o", color="0.5", alpha=0.4, markersize=5)
plt.show()
