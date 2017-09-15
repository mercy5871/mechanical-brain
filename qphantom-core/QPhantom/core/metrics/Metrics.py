#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:52:06 2017

@author: mercy
"""

import sys
import io
import numpy as np
import pandas as pd
import seaborn as sns
import collections
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve as pr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import bottleneck as bn
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY, date2num, MonthLocator, HourLocator, MinuteLocator
from matplotlib.lines import Line2D, TICKLEFT, TICKRIGHT
from matplotlib.patches import Rectangle

import QPhantom.core.utils as qutils

sns.set(color_codes=True)

# plt.rcParams['font.sans-serif'] = ["SimHei", "WenQuanYi Micro Hei"] + plt.rcParams['font.sans-serif'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def _candlestick(ax, quotes, width=0.2, colorup='k', colordown='r',
                 alpha=1.0, ochl=False):

    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes
    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added
    """

    OFFSET = width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


class Metrics(object):
    '''
    用于展示常用的分类评估，回测结果等
    '''

    def __init__(self, size=(14, 14), fontsize=13, title_size=16, text_size=15, plotter=None, dpi=None):
        """
        Args:
            size: plot figure size
            fontsize: font size for legend and so on
            title_size: font size for title_size
            text_size: font size for table content
            plotter: object has image(image_in_bytes) method
            dpi: plotter's dpi of image
        """
        self.size = size
        self.width = self.size[0]
        self.height = self.size[1]
        self.xticks01 = np.linspace(0, 1, 11)
        self.yticks01 = np.linspace(0, 1, 11)[1:]
        self.fontsize = fontsize
        self.title_size = title_size
        self.text_size = text_size
        # plt.tick_params(axis='both', which='major', labelsize=self.fontsize)
        # plt.tick_params(axis='both', which='minor', labelsize=8)
        self.plotter = plotter
        self.dpi = dpi if dpi is not None else 600
        self.show = self.__plot_png if self.plotter is not None else self.__default_plot

    def __plot_png(self):
        with io.BytesIO() as buf:
            plt.savefig(buf, format="png", dpi=self.dpi)
            plt.close()
            buf.seek(0)
            img = buf.read()
            self.plotter.image(img)

    def __default_plot(self):
        plt.show()
        plt.close()

    def risk_analysis(self, value, benchmark, frequency='D', Rf=0.03):
        '''
        返回sharp,Sortino,IR,Alpha,Beta等重要指标
        参数account是DataFrame格式，包含2列['value','benchmark']

        Args:
            account:
            freq: 调仓频率，'D':日频 'W':周频 'M':月频

        value:    账户内资本金额
        benchmark:基准指数具体点位

        '''

        if frequency == 'D':
            freq = 250
        elif frequency == 'W':
            freq = 50
        elif frequency == 'M':
            freq = 12

        value = np.array(value)
        benchmark = np.array(benchmark)
        n = len(value)
        an = value[-1]
        a0 = value[0]
        bn = benchmark[-1]
        b0 = benchmark[0]

        account = pd.DataFrame({"value": value, "benchmark": benchmark})

        account_cummax = np.maximum.accumulate(account['value'])
        max_drawdowns= (account_cummax - account['value'])/account_cummax

        account['account_return'] = account['value'].pct_change(periods=1, fill_method='pad')
        account['bench_return'] = account['benchmark'].pct_change(periods=1, fill_method='pad')
        account['a-b'] = account['account_return'] - account['bench_return']

        ab_cov = account.loc[:,['account_return','bench_return']].cov()
        ab_cov = ab_cov['account_return']['bench_return']
        a_std = account['account_return'].std()
        b_std = account['bench_return'].std()
        Bench_return = (bn/b0)**(freq/n)-1
        Annual_return = (an/a0)**(freq/n)-1
        Volatility = np.nanstd(account['account_return'])*np.sqrt(freq)
        return_std = np.nanstd(account['account_return'])
        Sharp = \
            (account['account_return'].mean() - ((1 + Rf)**(1/freq) - 1))/return_std * np.sqrt(freq) \
            if return_std > 0.0 else np.nan
        ab_mean, ab_std = np.nanmean(account['a-b']), np.nanstd(account['a-b'])
        IR = np.nanmean(account['a-b']) / np.nanstd(account['a-b']) * np.sqrt(freq) if ab_mean != 0 else 0
        Beta = ab_cov/(b_std*b_std)
        Alpha= Annual_return-Rf-Beta*(Bench_return-Rf)
        Max_drawdown = max_drawdowns.max()
        downside = account['account_return']
        downside = downside[downside<((1 + Rf)**(1/freq) - 1)]
        if downside.shape[0] == 0:
            Sortino = np.nan
        else:
            down_std = np.std(downside)
            if down_std == 0:
                Sortino = np.nan
            else:
                Sortino = (account['account_return'].mean() - ((1 + Rf)**(1/freq) - 1))/down_std * np.sqrt(freq)

        Calmar = (Annual_return - Rf) / Max_drawdown

        return [
            ("Return", Annual_return),
            ("Alpha", Alpha),
            ("Beta", Beta),
            ("Sharp", Sharp),
            ("Sortino", Sortino),
            ("Calmar", Calmar),
            ("Max_drawdown", Max_drawdown),
            ("IR", IR),
            ("Volatility", Volatility)
        ]

    @staticmethod
    def __try_convert_as_mdate(values):
        try:
            # try convert python datetime
            values = pd.to_datetime(values)
            if hasattr(values, 'dt'):
                values = values.dt.to_pydatetime()
            elif hasattr(values, 'to_pydatetime'):
                values = values.to_pydatetime()
            values = date2num(values)
        except Exception as e:
            print("convert_datetime_fail", file=sys.stderr)
            print(e, file=sys.stderr)
            #no convert need
            pass
        return values

    def apply_xticks(self, indices, num_ticks=20, rotate=True):
        step = len(indices) // num_ticks
        xticks = list(indices[0:-step:step]) + [indices[-1]]
        plt.xlim([indices[0], indices[-1]])
        if rotate == True:
            plt.xticks(xticks, fontsize=self.fontsize, rotation=30, ha='right')
        else:
            plt.xticks(xticks, fontsize=self.fontsize)

    def apply_date_xticks(self, ax=None, major_unit='week', minor_unit='day', major_format="%Y-%m-%d", xdata=None):
        """
        Params:
        major_unit: major locator unit, could be {month, day, week, hour, minute}
        minor_unit: minor locator unit,
        """
        locators = {
            "month": MonthLocator(),
            "month3": MonthLocator(bymonth=[1,4,7,10]),
            "week": WeekdayLocator(MONDAY),
            "day": DayLocator(),
            "hour": HourLocator(),
            "minute": MinuteLocator
        }
        ax = plt.gca() if ax is None else ax
        ax.xaxis.set_major_locator(locators[major_unit])
        if minor_unit is not None:
            ax.xaxis.set_minor_locator(locators[minor_unit])
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.margins(x=0)
        plt.setp(plt.gca().get_xticklabels(), fontsize=self.fontsize, rotation=30, horizontalalignment='right')
        if xdata is not None:
            return self.__try_convert_as_mdate(xdata)

    def score_on_topk(self, actual, predict, groups=None, mask=None, k=50, ascending=False, show_rank=True, show_score=False, show_range=False, plot_move_window=1, rotation=30):
        """
        return average score and rank on topk in each group

        return DataFrame("group", "actual", "rank")
        """
        actual = np.array(actual)
        predict = np.array(predict)
        assert actual.shape[0] == predict.shape[0], "actual and predict size must be equal"
        if groups is not None:
            groups = np.array(groups)
            assert actual.shape[0] == groups.shape[0], "actual and groups size must be equal"
        size = actual.shape[0]
        mask = np.ones(size, dtype=np.bool) if mask is None else np.array(mask, dtype=np.bool)
        indices = np.arange(size)[mask].copy()
        np.random.shuffle(indices)
        df = pd.DataFrame(collections.OrderedDict([
            ("group", groups[indices] if groups is not None else 0),
            ("actual", actual[indices]),
            ("predict", predict[indices])
        ])).sort_values(["group", "predict"], ascending=[True, ascending]).reset_index(drop=True)
        df["rank"] = df.groupby("group")["actual"].rank(ascending=ascending, pct=True)
        rdf = df.groupby("group").head(k).groupby("group")[["actual", "rank"]].mean()
        rdf["random"] = np.array(df.groupby("group")["actual"].mean())
        rdf[['mean', 'min', 'max', 'std']] = df.groupby("group")['predict'].agg(['mean', 'min', 'max', 'std'])
        rdf.reset_index(inplace=True)
        if show_rank:
            plt.figure(figsize=self.size)
            plt.title(f"Rank@Top_{k}", fontsize=self.title_size)
            plt.plot(
                np.array(rdf["group"])[plot_move_window-1:],
                bn.move_mean(rdf["rank"], plot_move_window)[plot_move_window-1:],
                label=f"mean:{rdf['rank'].mean()*100:.2f}% std: {rdf['rank'].std()*100:.2f}%",
                color='b',
                alpha=0.6
            )
            plt.plot(np.array(rdf["group"]), np.array(rdf["rank"]), 'o', color="0.4", alpha=0.3, markersize=3)
            plt.yticks(fontsize=self.fontsize)
            if rotation is not None:
                plt.xticks(fontsize=self.fontsize, rotation=rotation, ha='right')
            else:
                plt.xticks(fontsize=self.fontsize)
            plt.legend(loc="upper left", fontsize=self.fontsize)
            self.show()
        if show_score:
            plt.figure(figsize=self.size)
            plt.title(f"Expect@Top_{k}", fontsize=self.title_size)
            if show_range == True:
                xs = np.array(rdf['group'][plot_move_window - 1:])
                a_mean = bn.move_mean(rdf['mean'], plot_move_window)[plot_move_window - 1:]
                a_min = bn.move_mean(rdf['min'], plot_move_window)[plot_move_window - 1:]
                a_max = bn.move_mean(rdf['max'], plot_move_window)[plot_move_window - 1:]
                # a_std_p = bn.move_mean(rdf['mean'] + rdf['std'], plot_move_window)[plot_move_window - 1:]
                # a_std_m = bn.move_mean(rdf['mean'] - rdf['std'], plot_move_window)[plot_move_window - 1:]
                lw = 0.4

                plt.plot(xs, a_mean, label="mean", color='w', linewidth=lw)
                plt.fill_between(xs, y1=a_max, y2=a_min, color='0.6', alpha=0.3, label='min-max')
                # plt.fill_between(xs, y1=a_std_p, y2=a_std_m, color='0.4', alpha=0.3, label='std')
            plt.plot(
                np.array(rdf["group"])[plot_move_window-1:],
                bn.move_mean(rdf["actual"], plot_move_window)[plot_move_window-1:],
                label=f"score: mean: {rdf['actual'].mean():.4f} std: {rdf['actual'].std():.4f}",
                color='b',
                alpha=0.6
            )
            plt.plot(
                np.array(rdf["group"])[plot_move_window-1:],
                bn.move_mean(rdf["random"], plot_move_window)[plot_move_window-1:],
                label=f"random: mean: {rdf['random'].mean():.4f} std: {rdf['random'].std():.4f}",
                color='0.5',
                alpha=0.6
            )

            plt.plot(np.array(rdf["group"]), np.array(rdf["random"]), 'x', color='0.5', alpha=0.3, markersize=3, markeredgewidth=1)
            p_mask = np.array(rdf['actual'] <= rdf['random'])
            plt.plot(np.array(rdf["group"])[p_mask], np.array(rdf["actual"])[p_mask], 'o', color='g', alpha=0.3, markersize=3)
            plt.plot(np.array(rdf["group"])[~p_mask], np.array(rdf["actual"])[~p_mask], 'o', color='r', alpha=0.3, markersize=3)
            plt.yticks(fontsize=self.fontsize)
            if rotation is not None:
                plt.xticks(fontsize=self.fontsize, rotation=rotation, ha='right')
            else:
                plt.xticks(fontsize=self.fontsize)
            plt.legend(loc="upper left", fontsize=self.fontsize)
            self.show()
        return rdf

    def plot_trade_summary(self, indices, value, benchmark, log_scale=None, account_label="value", benchmark_label="benchmark", frequency='D', Rf=0.03, subplot=None, major_unit="month3", minor_unit="month"):
        '''
        展示净值曲线变化

        Args:

            indices: 时间轴
            value: 净值
            benchmark: 基准指数
            frequency: 时间粒度
            Rf: 年化无风险利率
            log_scale: 2 to use log2

        Returns:
            None
        '''
        if subplot is not None:
            plt.subplot(subplot)
        else:
            plt.figure(figsize=self.size)
        value = np.array(value)
        benchmark = np.array(benchmark)
        assert(len(value) > 0 and len(value) == len(benchmark) and value[0] > 0 and benchmark[0] > 0)
        value = value / value[0]
        benchmark = benchmark / benchmark[0]
        indices = Metrics.__try_convert_as_mdate(indices)
        # calc
        value_cummax = np.maximum.accumulate(value)
        max_drawdowns= (value_cummax - value)/value_cummax
        value_argcummax = qutils.acc_argmax(value)
        max_drawdown_end = max_drawdowns.argmax()
        max_drawdown_start = value_argcummax[max_drawdown_end]
        max_drawdown_len = max_drawdown_end - max_drawdown_start

        # plot
        plt.plot(indices, value, 'b', label=account_label, alpha=0.8)
        plt.plot(indices, benchmark, 'g', label=benchmark_label, alpha=0.8)
        plt.plot(indices, value / benchmark, 'orange', label="Alpha", alpha=0.2)
        plt.scatter(
            indices[[max_drawdown_start, max_drawdown_end]],
            value[[max_drawdown_start, max_drawdown_end]],
            facecolors='none', edgecolors='r',
            lw=2, alpha=0.5,
            label="Max Drawdown"
        )
        if log_scale is not None:
            plt.yscale("log", base_y=log_scale)
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y)/np.log10(log_scale), 0)))).format(y)))
        # plt.plot(indices, [0.0]*len(indices), '0.7', lw=2, alpha=0.5)
        plt.ylabel("Return", fontsize=self.fontsize)
        plt.legend(loc="upper left", fontsize=self.fontsize)
        indices_name, indices_value = list(zip(*self.risk_analysis(value, benchmark, frequency, Rf)))
        b_indices_name, b_indices_value = list(zip(*self.risk_analysis(benchmark, benchmark, frequency, Rf)))
        indices_value, b_indices_value = zip(*[
            ("%.2f%%" % (v*100), "%.2f%%" % (b*100)) if k not in {'IR', 'Beta', 'Sharp','Sortino', 'Calmar'}
                else ("N/A" if np.isnan(v) == True else "%.2f" % v, "N/A" if np.isnan(b) == True else "%.2f" % b)
            for k, v, b in zip(indices_name, indices_value, b_indices_value)
        ])
        b_kv = {k: v for k, v in zip(indices_name, b_indices_value)}
        b_kv['IR'] = '-'
        b_indices_value = [b_kv[k] for k in indices_name]
        cellColours = [['b' for _ in indices_value], ['g' for _ in b_indices_value]]
        the_table = plt.table(cellText=[indices_value, b_indices_value],
                  colWidths = [0.1]*len(indices_value),
                  cellColours=cellColours,
                #   rowColours=['b', 'g'],
                  colLabels=indices_name,
                #   rowLabels=[account_label, benchmark_label],
                  loc='lower right',
                  cellLoc='center',
                  bbox=[0, 1, 1, 0.15]
                  )
        the_table.set_alpha(0.5)
        the_table.set_fontsize(self.text_size)
        the_table.scale(1.5, 1.5)
        for c in the_table._cells:
            if c[0] > 0:
                the_table._cells[c]._text.set_color("w")
        # plt.ylim(ymin=0)
        plt.yticks(fontsize=self.fontsize)
        # self.__apply_xticks(indices=indices, num_ticks=20, rotate=True)
        self.apply_date_xticks(plt.gca(), major_unit=major_unit, minor_unit=minor_unit)
        plt.tight_layout(rect=(0, 0, 1, 0.87))
        if subplot is None:
            self.show()

    def plot_predict_summary(self, actural_label, predict_probability, beta=1, frac=1.0):
        '''
        plot predict summary

        Args:
            actual_label: real labels
            predict_probability: predicted probability like score by model

        Returns:
            None
        '''
        if frac < 1.0:
            mask = np.random.sample(len(actural_label)) < frac
            actural_label = actural_label[mask]
            predict_probability = predict_probability[mask]
        size = (self.width, self.height)
        plt.figure(figsize=size)
        self.plot_PR(actural_label, predict_probability, subplot=221)
        self.plot_roc_auc(actural_label, predict_probability, subplot=222)
        self.plot_precision_recall_fscore(actural_label, predict_probability, beta, subplot=212)
        plt.tight_layout()
        self.show()

    def plot_roc_auc(self, actural_label, predict_probability, subplot=None):
        '''
        plot ROC curve show AUC

        Args:
            actual_label: real labels
            predict_probability: predicted probability like score by model

        Returns:
            None
        '''

        if subplot is not None:
            plt.subplot(subplot)
        else:
            plt.figure(figsize=self.size)
        auc = roc_auc_score(actural_label, predict_probability)
        fpr, tpr, thresholds = roc_curve(actural_label, predict_probability)
        lw=2
        plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.fill_between(fpr, tpr, 0, color='g', alpha=0.2)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.title("ROC curve", fontsize=self.title_size)
        plt.xlabel("FPR", fontsize=self.fontsize)
        plt.ylabel("TPR", fontsize=self.fontsize)
        plt.legend(loc='lower right', fontsize=self.fontsize)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(self.xticks01, fontsize=self.fontsize)
        plt.yticks(self.yticks01, fontsize=self.fontsize)
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_ks_curve(self, actural_label, predict_probability, evenly_xlim=True, thresholds_density=False, subplot=None):
        '''
        plot KS curve show AUC

        Args:
            actual_label: real labels
            predict_probability: predicted probability like score by model

        Returns:
            None
        '''

        if subplot is not None:
            plt.subplot(subplot)
        else:
            plt.figure(figsize=self.size)
        fpr, tpr, thresholds = roc_curve(actural_label, predict_probability)
        x = np.arange(0,1,1/len(tpr))
        if len(x) != len(tpr):
            x = x[:len(tpr)]
        ks = max(abs(tpr-fpr))
        lw=2
        if evenly_xlim == True:
            plt.plot(x, tpr, color='blue',lw=lw, label='TPR')
            plt.plot(x, fpr, color='red' ,lw=lw, label='FPR')
            plt.plot(x, tpr-fpr, color='green',lw=lw,  label='KS = %0.3f'%ks)
        else:
            plt.plot(1-thresholds, tpr, color='blue',lw=lw, label='TPR')
            plt.plot(1-thresholds, fpr, color='red' ,lw=lw, label='FPR')
            plt.plot(1-thresholds, tpr-fpr, color='green',lw=lw,  label='KS = %0.3f'%ks)  
        plt.title("KS curve", fontsize=self.title_size)
        plt.xlabel("Thresholds", fontsize=self.fontsize)
        plt.ylabel("Rate", fontsize=self.fontsize)
        plt.legend(loc='upper left', fontsize=self.fontsize)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(self.xticks01, fontsize=self.fontsize)
        plt.yticks(self.yticks01, fontsize=self.fontsize)
        ax2 = plt.gca().twinx()
        # ax2.set_ylim(0, 10)
        ax2.set_xlim(0, 1)
        ax2.grid(False)
        if thresholds_density == True:
            sns.distplot(thresholds, bins=500, color='0.75', label="Density", hist=True, kde=False, norm_hist=True, ax=ax2)
        else:    
            sns.distplot(predict_probability, bins=500, color='0.75', label="Density", hist=True, kde=False, norm_hist=True, ax=ax2)
        ax2.set_ylabel("Density", fontsize=self.fontsize)
        ax2.tick_params(axis="y", labelsize=self.fontsize)
        ax2.legend(fontsize=self.fontsize) 
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_precision_recall_fscore(self, actural_label, predict_probability, beta=1, plot_ks=True, plot_pr=True, thresholds_density=False, subplot=None):
        '''
        plot precision recall f-score

        Args:
            actual_label: real label from test DataFrame
            predict_probability: predicted probability like score from model

        Returns:
            None
        '''
        beta_show = ('%f' % beta).rstrip('0').rstrip('.')
        if subplot is not None:
            plt.subplot(subplot)
        else:
            plt.figure(figsize=self.size)
        P, R, thresholds = pr(actural_label, predict_probability)
        fpr, tpr, rthresholds = roc_curve(actural_label, predict_probability)
        ks = max(abs(tpr - fpr))
        thresholds = np.pad(thresholds, (1, 0), mode="constant", constant_values=(0.0, 0.0))
        rthresholds = rthresholds[::-1]
        dv = np.where(P + R == 0.0, 1.0, beta * beta * P + R)
        Fscore = (1 + beta*beta) * P * R / dv

        if plot_pr == True:
            plt.plot(thresholds, P, label='Precision', color='g')
            plt.plot(thresholds, R, label='Recall', color='b')
            plt.plot(thresholds, Fscore, label=r'$F_{%s}$' % beta_show, color='r')

        if plot_ks == True:
            plt.plot(rthresholds, tpr, color='limegreen', label='TPR')
            plt.plot(rthresholds, fpr, color='orange', label='FPR')
            plt.plot(rthresholds, tpr - fpr, color='skyblue', label='KS = %0.3f' % ks)

        plt.title("Score On Threshold", fontsize=self.title_size)
        plt.xlabel("Threshold", fontsize=self.fontsize)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(self.xticks01, fontsize=self.fontsize)
        plt.yticks(self.yticks01, fontsize=self.fontsize)
        plt.legend(loc='upper left', fontsize=self.fontsize)
        ax2 = plt.gca().twinx()
        # ax2.set_ylim(0, 100)
        ax2.set_xlim(0, 1)
        ax2.grid(False)
        if thresholds_density == True:
            sns.distplot(rthresholds, bins=500, color='0.75', label="Density", hist=True, kde=False, norm_hist=True, ax=ax2)
        else:
            sns.distplot(predict_probability, bins=500, color='0.75', label="Density", hist=True, kde=False, norm_hist=True, ax=ax2)
        ax2.set_ylabel("Density", fontsize=self.fontsize)
        ax2.tick_params(axis="y", labelsize=self.fontsize)
        # hist, bins = np.histogram(predict_probability, bins=200, density=True, range=(0, 1))
        # width = np.diff(bins)
        # center = (bins[:-1] + bins[1:]) / 2
        # plt.bar(center, hist, align='center', width=width, facecolor='0.75', alpha=0.5, label="Density")
        # plt.hist(predict_probability, bins=256, normed=True, facecolor='0.75', alpha=0.5)
        ax2.legend(fontsize=self.fontsize)
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_PR(self, actural_label, predict_probability, beta=1, subplot=None):
        '''
        展示P,R的结果，并画出P-R图

        Args:
            actural_label: 真实结果
            predict_probability: 预测结果的概率
        '''
        beta_show = ('%f' % beta).rstrip('0').rstrip('.')
        if subplot is not None:
            plt.subplot(subplot)
        else:
            plt.figure(figsize=self.size)
        P, R, thresholds = pr(actural_label, predict_probability)
        dv = np.where(P + R == 0.0, 1.0, beta * beta * P + R)
        Fscore = (1 + beta*beta) * P * R / dv
        plt.plot(R, P, color='g', label='Precision')
        plt.plot(R, Fscore, color='r', label=r'$F_{%s}$' % beta_show)
        plt.title("PR curve", fontsize=self.title_size)
        plt.xlabel("Recall", fontsize=self.fontsize)
        plt.legend(loc='upper right', fontsize=self.fontsize)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(self.xticks01, fontsize=self.fontsize)
        plt.yticks(self.yticks01, fontsize=self.fontsize)
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_window_precision_recall_fscore(self, indices, actual_label, predict_score, threshold=0.5, window_size=22, beta=1, major_unit="month3", minor_unit="month"):
        '''
        展示基于窗口的指标，用于展示model的稳定性，目前包含preciion
        '''
        plt.figure(figsize=self.size)
        df = pd.DataFrame({
            "indices": indices,
            "actual_label": actual_label,
            "predict_score": predict_score
        }).sort_values("indices")
        df['predict_label'] = df['predict_score'] >= threshold
        df['TP'] = (df['actual_label'] == True) & (df['predict_label'] == True)
        df['FP'] = (df['actual_label'] == False) & (df['predict_label'] == True)
        df['TN'] = (df['actual_label'] == False) & (df['predict_label'] == False)
        df['FN'] = (df['actual_label'] == True) & (df['predict_label'] == False)

        groups = df.groupby("indices")
        unit_df = groups[['TP', 'FP', 'TN', 'FN']].sum()
        window_df = pd.DataFrame({
            "TP": bn.move_sum(unit_df['TP'], window_size),
            "FP": bn.move_sum(unit_df['FP'], window_size),
            "TN": bn.move_sum(unit_df['TN'], window_size),
            "FN": bn.move_sum(unit_df['FN'], window_size),
        })
        metric_df = pd.DataFrame({
            "p": window_df["TP"] / (window_df["TP"] + window_df["FP"]),
            "r": window_df["TP"] / (window_df["TP"] + window_df["FN"]),
        })

        beta2 = beta * beta
        metric_df['f'] = np.where(
            metric_df["p"] + metric_df["r"] > 0.0,
            (1 + beta2) * metric_df["p"] * metric_df["r"] / (beta2 * metric_df["p"] + metric_df["r"]),
            0.0
        )
        metric_df["date"] = unit_df.index.get_values()

        res_df = metric_df[window_size-1:]
        indices = Metrics.__try_convert_as_mdate(np.array(res_df['date']))
        plt.plot(indices, res_df['p'], label="precision")
        plt.plot(indices, res_df['r'], label="recall")
        plt.plot(indices, res_df['f'], label="$F_{beta}$".format(beta=beta))
        plt.ylim([0, 1])
        plt.legend(loc='upper left', fontsize=self.fontsize)
        plt.yticks(self.yticks01, fontsize=self.fontsize)
        # self.apply_xticks(indices=indices, num_ticks=8, rotate=True)
        self.apply_date_xticks(plt.gca(), major_unit=major_unit, minor_unit=minor_unit)
        plt.tight_layout()
        self.show()

    def plot_trade_log(self, trade_log, log_scale=None, window_size=1, bins=256, account_label="value", benchmark_label="benchmark", frequency='D', Rf=0.03, major_unit="month3", minor_unit="month"):
        """
        trade_log is the output from back_test, which is a dataframe
        """
        figsize=(self.size[0], self.size[1] * 2)
        plt.figure(figsize=figsize)
        trade_df, hold_df = trade_log
        self.plot_trade_summary(
            trade_df["index"],
            trade_df["position"] + trade_df["funds"],
            trade_df["benchmark"],
            log_scale=log_scale,
            account_label=account_label,
            benchmark_label=benchmark_label,
            frequency=frequency,
            Rf=Rf,
            subplot=211,
            major_unit=major_unit,
            minor_unit=minor_unit
        )
        if hold_df.shape[0] == 0:
            # exit when no hold
            return
        # plt.figure(figsize=self.size)
        plt.subplot(6, 1, 4)
        plt.title("Trade Summary", fontsize=self.title_size)
        indices = Metrics.__try_convert_as_mdate(trade_df["index"])
        plt.ylabel("Rate")
        plt.plot(
            indices,
            bn.move_mean(
                trade_df["position"]/(trade_df["position"] + trade_df["funds"]),
                window_size,
                min_count=1
            ),
            label="position rate",
            color="0.1",
            alpha=0.3
        )
        plt.ylim((0, 1.05))
        plt.yticks(fontsize=self.fontsize)
        plt.legend(loc="upper left", fontsize=self.fontsize)
        self.apply_date_xticks(plt.gca(), major_unit=major_unit, minor_unit=minor_unit)
        # self.ply_xticks(indices=indices, num_ticks=20, rotate=True)
        ax2 = plt.gca().twinx()
        ax2.grid(False)
        buy_count = bn.move_mean(
            trade_df["buy_count"].astype(np.float32),
            window_size,
            min_count=1
        )
        sell_count = bn.move_mean(
            trade_df["sell_count"].astype(np.float32),
            window_size,
            min_count=1
        )
        buy_index, buy_value = zip(*[
            (x, y)
            for bx, by in zip(indices, buy_count)
            for x, y in [(bx-0.4, 0), (bx-0.39, by), (bx-0.01, by), (bx, 0)]
        ])
        sell_index, sell_value = zip(*[
            (x, y)
            for bx, by in zip(indices, sell_count)
            for x, y in [
                (bx, 0),
                (bx+0.01, by),
                (bx+0.39, by),
                (bx+0.4, 0)
            ]
        ])
        plt.fill_between(buy_index, 0, buy_value, label="buy", color="r", alpha=0.6)
        plt.fill_between(sell_index, 0, sell_value, label="sell", color="g", alpha=0.6)
        # self.apply_xticks(indices=indices, num_ticks=20, rotate=True)
        self.apply_date_xticks(plt.gca(), major_unit=major_unit, minor_unit=minor_unit)
        max_cnt_2 = max(trade_df["buy_count"].max(), trade_df["sell_count"].max()) * 2
        ax2.set_ylim((0, max_cnt_2))
        ax2.legend(loc="upper right", fontsize=self.fontsize)
        ax2.tick_params(axis='y', labelsize=self.fontsize)
        plt.subplot(6, 2, 9)
        sns.distplot(np.array(hold_df["period"]), bins=bins, label=f'period = {round(hold_df["period"].mean(), 2)}')
        plt.yticks(fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.legend(loc="upper right", fontsize=self.fontsize)
        plt.subplot(6, 2, 10)
        rate = hold_df["close"] / hold_df["open"] - 1.0
        sns.distplot(rate, bins=bins, label=f'rate = {round(rate.mean()*100, 2)}%')
        plt.yticks(fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.legend(loc="upper right", fontsize=self.fontsize)
        plt.subplot(6, 2, 11)
        sns.distplot(np.array(hold_df["max_drawdown"]), bins=bins, label=f'max_drawdown = {round(hold_df["max_drawdown"].mean()*100, 2)}%')
        plt.yticks(fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.legend(loc="upper right", fontsize=self.fontsize)
        plt.subplot(6, 2, 12)
        unit_rate = np.power(1.0 + rate, 1.0 / hold_df["period"]) - 1.0
        mean_unit_rate = np.power(10, np.log10(1.0 + rate).sum() / hold_df["period"].sum()) - 1.0
        sns.distplot(unit_rate, bins=bins, label=f'unit_rate = {round(mean_unit_rate * 100, 2)}%')
        plt.yticks(fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.legend(loc="upper right", fontsize=self.fontsize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.925, hspace=0.3)
        self.show()

    def plot_ohlc_summary(self, title, times, opens, highs, lows, closes, volumes=None, extra_plots=[], subplot_height=2, major_unit="month3", minor_unit="month"):
        tm = self.__try_convert_as_mdate(times)
        def volume_plot(vax, times):
            vax.grid(False)
            vax.set_ylabel("vol", fontsize=self.fontsize)
            bcloses = np.roll(closes, 1)
            up_idx = closes >= bcloses
            up_idx[0] = False
            down_idx = closes < bcloses
            down_idx[0] = False
            def fill_idx(xs, ys):
                return zip(*[
                    (x, y)
                    for bx, by in zip(xs, ys)
                    for x, y in [(bx-0.5, 0), (bx-0.49, by), (bx+0.49, by), (bx+0.5, 0)]
                ])
            up_t, up_v = fill_idx(times[up_idx], volumes[up_idx])
            down_t, down_v = fill_idx(times[down_idx], volumes[down_idx])
            s_t, s_v = fill_idx(times[:1], volumes[:1])
            vax.fill_between(up_t, 0, up_v, color="r", alpha=0.6)
            vax.fill_between(down_t, 0, down_v, color="g", alpha=0.6)
            vax.fill_between(s_t, 0, s_v, color="w", alpha=0.6)
            self.apply_date_xticks(ax, major_unit=major_unit, minor_unit=minor_unit)

        df = pd.DataFrame(collections.OrderedDict([
            ("time", tm),
            ("open", opens),
            ("high", highs),
            ("low", lows),
            ("close", closes),
            ("volume", volumes if volumes is not None else 0.0)
        ]))
        quotes = df[["time", "open", "high", "low", "close", "volume"]].values

        extra_real_plots = []
        if volumes is not None:
            extra_real_plots.append(volume_plot)
        extra_real_plots.extend(extra_plots)
        subplot_count = len(extra_real_plots) + 1
        ratios = [3] + [1] * len(extra_real_plots)
        fig, subplots = plt.subplots(subplot_count, sharex=True, figsize=(self.size[0], subplot_height*sum(ratios)), gridspec_kw={'height_ratios': ratios})
        ax = subplots[0] if subplot_count > 1 else subplots
        ax.set_ylabel("price", fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.title_size)
        ax.set_facecolor('0.95')
        ax.grid(False)
        ax.grid(axis='y', color='0.7', alpha=0.2)
        ax.xaxis_date()
        ax.autoscale_view()
        ax.tick_params(axis='both', labelsize=self.fontsize)
        _candlestick(ax, quotes, width=0.9, colordown='g', colorup='r', alpha=0.7)
        self.apply_date_xticks(ax, major_unit=major_unit, minor_unit=minor_unit)

        if subplot_count > 1:
            for e_ax, e_p in zip(subplots[1:], extra_real_plots):
                e_ax.set_facecolor('0.95')
                e_ax.tick_params(axis='both', labelsize=self.fontsize)
                e_p(e_ax, tm)

        plt.subplots_adjust(wspace=0, hspace=0.01)
        self.show()
