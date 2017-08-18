#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:52:06 2017

@author: mercy
"""

import io
import math
import numpy as np
import pandas as pd
import seaborn as sns
import collections
import functools
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve as pr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc
from contextlib import contextmanager
import matplotlib.dates as mdates
from datetime import datetime
import bottleneck as bn

import QPhantom.core.utils as qutils

sns.set(color_codes=True)

# plt.rcParams['font.sans-serif'] = ["SimHei", "WenQuanYi Micro Hei"] + plt.rcParams['font.sans-serif'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号



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
        返回sharp,IR,Alpha,Beta等重要指标
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
        ab_cov = ab_cov.ix['account_return','bench_return']
        a_std = account['account_return'].std()
        b_std = account['bench_return'].std()
        Bench_return = (bn/b0)**(freq/n)-1
        Annual_return = (an/a0)**(freq/n)-1
        Volatility = np.nanstd(account['account_return'])*np.sqrt(freq)
        Sharp = (account['account_return'].mean() - ((1 + Rf)**(1/freq) - 1))/np.nanstd(account['account_return']) * np.sqrt(freq)
        ab_mean, ab_std = np.nanmean(account['a-b']), np.nanstd(account['a-b'])
        IR = np.nanmean(account['a-b']) / np.nanstd(account['a-b']) * np.sqrt(freq) if ab_mean != 0 else 0
        Beta = ab_cov/(b_std*b_std)
        Alpha= Annual_return-Rf-Beta*(Bench_return-Rf)
        Max_drawdown = max_drawdowns.max()

        return [
            ("Return", Annual_return),
            ("Alpha", Alpha),
            ("Beta", Beta),
            ("Sharp", Sharp),
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
            values = mdates.date2num(values)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        except Exception as e:
            print("convert_datetime_fail")
            print(e)
            #no convert need
            pass
        return values

    @staticmethod
    def __apply_sticks(indices, num_sticks=20, rotate=True):
        step = len(indices) // num_sticks
        xticks = list(indices[0:-step:step]) + [indices[-1]]
        plt.xlim([indices[0], indices[-1]])
        if rotate is True:
            plt.xticks(xticks, rotation=30, ha='right')
        else:
            plt.xticks(xticks)

    def score_on_topk(self, actual, predict, groups=[0], k=50, ascending=False, show_rank=True, show_score=False, plot_move_window=1):
        """
        return average score and rank on topk in each group

        return DataFrame("group", "actual", "rank")
        """
        size = len(actual)
        indices = np.arange(size)
        np.random.shuffle(indices)
        df = pd.DataFrame(collections.OrderedDict([
            ("group", groups[indices]),
            ("actual", actual[indices]),
            ("predict", predict[indices])
        ])).sort_values(["group", "predict"], ascending=[True, ascending]).reset_index(drop=True)
        gsize = df.groupby("group")["predict"].count()
        df["rank"] = df.groupby("group")["actual"].rank(ascending=ascending, pct=True)
        rdf = df.groupby("group").head(k).groupby("group")[["actual", "rank"]].mean()
        rdf.reset_index(inplace=True)
        if show_rank:
            plt.figure(figsize=self.size)
            plt.title("Rank")
            plt.plot(
                np.array(rdf["group"]),
                bn.move_mean(rdf["rank"], plot_move_window, min_count=1),
                label=f"mean:{rdf['rank'].mean()*100:.2f}% std: {rdf['rank'].std()*100:.2f}%"
            )
            plt.plot(np.array(rdf["group"]), np.array(rdf["rank"]), 'o', color="0.4", alpha=0.3, markersize=3)
            plt.legend(loc="upper left", fontsize=self.fontsize)
            self.show()
        if show_score:
            plt.figure(figsize=self.size)
            plt.title("Actual Score")
            plt.plot(
                np.array(rdf["group"]),
                bn.move_mean(rdf["actual"], plot_move_window, min_count=1),
                label=f"score: mean: {rdf['actual'].mean():.4f} std: {rdf['actual'].std():.4f}"
            )
            plt.plot(np.array(rdf["group"]), np.array(rdf["actual"]), 'o', color='0.4', alpha=0.3, markersize=3)
            plt.legend(loc="upper left", fontsize=self.fontsize)
            self.show()
        return rdf

    def plot_trade_summary(self, indices, value, benchmark, log_scale=False, account_label="value", benchmark_label="benchmark", frequency='D', Rf=0.03, subplot=None):
        '''
        展示净值曲线变化

        Args:

            indices: 时间轴
            value: 净值
            benchmark: 基准指数
            frequency: 时间粒度
            Rf: 年化无风险利率

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
        if log_scale is True:
            plt.yscale("log", base_y=10)
            plt.plot(indices, value, 'b', label=account_label, alpha=0.8)
            plt.plot(indices, benchmark, 'g', label=benchmark_label, alpha=0.8)
            plt.plot(indices, value / benchmark, 'orange', label="Alpha", alpha=0.2)
        else:
            plt.plot(indices, value, 'b', label=account_label, alpha=0.8)
            plt.plot(indices, benchmark, 'g', label=benchmark_label, alpha=0.8)
            plt.plot(indices, value - benchmark, 'orange', label="Alpha", alpha=0.2)
        plt.scatter(
            indices[[max_drawdown_start, max_drawdown_end]],
            value[[max_drawdown_start, max_drawdown_end]],
            facecolors='none', edgecolors='r',
            lw=2, alpha=0.5,
            label="Max Drawdown"
        )
        plt.plot(indices, [0.0]*len(indices), '0.7', lw=2, alpha=0.5)
        plt.ylabel("Return", fontsize=self.fontsize)
        plt.legend(loc="upper left", fontsize=self.fontsize)
        indices_name, indices_value = list(zip(*self.risk_analysis(value, benchmark, frequency, Rf)))
        b_indices_name, b_indices_value = list(zip(*self.risk_analysis(benchmark, benchmark, frequency, Rf)))
        indices_value, b_indices_value = zip(*[
            ("%.2f%%" % (v*100), "%.2f%%" % (b*100)) if k not in {'IR', 'Beta', 'Sharp'}
                else ("%.2f" % v, "%.2f" % b)
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
        Metrics.__apply_sticks(indices=indices, num_sticks=20, rotate=True)
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
        plt.xticks(self.xticks01)
        plt.yticks(self.yticks01)
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_precision_recall_fscore(self, actural_label, predict_probability, beta=1, subplot=None):
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
        thresholds = np.pad(thresholds, (1, 0), mode="constant", constant_values=(0.0, 0.0))
        dv = np.where(P + R == 0.0, 1.0, beta * beta * P + R)
        Fscore = (1 + beta*beta) * P * R / dv

        plt.plot(thresholds, P, label='Precision', color='g')
        plt.plot(thresholds, R, label='Recall', color='b')
        plt.plot(thresholds, Fscore, label=r'$F_{%s}$' % beta_show, color='r')

        plt.title(r'Precision / Recall / $F_{%s}$' % beta_show, fontsize=self.title_size)
        plt.xlabel("Threshold", fontsize=self.fontsize)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(self.xticks01)
        plt.yticks(self.yticks01)
        plt.legend(loc='upper left', fontsize=self.fontsize)
        ax2 = plt.gca().twinx()
        bins = 200
        # ax2.set_ylim(0, 100)
        ax2.set_xlim(0, 1)
        ax2.grid(False)
        sns.distplot(predict_probability, bins=500, color='0.75', label="Density", hist=True, kde=False, norm_hist=True, ax=ax2)
        ax2.set_ylabel("Density", fontsize=self.fontsize)
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
        plt.xticks(self.xticks01)
        plt.yticks(self.yticks01)
        if subplot is None:
            plt.tight_layout()
            self.show()

    def plot_window_precision_recall_fscore(self, indices, actual_label, predict_score, threshold=0.5, window_size=22, beta=1):
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
        plt.xticks(self.xticks01)
        plt.yticks(self.yticks01)
        plt.legend(loc='upper left', fontsize=self.fontsize)
        Metrics.__apply_sticks(indices=indices, num_sticks=8, rotate=True)
        plt.tight_layout()
        self.show()

    def plot_trade_log(self, trade_log, log_scale=False, window_size=1, bins=256, account_label="value", benchmark_label="benchmark", frequency='D', Rf=0.03):
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
            subplot=211
        )
        # plt.figure(figsize=self.size)
        plt.subplot(6, 1, 4)
        plt.title("Trade Summary")
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
        plt.legend(loc="upper left", fontsize=self.fontsize)
        Metrics.__apply_sticks(indices=indices, num_sticks=20, rotate=True)
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
        Metrics.__apply_sticks(indices=indices, num_sticks=20, rotate=True)
        max_cnt_2 = max(trade_df["buy_count"].max(), trade_df["sell_count"].max()) * 2
        ax2.set_ylim((0, max_cnt_2))
        ax2.legend(loc="upper right", fontsize=self.fontsize)
        plt.subplot(6, 2, 9)
        sns.distplot(np.array(hold_df["period"]), bins=bins, label=f'period = {round(hold_df["period"].mean(), 2)}')
        plt.legend(loc="upper right")
        plt.subplot(6, 2, 10)
        rate = hold_df["close"] / hold_df["open"] - 1.0
        sns.distplot(rate, bins=bins, label=f'rate = {round(rate.mean()*100, 2)}%')
        plt.legend(loc="upper right")
        plt.subplot(6, 2, 11)
        sns.distplot(np.array(hold_df["max_drawdown"]), bins=bins, label=f'max_drawdown = {round(hold_df["max_drawdown"].mean()*100, 2)}%')
        plt.legend(loc="upper right")
        plt.subplot(6, 2, 12)
        unit_rate = np.power(1.0 + rate, 1.0 / hold_df["period"]) - 1.0
        mean_unit_rate = np.power(10, np.log10(1.0 + rate).sum() / hold_df["period"].sum()) - 1.0
        sns.distplot(unit_rate, bins=bins, label=f'unit_rate = {round(mean_unit_rate * 100, 2)}%')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.subplots_adjust(top=0.925, hspace=0.3)
        self.show()
