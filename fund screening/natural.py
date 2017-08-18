import re
import numpy as np
import pandas as pd
import json as js
import MySQLdb
from flask import Flask
from flask import request
from flask import jsonify
from datetime import datetime, timedelta
from sqlalchemy import create_engine
db_buffer = create_engine("mysql://data:showmethemoney@192.168.215.201/cn_stock_data?charset=utf8mb4", pool_size=64, pool_recycle=400, max_overflow=0)
app = Flask(__name__)

# line = "过去7天涨幅超过2.7%的前10只股票"
# line = "过去5天收盘价大于100的前6只股票"

def get_keyword(line):
    r_time      = r'(?P<时间>((?P<汉语天数>[今昨一二两俩三仨四五六七八九十百千]*)|(?P<数字天数>[1-9999]*))个*?(?P<时间单位>天|日|周|星期|月))'
    r_direction = r'(?P<方向>收益率|涨|跌)'
    r_how       = r'(?P<幅度>大|超|小)'
    r_target    = r'(?P<目标>开盘价|收盘价|最高价|最低价|成交量|成交额|换手率|总市值|动态市盈率|静态市盈率|市净率)'
    r_order     = r'(?P<排序>前|后)'
    r_limit     = r'(?P<LIMIT>[0-9]+\.?[0-9]*)'
    r_per       = r'(?P<PER>百分之|千分之|万分之|\%)'

    res = list(re.finditer('|'.join([r_time,r_direction,r_how, r_target, r_order, r_limit, r_per]),line,re.M|re.I))
    print('|'.join([r_time,r_direction,r_how, r_target, r_order, r_limit, r_per]))
    name_list = []
    vale_list = []
    for i,k in enumerate(res):
        for n in ['时间','方向','幅度','目标','排序','LIMIT','PER']:
            if repr(k.group(n)) != 'None':
                name_list.append(n)
                vale_list.append(k.group(n))
                if n=='LIMIT' and repr(k.group(n))!= 'None':
                    vale_list[i] = float(k.group(n))
                if n=='PER' and repr(k.group(n)) != 'None':
                    vale_list[i-1] = vale_list[i-1] / 100
        if (i==len(res)-1) and ('目标' not in name_list):
            name_list.append('目标')
            vale_list.append('wd_close')
        if (i==len(res)-1) and '时间' not in name_list:
            name_list.append('时间')
            vale_list.append('今天')
        if (i==len(res)-1) and '方向' not in name_list:
            name_list.append('方向')
            vale_list.append('涨')

    name_vale = pd.Series(vale_list,index=name_list)

    if type(name_vale['LIMIT']) == np.float:
        dig_limit = name_vale['LIMIT']
        rank_limit = name_vale['LIMIT']
    else:
        dig_limit = name_vale['LIMIT'].values[0]
        rank_limit = int(name_vale['LIMIT'].values[1])

    chinese_name = ['开盘价','收盘价','最高价','最低价','成交量','成交额','换手率','总市值','动态市盈率','静态市盈率','市净率']
    english_name = ['wd_open','wd_close','wd_high','wd_low','wd_vol','wd_amt','wd_turn','wd_ev','wd_pe_ttm','wd_pe_lyr','wd_pb']
    for i,name in enumerate(chinese_name):
        if name == name_vale['目标']:
            name_vale['目标'] = english_name[i]
            break

    if name_vale['方向'] in ['涨','收益率'] and name_vale['幅度'] in ['大','超']:
        direction = '>'
    elif name_vale['方向'] in ['涨','收益率'] and name_vale['幅度'] in ['小']:
        direction = '<'
    elif name_vale['方向'] in ['跌','收益率'] and name_vale['幅度'] in ['大','超']:
        direction = '<'
        dig_limit = -dig_limit
    elif name_vale['方向'] in ['跌','收益率'] and name_vale['幅度'] in ['小']:
        direction = '>'
        dig_limit = -dig_limit

    return res,name_vale,dig_limit,rank_limit,direction

def chinese_to_dig(res):
    chinese_dig = {'今':1,'昨':2,'一':1, '二':2,'两':2,'俩':2, '三':3, '四':4, '五':5, '六':6, '七':7,'八':8, '九':9,'十':10}
    if repr(res[0].group('时间')) != 'None':
        if repr(res[0].group('汉语天数')) != 'None':
            if len(res[0].group('汉语天数')) == 1:
                n_days = chinese_dig[res[0].group('汉语天数')]
            elif len(res[0].group('汉语天数')) == 2:
                n_days = 10 + chinese_dig[res[0].group('汉语天数')[-1]]
            elif len(res[0].group('汉语天数')) == 3:
                n_days = 10*chinese_dig[res[0].group('汉语天数')[0]] + chinese_dig[res[0].group('汉语天数')[-1]]
            if res[0].group('时间单位') in ['周','星期']:
                n_days = int(n_days)*5
            elif res[0].group('时间单位') == '月':
                n_days = int(n_days)*30
        elif repr(res[0].group('汉语天数')) == 'None':
            if res[0].group('时间单位') in ['日','天']:
                n_days = int(res[0].group('数字天数'))
            elif res[0].group('时间单位') in ['周','星期']:
                n_days = int(res[0].group('数字天数'))*5
            elif res[0].group('时间单位') == '月':
                n_days = int(res[0].group('数字天数'))*30
    elif repr(res[0].group('时间')) == 'None':
        n_days = 1
    return n_days

def condition_mapping(n_days,name_vale,direction,dig_limit,rank_limit):
    constraints_mapping = {
        "last":      lambda : f"trade_date>{(datetime.now() - timedelta(days=n_days)).strftime('%Y%m%d')}",
        "wd_pe_ttm": lambda : f"wd_pe_ttm {direction} {dig_limit}",
        "wd_close":  lambda : f"wd_close {direction} {dig_limit}",
        "wd_open":   lambda : f"wd_open {direction} {dig_limit}",
        "wd_high":   lambda : f"wd_high {direction} {dig_limit}",
        "wd_low":    lambda : f"wd_low {direction} {dig_limit}",
        "wd_vol":    lambda : f"wd_vol {direction} {dig_limit}",
        "wd_amt":    lambda : f"wd_amt {direction} {dig_limit}",
        "wd_turn":   lambda : f"wd_turn {direction} {dig_limit}",
        "wd_ev":     lambda : f"wd_ev {direction} {dig_limit}",
        "wd_pe_ttm": lambda : f"wd_pe_ttm {direction} {dig_limit}",
        "wd_pe_lyr": lambda : f"wd_pe_lyr {direction} {dig_limit}",
        "wd_pb":     lambda : f"wd_pb {direction} {dig_limit}",
    }

    constraints_date = constraints_mapping['last']()
    constraints_pe = constraints_mapping['wd_pe_ttm']()
    constraints_close = constraints_mapping['wd_close']()

    if 'PER' in name_vale.keys():
        constraints_target = True
    else:
        constraints_target = constraints_mapping[f"{name_vale['目标']}"]()

    with db_buffer.connect() as conn:
        if 'PER' in name_vale.keys():
            data = pd.read_sql_query(f"select unique_code,trade_date, {name_vale['目标']},wd_div_x*wd_close as weight_close from history where {constraints_date} and {constraints_target}", conn, chunksize=None)
            if data.shape[0] != 0:
                stk_list = []
                rat_list = []
                data_gb = data.groupby('unique_code')
                for stk in data_gb.groups:
                    stk_list.append(stk)
                    df = data_gb.get_group(stk)
                    rat_list.append(df.ix[df.index[-1],'weight_close'] / df.ix[df.index[0],'weight_close'] - 1)
                if dig_limit > 0:
                    inter_list = list(set(stk_list).intersection(set(['.DJI.N','.INX.A','.IXIC.O'])))
                    rank_ser = pd.DataFrame(rat_list,index=stk_list,columns=['change_rate']).sort_values(by='change_rate',ascending=False).drop(inter_list)
                    if '排序' not in name_vale.keys():
                        if name_vale['幅度'] in ['大','超']:
                            rank_ser = rank_ser[rank_ser['change_rate'] > dig_limit][:20]
                        elif name_vale['幅度'] in ['小']:
                            rank_ser = rank_ser[rank_ser['change_rate'] < dig_limit][:20]
                    elif name_vale['排序'] == '前':
                        if name_vale['幅度'] in ['大','超']:
                            rank_ser = rank_ser[rank_ser['change_rate'] > dig_limit][:rank_limit]
                        elif name_vale['幅度'] in ['小']:
                            rank_ser = rank_ser[rank_ser['change_rate'] < dig_limit][:rank_limit]
                    elif name_vale['排序'] == '后':
                        if name_vale['幅度'] in ['大','超']:
                            rank_ser = rank_ser[rank_ser['change_rate'] > dig_limit][-rank_limit:]
                        elif name_vale['幅度'] in ['小']:
                            rank_ser = rank_ser[rank_ser['change_rate'] < dig_limit][-rank_limit:]
                elif dig_limit < 0:
                    inter_list = list(set(stk_list).intersection(set(['.DJI.N','.INX.A','.IXIC.O'])))
                    rank_ser = pd.DataFrame(rat_list,index=stk_list,columns=['change_rate']).sort_values(by='change_rate',ascending=True).drop(inter_list)
                    if name_vale['排序'] == '前':
                        if name_vale['幅度'] in ['大','超']:
                            rank_ser = rank_ser[rank_ser['change_rate'] < dig_limit][:rank_limit]
                        elif name_vale['幅度'] in ['小']:
                            rank_ser = rank_ser[rank_ser['change_rate'] > dig_limit][:rank_limit]
                    elif name_vale['排序'] == '后':
                        if name_vale['幅度'] in ['大','超']:
                            rank_ser = rank_ser[rank_ser['change_rate'] < dig_limit][-rank_limit:]
                        elif name_vale['幅度'] in ['小']:
                            rank_ser = rank_ser[rank_ser['change_rate'] > dig_limit][-rank_limit:]
            elif data.shape[0] == 0:
                rank_ser = ['The data have not download!']
            return rank_ser

        else:
            data = pd.read_sql_query(f"select unique_code,trade_date,wd_div_x, {name_vale['目标']} from history where {constraints_date} and {constraints_target}", conn, chunksize=None)
            if data.shape[0] != 0:
                data_gb = data.groupby('unique_code')
                stk_list = []
                val_list = []
                for stk in data_gb.groups:
                    stk_list.append(stk)
                    df = data_gb.get_group(stk).sort_values(by=name_vale['目标'],ascending=False)
                    val_list.append(df.ix[df.index[0],name_vale['目标']])
                inter_list = list(set(stk_list).intersection(set(['.DJI.N','.INX.A','.IXIC.O'])))
                rank_ser = pd.DataFrame(val_list,index=stk_list,columns=[name_vale['目标']]).sort_values(by=name_vale['目标'],ascending=False).drop(inter_list)
                if name_vale['排序'] == '前':
                    if name_vale['幅度'] in ['大','超']:
                        rank_ser = rank_ser[rank_ser[name_vale['目标']] > dig_limit][:rank_limit]
                    elif name_vale['幅度'] in ['小']:
                        rank_ser = rank_ser[rank_ser[name_vale['目标']] < dig_limit][:rank_limit]
                elif name_vale['排序'] == '后':
                    if name_vale['幅度'] in ['大','超']:
                        rank_ser = rank_ser[rank_ser[name_vale['目标']] > dig_limit][-rank_limit:]
                    elif name_vale['幅度'] in ['小']:
                        rank_ser = rank_ser[rank_ser[name_vale['目标']] < dig_limit][-rank_limit:]
            elif data.shape[0] == 0:
                rank_ser = ['There is no data yet,please try again after 5:00 p.m.']
            return rank_ser


@app.route('/', methods=["POST"])
def requirement():
    line = request.json["query"]
    res,name_vale,dig_limit,rank_limit,direction = get_keyword(line)
    print(name_vale)
    n_days = chinese_to_dig(res)
    rank_df = condition_mapping(n_days,name_vale,direction,dig_limit,rank_limit)
    if type(rank_df) == list:
        dt = {"sorry":"There is no data yet,please try again after 5:00 p.m."}
        return js.dumps(dt,skipkeys=True,ensure_ascii=False)
    else:
        return rank_df.to_json()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5877)


#%%
