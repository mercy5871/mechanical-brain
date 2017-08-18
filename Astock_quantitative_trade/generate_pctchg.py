import numpy as np
import pandas as pd
import bottleneck as bn

n_day = 1
n_month = 21
df = pd.read_pickle('/home/mercy/A_stocks/行情数据2007_2017/features_2007_20170630.pkl')
df['AdjClosePrice'] = df['closePrice']*df['accumAdjFactor']
cond = df['turnoverVol'] != 0
df = df.loc[cond,:]

date_list  = sorted(list(set(df['tradeDate'])))
secID_list = sorted(list(set(df['secID'])))
ld = []
lm = []
cnt = 1
for s in secID_list:
    onestk = df[df['secID']==s]
    chg_day = onestk[['AdjClosePrice']].pct_change(periods=n_day).shift(-n_day)
    chg_month = onestk[['AdjClosePrice']].pct_change(periods=n_month).shift(-n_month)
    ld.append(chg_day)
    lm.append(chg_month)
    cnt += 1
    if cnt%10 == 0:
        print(cnt)
    if cnt == len(secID_list):
        print('lm done!')
df_dchg = pd.concat(ld)
df_dchg.columns = ['pctchg1']
df_mchg = pd.concat(lm)
df_mchg.columns = ['pctchg21']
print('chg done!')

df_day_chg = df_dchg.sort_index(inplace=False)
df_month_chg = df_mchg.sort_index(inplace=False)
print('df_sort done!')
df['pctchg1'] = df_day_chg
df['pctchg21'] = df_month_chg

ss = df['pctchg21'].fillna(888)
cond = ss != 888
df_nona = df.loc[cond,:]
print('df_nona done!')

df_nona = df_nona[(df_nona['pctchg1']<=0.1)]
df_nona.to_pickle('/home/mercy/A_stocks/行情数据2007_2017/features_pctchg_nona.pkl')
print('all done!')
#%%


df21 = df_n.sort_values('pctchg21',ascending=False)
df1 =  df_n.sort_values('pctchg1',ascending=False)
df_nona.shape
df21[['secID','tradeDate','closePrice','AdjClosePrice','accumAdjFactor','pctchg1','pctchg21']][:10]

df_n = df_nona[(df_nona['pctchg1']<=0.1)&(df_nona['pctchg21']<=3)]
df_n[(df_n.secID=='600372.XSHG')&(df_n.tradeDate>='2009-03-10')][['secID','tradeDate','closePrice','AdjClosePrice','accumAdjFactor','pctchg1','pctchg21']][:30]
df_n.shape

#%%
import numpy as np
import pandas as pd
import mysqlclient
from sqlalchemy import create_engine
#import MySQLdb

DB_CONNECT_STRING ='mysql://root:feng1300@localhost/pp_try?charset=utf8mb4' 
engine = create_engine(DB_CONNECT_STRING, pool_size=64, pool_recycle=400, max_overflow=0)
df = pd.DataFrame(np.ones([3,2]),columns=['a','b'])
pd.io.sql.to_sql(df,'fund_rating', engine,index=False, schema='w_fund', if_exists='append',chunksize=10000)
