import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from QPhantom.core.metrics import Metrics
from sklearn import svm
import profile

df_data  = pd.read_pickle('/home/mercy/A_stocks/行情数据2007_2017/features_pctchg_nona.pkl')
hz_index = pd.read_csv('/home/mercy/A_stocks/行情数据2007_2017/HSZZ_index.csv',index_col='tradeDate')
df_data = df_data[df_data['pctchg21']<=2]
cond_hl = df_data['highestPrice']!=df_data['lowestPrice']
df_data = df_data.loc[cond_hl,:]

date_list  = sorted(list(set(df_data['tradeDate'])))
secID_list = sorted(list(set(df_data['secID'])))
df_test = df_data.set_index('tradeDate',inplace=False)
df_data.set_index('secID', inplace=True)
stk_nums = 20
interval = 22
division_rat = 0.1
test_start_date = '2012-01-04'
capital = 10**7
commission = 0.001
impact_cost = 0.002
account_return = []
benchmark = "ZZ500" #"HS300" #
capital_each = capital/stk_nums
account = np.ones(stk_nums)*capital_each
testdate_list  = date_list[date_list.index(test_start_date):]
change_stk_date  = date_list[date_list.index(test_start_date)::interval]
traindate_list = date_list[:date_list.index(test_start_date)]
data_lb_list = []
df_buylist = pd.DataFrame()
M = Metrics()

for day in traindate_list:
    data_oneday = df_test.loc[day,:].drop('secID')
    n = int(data_oneday.shape[0]*division_rat)
    #n = 20
    phg_high = data_oneday.sort_values('pctchg21',ascending=False).head(n)
    phg_low  = data_oneday.sort_values('pctchg21',ascending=False).tail(n)
    phg_high['pctchg21'] =  1
    phg_low['pctchg21']  = -1
    data_lb_oneday = pd.concat([phg_high,phg_low])
    data_lb_list.append(data_lb_oneday)
    if day == traindate_list[-1]:
        data_lb = pd.concat(data_lb_list)
        del data_lb['secID']
        data_lb.dropna(inplace=True)
        data_lb.index = np.random.permutation(data_lb.shape[0])
        data_lb.sort_index(inplace=True)
        train_y = data_lb['pctchg21']
        train_x = data_lb.drop('pctchg21',axis=1)
        clf = xgb.XGBClassifier(max_depth=3).fit(train_x, train_y)
        # clf = LogisticRegression(fit_intercept=True).fit(train_x, train_y)

for day in change_stk_date:
    testing = df_test.loc[day,:].set_index('secID').dropna()
    del testing['pctchg21']
    prediction = clf.predict_proba(testing)[:,1]
    buylist = list(testing.iloc[prediction.argsort()[-stk_nums:]].index)
    # predictS = pd.DataFrame(clf.predict_proba(testing)[:,1],columns=['predict'],index=testing.index)
    # buylist = predictS.sort_values('predict').tail(stk_nums).index.tolist()
    stk_each = pd.DataFrame(buylist,columns = [day])
    df_buylist = df_buylist.merge(stk_each, how='outer', left_index=True, right_index=True)

df_buylist.to_csv('/home/mercy/A_stocks/df_buylist1.csv',index=False)
df_buylist = pd.read_csv('/home/mercy/A_stocks/df_buylist1.csv')

df_pct = df_data.set_index('tradeDate',append=True)[['pctchg1']]
def test(testdate_list,change_stk_date,df_buylist,account,commission,impact_cost):
    for i,day in enumerate(testdate_list):
        if i < len(testdate_list):
            print(day)
        else:
            break
        if day not in change_stk_date:
            capital_current = sum(account)
            account_return.append(capital_current/capital)
        elif day in change_stk_date:
            tobuylist = df_buylist[day]
            capital_current = sum(account) * (1 - commission - impact_cost)
            account_return.append(capital_current/capital)
            capital_each = capital_current/stk_nums
            account = np.ones(stk_nums)*capital_each
        for s,stk in enumerate(tobuylist):
            try:
                Rr = df_pct.xs((stk,day)).values[0]
            except Exception as err:
                Rr = -0.02
                print(err)
                print(stk)
            account[s] = account[s]*(Rr+1)
    return account_return
# profile.run('test(testdate_list,change_stk_date,df_buylist,account,commission,impact_cost)')
# %load_ext line_profiler
# %lprun -f test account_return = test(testdate_list,change_stk_date,df_buylist,account,commission,impact_cost)

%time account_return  = test(testdate_list,change_stk_date,df_buylist,account,commission,impact_cost)
sock_index = hz_index.loc[testdate_list, benchmark]
M.plot_trade_summary(testdate_list,account_return,sock_index,account_label="Account", benchmark_label=benchmark,frequency='D')
plt.show()

#%%

i = 0
years = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
for d in date_list:
    if d > '{year}-01-01'.format(year=years[i]):
        print(d)
        i += 1
        if i>10:break
