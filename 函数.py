# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:52:16 2016

@author: Administrator
"""
'pip install -e git+ssh://gitlab@git.q-phantom.com/QPhantom/core.git#egg=QPhantom-core' # 安装陆云飞设计的Metrics.py
#%%
import numpy as np
import pandas as pd
import seaborn as sns
from Metrics import Metrics
m = Metrics()
PRF = pd.read_pickle('/home/mercy/Desktop/PRF.pkl') 
m.plot_predict_summary(PRF['actural_label'],PRF['predict_score'])
sns.plt.hist(PRF['predict_score'],bins=100,normed=True)

#%%
Mc.auc_f1
pwd 'linux查看当前路径'

(test['Embarked'] == 'S').astype(int) # 
test['Embarked'] = [1 if x=='S' else 0 for x in test.Embarked] # 列表推导式 
                   [x for x in test.Embarked if x=='S']        # 列表推导式
test['Embarked'] = np.where(test.Embarked == 'S',1,0)

df['new'] = [str(x)+y if y=='天' else str(x)+'个'+y for x,y in zip(df['项目期限时长'],df['项目期限单位'])] #列表推倒式,同时处理多列
df['new'] = np.where(df['项目期限单位'] == '天', df['项目期限时长'].astype(str) + df['项目期限单位'], df['项目期限时长'].astype(str) + '个' + df['项目期限单位'])

x = (df_Adjustday[i] == date).sum()
Adjust_num = x[x>0].index.get_values()

events['timestamp'] = [int(x.replace(':','').replace('-','')) if pd.notnull else x for x in events.timestamp]

#%%
import bottleneck as bn # bottleneck库中含有movewindow函数,包含move_mean,move_max等函数
Pred = test_new[test_new['predict'] == 1] # DataFrame按列单条件筛选
Mixed= test_new[(test_new['Survived']==test_new['predict']) & (test_new['Survived'] >10)] # DataFrame按列多条件筛选

a.loc[b.index.get_values()] # a的索引包含b的索引，这句的意思是 获取a中与b的index相同部分的数据         

cond = idata_all["open"] == 0.0  # 根据条件选取特定行(返回一个Serise)
idata_all.loc[cond, "open"] = idata_all.loc[cond, "low"] # 向特定行赋予特定值
                
x = np.array([3, 1, 2])
np.argsort(x) # 将x内的值从小到大排序后,返回其在原数组中的序号
array([1, 2, 0])

type(arr) is ndarray       
np.count_nonzero(arr!=arr) # 利用np.nan!=np.nan的性质统计nan的数量
#%%
df.drop_duplicates('device_id',inplace=True) # 除去device_id列中重复的值
new = pd.get_dummies(df) # 特征是类别型的，将其变为亚变量
phg = pd.concat([phg_low,phg_high]) # DataFrame连接
df  = df.sample(frac=1,replace=False) #将df随机采样,frac是采样比例,1是100%;replace=False不放回采样

#%%
pd.read_pickle('/home/mercy/data.pkl')
pd.to_pickle('/home/mercy/data.pkl')

df.to_hdf('/home/data/ushis/hdf_all.hdf','lable')
df_hdf = pd.read_hdf('/home/data/ushis/df_hdf.hdf','lable')

%time df_hdf = pd.read_hdf('/home/data/ushis/df_hdf.hdf','lable') '%time 可以查看该条语句所用的时间'
t1 = time.time()
t2 = time.time()
print(round(t2-t1),'s')

df.index.get_level_values('frute_name') 'df是双索引DataFrame,该项返回的是：apple apple pear pear banana banana'
df.index.levels[0]                      'df是双索引DataFrame,该项返回的是：apple pear banana'
df.groupby(df.index.get_level_values(level="trade_date") // 10000) '将df按年分组'
seaborn.plt.hist(np.log10(d.wd_ev),bins=100,normed=True) 'hist输出数据的概率分布，bins=100输出100个bar，nomed=True纵座标是百分比，否则纵座标是具体个数'
 
seaborn.set(color_codes=True) 
seaborn.set_style("darkgrid")
seaborn.reset_orig()
''' 通过以上两个语句的设定，matplotlib就可以输出seaborn形式的画图 ,第三个语句可以使效果还原为matplotlib'''
x.extend([4,5,6]) '''使用extend方法在列表末位添加多个元素，参数就变成了列表'''
#%%
df.fillna(df.median(), inplace=True) # 用中位数填补Na
df.fillna(df.mode(), inplace=True) # 用众数填补Na ; df.mode()计算众数，默认不计算Na
df.fillna(df.mean(), inplace=True) # 用均值填补Na
#%%
"通过python创建mysql"
DB_CONNECT_STRING ='mysql://root:feng1300@localhost/pp_try?charset=utf8mb4' 
engine = create_engine(DB_CONNECT_STRING, pool_size=64, pool_recycle=400, max_overflow=0)
df = pd.DataFrame(np.ones([3,2]),columns=['a','b'])
pd.io.sql.to_sql(df,'fund_rating', engine,index=False, schema='w_fund', if_exists='append',chunksize=10000) 
#%%
"通过POST方式读取WEB端数据"
url = 'http://218.240.157.156/external/203/5873/'
query = {"query": "过去5天涨幅大于1%的后6只股票"}
r = requests.post(url,json=query)
dt = r.json()
print(dt)
#%%
df.groupby(level='index2')     # df是双索引DataFrame文件，关键词level用于指定索引
df_grp = df.groupby(by=['col1', 'col2']) # by用于指定columns中的列名，可以是多个
'df_grp的类型是groupby Object , df_grp.groups得到的是一个字典 df.groups.keys()就是所有分类标签,'
"df_grp.get_group('key1')获取'key1'组中的具体内容"

#%%
#  强制git pull
git fetch --all  
git reset --hard origin/master 
git pull
#  删除远端文件夹
git rm -r --cached dirname
git commit -m 'say something'
git push origin master
print('F1:','%.2f%%'%F1) # 以百分数输出，并保留2位小数
print('AUC:','%.4f'%AUC) # 以小数输出，并保留4位小数
#%%
'查看哪个函数运行耗费时间:'
%load_ext line_profiler
%lprun -f train clf= train()
#%%
a = 20161218
In: datetime(year=a//10000, month=(a//100)%100, day=a%100)
Out:datetime.datetime(2016, 12, 18, 0, 0)

datetime.strptime(string, "%Y-%m-%d") # 字符串形如'2016-12-05'
datetime.strptime(string, "%Y%m%d") # 字符串形如'20161205'
大写Y:'2017' 小写y:'17'

datetime转成字符串：
date=datetime.datetime.now()
目标字符串 = date.strftime("%Y-%m-%d %H:%M:%S")

d1 = datetime.datetime(2005, 2, 16)
d2 = datetime.datetime(2004, 12, 31)
(d1 – d2).days # 计算两个日期相差多少天

datetime.datetime.now() - datetime.timedelta(3) # 当前日期减去3天
#%%
fig1 = plt.figure()
plt.plot(P,R,label='P--R')
plt.legend(loc = 'lower center') # 'upper right'
plt.title('Pigure1')
fig2 = plt.figure()
plt.plot(fpr,tpr,label='FPR--TPR')
plt.legend(loc = 'lower center') # 'upper left' 
plt.title('Pigure2')
'以上代码输出两张图，以下代码输出一张图的两个子图'
plt.subplot(211) # 两行一列中的第一个子图
plt.plot(P,R)
plt.subplot(212) # 两行一列中的第二个子图
plt.plot(fpr,tpr)


#%%    
df = pd.read_csv('/home/data/ushis/2002US.csv',dtype={'trade_date':object},usecols = ['unique_code','trade_date','wd_open'],index_col = ['trade_date','unique_code'],names=['a','b','c'])
'read_csv将文件读取时利用dtype直接将指定列转化为特定格式，此处object将数字默认转化为str格式,usecols 指定要读取的列,index_col指定某一列，或多列为index; names读取时强制改列名（列名数目要一致）'
#%%  
data_2002 = pd.read_csv('/home/data/ushis/2002US.csv',dtype={'trade_date':object},index_col = ['unique_code','trade_date'])
close_pct_month  = data_2002.query('trade_date== 20020204') or data_2002.query('code == "6000.SH"')     
close_pct_month.reset_index('trade_date',drop = True,inplace = True)
'第一句使用两个index导入，第二句利用其中一个index进行筛选，第三句删除其中一个index'
#%% 
'''
轴向连接 pd.concat() 就是单纯地把两个表拼在一起，这个过程也被称作连接（concatenation）、绑定（binding）或堆叠（stacking）。因此可以想见，这个函数的关键参数应该是 axis，用于指定连接的轴向。
在默认的 axis=0 情况下，pd.concat([obj1,obj2]) 函数的效果与 obj1.append(obj2) 是相同的；
而在 axis=1 的情况下，pd.concat([df1,df2],axis=1) 的效果与 pd.merge(df1,df2,left_index=True,right_index=True,how='outer') 是相同的。
可以理解为 concat 函数使用索引作为“连接键”。
本函数的全部参数为：
concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,keys=None, levels=None, names=None, verify_integrity=False, copy=True)

'''
#%%
'列表list取交集、并集、差集'
c = list(set(a)&set(b)) # list交集
c = list(set(a)|set(b)) # list并集
c = list(set(a)^set(b)) # 返回一个新的 list 包含 s 和 t 中不重复的元素
c = list(set(a)-set(b))# 返回一个新的 list 包含 a 中有但是 b 中没有的元素

字典转json:
data = {'name' : 'ACME','shares' : 100,'price' : 542.23}
json_str = json.dumps(data)
json转字典:
data = json.loads(json_str)

# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)

# 画直方图,normed=False纵坐标返回的是具体数量，normed=True返回的是比例
n, bins, patches = plt.hist(logReturn5, 50, normed=False, facecolor='green', alpha=0.75)
plt.show()

data = data.merge(ret,how = 'outer', left_index = True,right_index =True)
Total_Score = pd.merge(Total_Score,Capital_structure,how = 'inner',left_index=True,right_index=True)
ZZ500_Sus   = pd.merge(ZZ500_OneDay,Suspend_OneDay,how = 'inner',left_on = 'date',right_on = 'date')

Total_Score = Total_Score.rank(axis=0,method='average',ascending=True) # ascending = False 表示从大到小排列（递减）

weight = np.array([1, 1, 1, 1, 1])   #　信号合成，各因子权重  
Total_Score['total_score'] = np.dot(Total_Score, weight)

df[df['E'].isin(['two','four'])]
df.set_index('a',inplace=True)
df.set_index(['a','b'],inplace=True)
df.reset_index(level='trade_date',drop=True,inplace=True) # 将index还原为columns
df.pct_change(periods=1, fill_method='pad', limit=None, freq=None, **kwargs) # 相邻数据变化率
df.rolling(window, min_periods=None, freq=None, center=False, win_type=None, axis=0) # 移动窗口函数，可以对有纵深的数据进行阶段数据处理
df.shift(periods = 3,axis = 1) # 位移函数，df.shift(3)表示移动3位；df.shift(-3)表示移动3位；默认是向下(或向右)移动1位; axis = 0表示移动行，axis = 1表示移动列
df.diff(periods = 1,axis = 0) # diff函数的意思是：先进行shift位移，然后df-df.shift

df.ols(y = list(),x = DataFrame) # 显著性检验
cov = data.loc[:,['CLOSE','AMT']].cov()  # 协方差
corr= data.loc[:,['CLOSE','AMT']].corr() # 相关系数
var = data.loc[:,['CLOSE','AMT']].var()  # 方差

a12 = dict(a1, **a2) #a1,a2分别是两个字典，a12是合并后的字典
BT.rename(columns={'old1':'new1','old2':'new2'}, inplace=True)

a = pd.read_csv(u'E:/2009--2016/MarketData/adjfactorDataYear2009.csv',index_col='Unnamed: 0') #读取csv文件的时候舍弃索引列

stk_pool.to_excel(u'E:/Anaconda_script/Quer/stk_pool.xlsx','Sheet1',index = False)
stk_pool.read_excel(u'E:/Anaconda_script/Quer/stk_pool.xlsx','Sheet1',index = False)

plt.legend(loc = 'lower left',fontsize='x-small') # 图例位置有四个参数‘lower left’，‘lower right’ ，‘upper left’，‘upper right’其中默认是右上角

a = array([[ 1.,  1.],
           [ 5.,  8.]])
b = array([[ 3.,  3.],
           [ 6.,  0.]])
           
vstack((a,b)) == array([[ 1.,  1.],
                        [ 5.,  8.],
                        [ 3.,  3.],
                        [ 6.,  0.]])

hstack((a,b)) == array([[ 1.,  1.,  3.,  3.],
                        [ 5.,  8.,  6.,  0.]])


column_stack((a,b)) == array([[ 1.,  1.,  3.,  3.],
                              [ 5.,  8.,  6.,  0.]])
                              
"fix #7"                          

a = [1,2,3,4,5,6,7,8,9]       # 从list中获取奇数or偶数项
a[::2]  == [1, 3, 5, 7, 9]
a[1::2] == [2, 4, 6, 8]
a[2::3] == [3, 6, 9]

清理数据-------------------------------------------------------------------
df[df.isnull()]
df[df.notnull()]
df.dropna()将所有含有nan项的row删除
df.dropna(axis=1,thresh=3) 将在列的方向上至少有 3 个非 NaN 值时将其保留
df.dropna(how='all')将全部项都是nan的row删除
填充值
df.fillna(0)
df.fillna({1:0,2:0.5}) 对第一列nan值赋0，第二列赋值0.5
df.fillna(method='ffill') 在列方向上以前一个值作为值赋给NaN

# fix zero open with low 
cond = df["open"] == 0.0 
df.loc[cond, "open"] = df.loc[cond, "low"]
一些定式发现-------------------------------------------------------------------------------------------
默认是以column为单位进行操作
比如pd.dataframe(data)   pd.dataframe(dict)
比如df.rank()
比如pd.sort_index()
比如df.sum()
都需要设定axis=1或者指定index才能够进行亚row级别的操作
也就是说我们认知的时候，先认知的是column字段，然后是各个row

两级访问元素
s['a',2]
s[:,2]
df=s.unstack()
s=df.stack()
merge

from sklearn.utils import shuffle
shuffle(arr) # 将数组随机打乱顺序

random.random
random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
random.randint()的函数原型为：random.randint(a, b)，用于生成一个指定范围内的整数。
其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b

random.randrange的函数原型为：random.randrange([start], stop[, step])，
从指定范围内，按指定基数递增的集合中 获取一个随机数。如：random.randrange(10, 100, 2)，
结果相当于从[10, 12, 14, 16, ... 96, 98]序列中获取一个随机数。
random.randrange(10, 100, 2)在结果上与 random.choice(range(10, 100, 2) 等效。

random.sample的函数原型为：random.sample(sequence, k)，从指定序列中随机获取指定长度的片断。sample函数不会修改原有序列。
list=[1,2,3,4,5,6,7,8,9,10]
sl=random.sample(list,5)#从list中随机获取5个元素，作为一个片断返回
print(sl)
print(sl)#原有序列并没有改变。

IN:  np.random.permutation(5)
OUT: array([2, 4, 0, 3, 1])

---------------------------------多重索引columns from_product--------------------------------------------------------------
sex = ['Male', 'Female']
age = ['十','七','九']
cols = pd.MultiIndex.from_product([sex, age], names=['性别', '年龄'])
df1 = pd.DataFrame(np.random.randn(3, 6),columns=cols)
"df1的列数必须是6,因为要满足:len(df1.columns) = len(sex) * len(age) "
--------------------------------多重索引columns from_tuples--------------------------------------------------------
colour = ['green', 'gree', 'gree', 'red', 'yellow', 'yellow', 'blue']  
eg     = ['树叶', '梨', '青苹果', '红旗', '黄袍', '金子', '天空']                   
tuples = list(zip(*[colour, eg]))
cols = pd.MultiIndex.from_tuples(tuples, names=['颜色', '举例'])
df2 = pd.DataFrame(np.random.randn(13, 7))
df2.columns = cols
"df2的列数必须是len(eg)"

cols = pd.MultiIndex.from_tuples(tuples,names=['概述索引','具体索引'])
df4.columns=cols

#%%





