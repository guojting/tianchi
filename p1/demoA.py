#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

dir='../data/'
train = pd.read_table(dir + 'train_20171215.txt',engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt',engine='python')
sample_A = pd.read_table(dir + 'sample_A_20171225.txt',engine='python',header=None)
sample_A.columns = ['date','day_of_week']

# 因为第一赛季只是预测与时间相关的cnt的数量
# 所以可以对数据以dat和dow进行数据合并
train = train.groupby(['date','day_of_week'],as_index=False).cnt.sum()
plt.plot(train['day_of_week'],train['cnt'],'*')
plt.show()

for i in range(7):
    tmp = train[train['day_of_week']==i+1]
    plt.subplot(7, 1, i+1)
    plt.plot(tmp['date'],tmp['cnt'],'*')
plt.show()

xx_train = train[train['date']<=756]
xx_test = train[train['date']>756]
print('test shape',xx_test.shape)
print('train shape',xx_train.shape)

#方案零：均值大法（原始数据验证）
from sklearn.metrics import mean_squared_error
# 线下统计每周的均值数据，不加权
xx_train = xx_train.groupby(['day_of_week'],as_index=False).cnt.mean()
xx_result = pd.merge(xx_test,xx_train,on=['day_of_week'],how='left')
print('xx_result shape',xx_result.shape)
print(xx_result)
print(mean_squared_error(xx_result['cnt_x'],xx_result['cnt_y']))

for i in range(7):
    tmp = xx_result[xx_result['day_of_week']==i+1]
    print('周%d'%(i+1),mean_squared_error(tmp['cnt_x'],tmp['cnt_y']))


#方案一：加权平均大法
def xx(df):
   df['w_cnt'] = (df['cnt'] * df['weight']).sum() / sum(df['weight'])
   return df

xx_train = train[train['date']<=756]
xx_train['weight'] = ((xx_train['date'] + 1) / len(xx_train)) ** 6
xx_train = xx_train.groupby(['day_of_week'],as_index=False).apply(xx).reset_index()
xx_test = train[train['date']>756]
print('test shape',xx_test.shape)
print('train shape',xx_train.shape)
# #
from sklearn.metrics import mean_squared_error
# # 这里是加权的方案
xx_train = xx_train.groupby(['day_of_week'],as_index=False).w_cnt.mean()

xx_result = pd.merge(xx_test,xx_train,on=['day_of_week'],how='left')
print('xx_result shape',xx_result.shape)
print(xx_result)
print(mean_squared_error(xx_result['cnt'],xx_result['w_cnt']))

#方案二：时序转回归大法
from pandas import DataFrame
from pandas import concat


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

time_cnt = list(train['cnt'].values)
# nin 前看 nout后看 这个题目需要前看
time2sup = series_to_supervised(data=time_cnt,n_in=276,dropnan=True)

import lightgbm as lgb
gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=10000)

print(time2sup.shape)
x_train = time2sup[time2sup.index<755]
x_test = time2sup[time2sup.index>755]
# 这个方式其实是最简单的，后面还可以很多改善，比如滚动预测一类
print(x_train.shape)
print(x_test.shape)

y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')

# 损失函数mse
gbm0.fit(x_train.values,y_train,eval_set=[(x_test.values,y_test)],eval_metric='mse',early_stopping_rounds=15)
print(gbm0.predict(x_test.values))

from sklearn.metrics import mean_squared_error
line1 = plt.plot(range(len(x_test)),gbm0.predict(x_test.values),label=u'predict')
line2 = plt.plot(range(len(y_test)),y_test.values,label=u'true')
plt.legend()
plt.show()
