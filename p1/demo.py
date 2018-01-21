#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt


dir='../data/'
train=pd.read_table(dir+'train_20171215.txt',engine='python')
test_A=pd.read_table(dir+'test_A_20171225.txt',engine='python')
sample_A=pd.read_table(dir+'sample_A_20171225.txt',engine='python',header=None)

print(train.info())
print(test_A.info())

print(train['day_of_week'].unique())
print(test_A['day_of_week'].unique())

plt.boxplot(train['cnt'])
plt.show()

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew
sns.distplot(train['cnt'],fit=norm)

plt.plot(train['date'],train['cnt'])
plt.show()

print(train['cnt'].describe())

from sklearn.metrics import mean_squared_error
train['25%']=221
train['50%']=351
train['75%']=496
train['median']=train['cnt'].median()
train['mean']=train['cnt'].mean()
print(mean_squared_error(train['cnt'],train['25%']))
print(mean_squared_error(train['cnt'],train['50%']))
print(mean_squared_error(train['cnt'],train['75%']))
print(mean_squared_error(train['cnt'],train['median']))
print(mean_squared_error(train['cnt'],train['mean']))

monday=train[train['day_of_week']==1]
plt.plot(range(len(monday)),monday['cnt'])
plt.show()

res = train.groupby(['day_of_week'],as_index=False).cnt.mean()
xx = train.merge(res,on=['day_of_week'])
print(xx.head())
print(mean_squared_error(xx['cnt_x'],xx['cnt_y']))
