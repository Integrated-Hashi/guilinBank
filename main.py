# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:01:01 2021

@author: integrated_hashi
"""

import  xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#数据处理
key = 'acct_'
name = ['客户号', '卡号']
for i in range(12):
    name.append(key+str(i))
acct_test = pd.read_csv("/data/acct_test.csv",names=name, skiprows=1)
name.append('label')
acct_train = pd.read_csv("/data/acct_train.csv",names=name, skiprows=1)
acct_train.drop(['acct_3', 'acct_4', 'acct_5', 'acct_9'], axis = 1,inplace=True)
acct_test.drop(['acct_3', 'acct_4', 'acct_5', 'acct_9'], axis = 1,inplace=True)

pivot = pd.pivot_table(acct_train, index='acct_1', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'acct1_differ_kahao_cnt'}).reset_index()   
acct_train = pd.merge(acct_train, pivot, on='acct_1', how='left') 

pivot = pd.pivot_table(acct_test, index='acct_1', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'acct1_differ_kahao_cnt'}).reset_index()   
acct_test = pd.merge(acct_test, pivot, on='acct_1', how='left') 

pivot = pd.pivot_table(acct_train, index='acct_0', values='acct_11', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'acct_11': 'acct_0_differ_acct_11_cnt'}).reset_index()   
acct_train = pd.merge(acct_train, pivot, on='acct_0', how='left') 

pivot = pd.pivot_table(acct_test, index='acct_0', values='acct_11', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'acct_11': 'acct_0_differ_acct_11_cnt'}).reset_index()   
acct_test = pd.merge(acct_test, pivot, on='acct_0', how='left') 

key = 'cust_'
name = ['客户号']
for i in range(12):
    name.append(key+str(i))
cust_train = pd.read_csv("/data/cust_train.csv",names=name, skiprows=1)
cust_test = pd.read_csv("/data/cust_test.csv",names=name, skiprows=1)
cust_train.drop(['cust_2', 'cust_3', 'cust_4', 'cust_5', 'cust_6', 'cust_7', 'cust_10'],
                axis = 1, inplace=True)
cust_test.drop(['cust_2', 'cust_3', 'cust_4', 'cust_5', 'cust_6', 'cust_7', 'cust_10'], 
               axis = 1, inplace=True)

pivot = pd.pivot_table(cust_train, index='cust_11', values='客户号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'客户号': 'cust11_differ_kahao_cnt'}).reset_index()   
cust_train = pd.merge(cust_train, pivot, on='cust_11', how='left') 

pivot = pd.pivot_table(cust_test, index='cust_11', values='客户号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'客户号': 'cust11_differ_kahao_cnt'}).reset_index()   
cust_test = pd.merge(cust_test, pivot, on='cust_11', how='left') 

train = pd.merge(acct_train,cust_train,on='客户号')
test = pd.merge(acct_test,cust_test,on='客户号')

key = 'tran_'
name = ['卡号']
for i in range(25):
    name.append(key+str(i))
tran_train = pd.read_csv("/data/tran_train.csv",names=name, skiprows=1)
tran_test = pd.read_csv("/data/tran_test.csv",names=name, skiprows=1)

le = LabelEncoder()
tran_19 = pd.concat([tran_train['tran_19'],tran_test['tran_19']])
tran_20 = pd.concat([tran_train['tran_20'],tran_test['tran_20']])
tran_1920 = pd.concat([tran_19, tran_20])
le.fit(tran_1920)
tran_train['tran_19']=le.transform(tran_train['tran_19'])
tran_test['tran_19']=le.transform(tran_test['tran_19'])
tran_train['tran_20']=le.transform(tran_train['tran_20'])
tran_test['tran_20']=le.transform(tran_test['tran_20'])


tran_train['tran_0_divide_tran_23'] = tran_train['tran_0']*tran_train['tran_23']
tran_test['tran_0_divide_tran_23'] = tran_test['tran_0']*tran_test['tran_23']

train=pd.merge(train,tran_train,on='卡号')
test=pd.merge(test,tran_test,on='卡号')

key = 'cards_'
name = ['卡号', '客户号']
for i in range(10):
    name.append(key+str(i))
cards_train = pd.read_csv("/data/cards_train.csv",names=name, skiprows=1)
cards_test = pd.read_csv("/data/cards_test.csv",names=name, skiprows=1)
cards_train.drop(['客户号', 'cards_8', 'cards_9'], 
                axis = 1, inplace=True)
cards_test.drop(['客户号', 'cards_8', 'cards_9'], 
               axis = 1, inplace=True)

cards_train['date_4'] = pd.to_datetime(cards_train['cards_4'], format='%Y-%m-%d')
cards_train['year1'] =cards_train['date_4'].dt.year.fillna(0).astype("int")
cards_train['month1'] =cards_train['date_4'].dt.month.fillna(0).astype("int")
cards_train['day1'] =cards_train['date_4'].dt.day.fillna(0).astype("int")

cards_train['date_6'] = pd.to_datetime(cards_train['cards_6'], format='%Y-%m-%d')
cards_train['year2'] =cards_train['date_6'].dt.year.fillna(0).astype("int")
cards_train['month2'] =cards_train['date_6'].dt.month.fillna(0).astype("int")
cards_train['day2'] =cards_train['date_6'].dt.day.fillna(0).astype("int")

cards_test['date_4'] = pd.to_datetime(cards_test['cards_4'], format='%Y-%m-%d')
cards_test['year1'] =cards_test['date_4'].dt.year.fillna(0).astype("int")
cards_test['month1'] =cards_test['date_4'].dt.month.fillna(0).astype("int")
cards_test['day1'] =cards_test['date_4'].dt.day.fillna(0).astype("int")

cards_test['date_6'] = pd.to_datetime(cards_test['cards_6'], format='%Y-%m-%d')
cards_test['year2'] =cards_test['date_6'].dt.year.fillna(0).astype("int")
cards_test['month2'] =cards_test['date_6'].dt.month.fillna(0).astype("int")
cards_test['day2'] =cards_test['date_6'].dt.day.fillna(0).astype("int")

#cards_train['week1'] = cards_train['date_4'].map(lambda x: x.weekday()) 
#cards_train['is_weekend1'] = cards_train['week1'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#cards_train['week2'] = cards_train['date_6'].map(lambda x: x.weekday()) 
#cards_train['is_weekend2'] = cards_train['week2'].map(lambda x: 0 if x == 5 or x == 6 else 0)

#cards_test['week1'] = cards_test['date_4'].map(lambda x: x.weekday()) 
#cards_test['is_weekend1'] = cards_test['week1'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#cards_test['week2'] = cards_test['date_6'].map(lambda x: x.weekday()) 
#cards_test['is_weekend2'] = cards_test['week2'].map(lambda x: 1 if x == 5 or x == 6 else 0)

cards_train.drop(['date_4', 'date_6'], 
                axis = 1, inplace=True)
cards_test.drop(['date_4', 'date_6'], 
                axis = 1, inplace=True)

le = LabelEncoder()
cards_4 = pd.concat([cards_train['cards_4'],cards_test['cards_4']])
cards_6 = pd.concat([cards_train['cards_6'],cards_test['cards_6']])
cards_46 = pd.concat([cards_4, cards_6])
le.fit(cards_46)
cards_train['cards_4']=le.transform(cards_train['cards_4'])
cards_test['cards_4']=le.transform(cards_test['cards_4'])
cards_train['cards_6']=le.transform(cards_train['cards_6'])
cards_test['cards_6']=le.transform(cards_test['cards_6'])

pivot = pd.pivot_table(cards_train, index='cards_4', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'cards4_differ_kahao_cnt'}).reset_index()   
cards_train = pd.merge(cards_train, pivot, on='cards_4', how='left') 

pivot = pd.pivot_table(cards_test, index='cards_4', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'cards4_differ_kahao_cnt'}).reset_index()   
cards_test = pd.merge(cards_test, pivot, on='cards_4', how='left') 

pivot = pd.pivot_table(cards_train, index='cards_3', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'cards3_differ_kahao_cnt'}).reset_index()   
cards_train = pd.merge(cards_train, pivot, on='cards_3', how='left') 

pivot = pd.pivot_table(cards_test, index='cards_3', values='卡号', aggfunc=lambda x: len(set(x)))   
pivot = pd.DataFrame(pivot).rename(columns={'卡号': 'cards3_differ_kahao_cnt'}).reset_index()   
cards_test = pd.merge(cards_test, pivot, on='cards_3', how='left') 

train=pd.merge(train,cards_train,on='卡号')
test=pd.merge(test,cards_test,on='卡号')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cat = ['acct_0', 'tran_19', 'tran_20']
num = ['acct_11','tran_0', 'cust_11', 'tran_1', 'tran_2',
       'tran_6', 'tran_7', 'tran_16', 'tran_23', 'tran_24']
for i in cat:
    for j in num:
        tmp_mean = train.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j:i+'_'+j+'_mean'},inplace=True)
        train = pd.merge(train, tmp_mean, on=i, how='left') 

        tmp_median = train.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j:i+'_'+j+'_median'},inplace=True)
        train = pd.merge(train, tmp_median, on=i, how='left')

        tmp_std = train.groupby(i)[j].std().reset_index()
        tmp_std.rename(columns={j:i+'_'+j+'_std'},inplace=True)
        train = pd.merge(train, tmp_std, on=i, how='left')

        tmp_minus = tmp_median 
        tmp_minus[i+'_'+j+'_median'] = abs(tmp_median[i+'_'+j+'_median'] - tmp_mean[i+'_'+j+'_mean'])
        tmp_minus.rename(columns={i+'_'+j+'_median':i+'_'+j+'_minus'},inplace=True)
        train = pd.merge(train, tmp_minus, on=i, how='left')

        tmp_mean = train.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j:i+'_'+j+'_mean'},inplace=True) 
        tmp_median = train.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j:i+'_'+j+'_median'},inplace=True)
        tmp_divde = tmp_median
        tmp_divde[i+'_'+j+'_median'] = tmp_median[i+'_'+j+'_median'] / tmp_mean[i+'_'+j+'_mean']
        tmp_divde.rename(columns={i+'_'+j+'_median':i+'_'+j+'_divde'},inplace=True)
        train = pd.merge(train, tmp_divde, on=i, how='left')

for i in cat:
    for j in num:
        tmp_mean = test.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j:i+'_'+j+'_mean'},inplace=True)
        test = pd.merge(test, tmp_mean, on=i, how='left') 

        tmp_median = test.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j:i+'_'+j+'_median'},inplace=True)
        test = pd.merge(test, tmp_median, on=i, how='left')

        tmp_std = test.groupby(i)[j].std().reset_index()
        tmp_std.rename(columns={j:i+'_'+j+'_std'},inplace=True)
        test = pd.merge(test, tmp_std, on=i, how='left')

        tmp_minus = tmp_median 
        tmp_minus[i+'_'+j+'_median'] = abs(tmp_median[i+'_'+j+'_median'] - tmp_mean[i+'_'+j+'_mean'])
        tmp_minus.rename(columns={i+'_'+j+'_median':i+'_'+j+'_minus'},inplace=True)
        test = pd.merge(test, tmp_minus, on=i, how='left')

        tmp_mean = test.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j:i+'_'+j+'_mean'},inplace=True) 
        tmp_median = test.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j:i+'_'+j+'_median'},inplace=True)
        tmp_divde = tmp_median
        tmp_divde[i+'_'+j+'_median'] = tmp_median[i+'_'+j+'_median'] / tmp_mean[i+'_'+j+'_mean']
        tmp_divde.rename(columns={i+'_'+j+'_median':i+'_'+j+'_divde'},inplace=True)
        test = pd.merge(test, tmp_divde, on=i, how='left')

train.fillna(-1, inplace = True)
test.fillna(-1, inplace = True)

train_label = train['label'].tolist()

param = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0.1,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}

dtrain = xgb.DMatrix(train.drop(['客户号', '卡号', 'label'], axis=1), label=train['label'])
dtest = xgb.DMatrix(test.drop(['客户号', '卡号'], axis=1))
watchlist = [(dtrain, 'train')]
bst = xgb.train(param,dtrain,num_boost_round=1500, evals=watchlist)
predict = bst.predict(dtest)
predict = pd.DataFrame(predict, columns=['target'])
result = pd.concat([test[['卡号']], predict], axis=1)

feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
feat_importance['feature_name'] = bst.get_score().keys()
feat_importance['importance'] = bst.get_score().values()
feat_importance.sort_values(['importance'], ascending=False, inplace=True)
#result['target'] = result['target'].map(lambda x: 0 if x<0.01 else x)
print(result)
print(feat_importance)
result.to_csv('./pre_result.csv',index=False)