# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:01:01 2021

@author: integrated_hashi
"""

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#数据处理
key = 'acct_'
name1 = [
    '客户号', '卡号', 'acct_x_cat1', 'acct_x_cat2', 'acct_x_cat3', 'acct_x_date1',
    'acct_x_date2', 'acct_x_cat4', 'acct_x_cat5', 'acct_x_cat6', 'acct_x_cat7',
    'acct_x_cat8', 'acct_x_cat9', 'acct_x_num1', 'label'
]
name2 = [
    '客户号', '卡号', 'acct_x_cat1', 'acct_x_cat2', 'acct_x_cat3', 'acct_x_date1',
    'acct_x_date2', 'acct_x_cat4', 'acct_x_cat5', 'acct_x_cat6', 'acct_x_cat7',
    'acct_x_cat8', 'acct_x_cat9', 'acct_x_num1'
]
acct_test = pd.read_csv("/data/acct_test.csv", names=name2, skiprows=1)
acct_train = pd.read_csv("/data/acct_train.csv", names=name1, skiprows=1)
acct_train.drop(['acct_x_date1', 'acct_x_date2', 'acct_x_cat4', 'acct_x_cat8'],
                axis=1,
                inplace=True)
acct_test.drop(['acct_x_date1', 'acct_x_date2', 'acct_x_cat4', 'acct_x_cat8'],
               axis=1,
               inplace=True)

pivot = pd.pivot_table(acct_train,
                       index='acct_x_cat2',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'acct_x_cat2_differ_kahao_cnt'
}).reset_index()
acct_train = pd.merge(acct_train, pivot, on='acct_x_cat2', how='left')

pivot = pd.pivot_table(acct_test,
                       index='acct_x_cat2',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'acct_x_cat2_differ_kahao_cnt'
}).reset_index()
acct_test = pd.merge(acct_test, pivot, on='acct_x_cat2', how='left')

pivot = pd.pivot_table(acct_train,
                       index='acct_x_cat1',
                       values='acct_x_num1',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(
    columns={
        'acct_x_num1': 'acct_x_cat1_differ_acct_x_num1_cnt'
    }).reset_index()
acct_train = pd.merge(acct_train, pivot, on='acct_x_cat1', how='left')

pivot = pd.pivot_table(acct_test,
                       index='acct_x_cat1',
                       values='acct_x_num1',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(
    columns={
        'acct_x_num1': 'acct_x_cat1_differ_acct_x_num1_cnt'
    }).reset_index()
acct_test = pd.merge(acct_test, pivot, on='acct_x_cat1', how='left')

key = 'cust_'
name = [
    '客户号', 'cust_x_cat1', 'cust_x_cat2', 'cust_x_cat3', 'cust_x_cat4',
    'cust_x_cat5', 'cust_x_cat6', 'cust_x_cat7', 'cust_x_cat8', 'cust_x_cat9',
    'cust_x_cat10', 'cust_x_cat11', 'cust_x_num1'
]
cust_train = pd.read_csv("/data/cust_train.csv", names=name, skiprows=1)
cust_test = pd.read_csv("/data/cust_test.csv", names=name, skiprows=1)
cust_train.drop([
    'cust_x_cat3', 'cust_x_cat4', 'cust_x_cat5', 'cust_x_cat6', 'cust_x_cat7',
    'cust_x_cat8', 'cust_x_cat11'
],
                axis=1,
                inplace=True)
cust_test.drop([
    'cust_x_cat3', 'cust_x_cat4', 'cust_x_cat5', 'cust_x_cat6', 'cust_x_cat7',
    'cust_x_cat8', 'cust_x_cat11'
],
               axis=1,
               inplace=True)

pivot = pd.pivot_table(cust_train,
                       index='cust_x_num1',
                       values='客户号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '客户号': 'cust_x_num1_differ_kehuhao_cnt'
}).reset_index()
cust_train = pd.merge(cust_train, pivot, on='cust_x_num1', how='left')

pivot = pd.pivot_table(cust_test,
                       index='cust_x_num1',
                       values='客户号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '客户号': 'cust_x_num1_differ_kehuhao_cnt'
}).reset_index()
cust_test = pd.merge(cust_test, pivot, on='cust_x_num1', how='left')

train = pd.merge(acct_train, cust_train, on='客户号')
test = pd.merge(acct_test, cust_test, on='客户号')

key = 'tran_'
name = [
    '卡号', 'tran_x_num1', 'tran_x_num2', 'tran_x_num3', 'tran_x_num4',
    'tran_x_num5', 'tran_x_num6', 'tran_x_num7', 'tran_x_num8', 'tran_x_num9',
    'tran_x_num10', 'tran_x_num11', 'tran_x_num12', 'tran_x_num13',
    'tran_x_num14', 'tran_x_num15', 'tran_x_num16', 'tran_x_num17',
    'tran_x_num18', 'tran_x_num19', 'tran_x_cat1', 'tran_x_cat2',
    'tran_x_num20', 'tran_x_num21', 'tran_x_num22', 'tran_x_num23'
]
tran_train = pd.read_csv("/data/tran_train.csv", names=name, skiprows=1)
tran_test = pd.read_csv("/data/tran_test.csv", names=name, skiprows=1)

le = LabelEncoder()
tran_cat1 = pd.concat([tran_train['tran_x_cat1'], tran_test['tran_x_cat1']])
tran_cat2 = pd.concat([tran_train['tran_x_cat2'], tran_test['tran_x_cat2']])
tran_cat = pd.concat([tran_cat1, tran_cat2])
le.fit(tran_cat)
tran_train['tran_x_cat1'] = le.transform(tran_train['tran_x_cat1'])
tran_test['tran_x_cat1'] = le.transform(tran_test['tran_x_cat1'])
tran_train['tran_x_cat2'] = le.transform(tran_train['tran_x_cat2'])
tran_test['tran_x_cat2'] = le.transform(tran_test['tran_x_cat2'])

tran_train['tran_x_num1_plus_tran_x_num19'] = tran_train[
    'tran_x_num1'] * tran_train['tran_x_num19']
tran_test['tran_x_num1_plus_tran_x_num19'] = tran_test[
    'tran_x_num1'] * tran_test['tran_x_num19']

tran_train['tran_x_num1_minus_tran_x_num19'] = tran_train[
    'tran_x_num1'] - tran_train['tran_x_num19']
tran_test['tran_x_num1_minus_tran_x_num19'] = tran_test[
    'tran_x_num1'] - tran_test['tran_x_num19']

tran_train['tran_x_num2_minus_tran_x_num7'] = tran_train[
    'tran_x_num2'] / tran_train['tran_x_num7']
tran_test['tran_x_num2_minus_tran_x_num7'] = tran_test[
    'tran_x_num2'] / tran_test['tran_x_num7']

tran_train['tran_x_num1_plus_tran_x_num22'] = tran_train[
    'tran_x_num1'] * tran_train['tran_x_num22']
tran_test['tran_x_num1_plus_tran_x_num22'] = tran_test[
    'tran_x_num1'] * tran_test['tran_x_num22']

tran_train['tran_x_num1_plus_tran_x_num23'] = tran_train[
    'tran_x_num1'] * tran_train['tran_x_num23']
tran_test['tran_x_num1_plus_tran_x_num23'] = tran_test[
    'tran_x_num1'] * tran_test['tran_x_num23']

tran_train['tran_x_num23_plus_tran_x_num22'] = tran_train[
    'tran_x_num23'] * tran_train['tran_x_num22']
tran_test['tran_x_num23_plus_tran_x_num22'] = tran_test[
    'tran_x_num23'] * tran_test['tran_x_num22']

train = pd.merge(train, tran_train, on='卡号')
test = pd.merge(test, tran_test, on='卡号')

key = 'cards_'
name = [
    '卡号', '客户号', 'cards_x_cat1', 'cards_x_cat2', 'cards_x_cat3',
    'cards_x_cat4', 'cards_x_date1', 'cards_x_cat5', 'cards_x_date1.1',
    'cards_x_cat6', 'cards_x_cat7', 'cards_x_cat8'
]
cards_train = pd.read_csv("/data/cards_train.csv", names=name, skiprows=1)
cards_test = pd.read_csv("/data/cards_test.csv", names=name, skiprows=1)
cards_train.drop(['客户号', 'cards_x_cat7', 'cards_x_cat8'], axis=1, inplace=True)
cards_test.drop(['客户号', 'cards_x_cat7', 'cards_x_cat8'], axis=1, inplace=True)

cards_train['date_4'] = pd.to_datetime(cards_train['cards_x_date1'],
                                       format='%Y-%m-%d')
cards_train['year1'] = cards_train['date_4'].dt.year.fillna(0).astype("int")
cards_train['month1'] = cards_train['date_4'].dt.month.fillna(0).astype("int")
cards_train['day1'] = cards_train['date_4'].dt.day.fillna(0).astype("int")

cards_train['date_6'] = pd.to_datetime(cards_train['cards_x_date1.1'],
                                       format='%Y-%m-%d')
cards_train['year2'] = cards_train['date_6'].dt.year.fillna(0).astype("int")
cards_train['month2'] = cards_train['date_6'].dt.month.fillna(0).astype("int")
cards_train['day2'] = cards_train['date_6'].dt.day.fillna(0).astype("int")

cards_test['date_4'] = pd.to_datetime(cards_test['cards_x_date1'],
                                      format='%Y-%m-%d')
cards_test['year1'] = cards_test['date_4'].dt.year.fillna(0).astype("int")
cards_test['month1'] = cards_test['date_4'].dt.month.fillna(0).astype("int")
cards_test['day1'] = cards_test['date_4'].dt.day.fillna(0).astype("int")

cards_test['date_6'] = pd.to_datetime(cards_test['cards_x_date1.1'],
                                      format='%Y-%m-%d')
cards_test['year2'] = cards_test['date_6'].dt.year.fillna(0).astype("int")
cards_test['month2'] = cards_test['date_6'].dt.month.fillna(0).astype("int")
cards_test['day2'] = cards_test['date_6'].dt.day.fillna(0).astype("int")

cards_train.drop(['date_4', 'date_6'], axis=1, inplace=True)
cards_test.drop(['date_4', 'date_6'], axis=1, inplace=True)

le = LabelEncoder()
cards_4 = pd.concat(
    [cards_train['cards_x_date1'], cards_test['cards_x_date1']])
cards_6 = pd.concat(
    [cards_train['cards_x_date1.1'], cards_test['cards_x_date1.1']])
cards_46 = pd.concat([cards_4, cards_6])
le.fit(cards_46)
cards_train['cards_x_date1'] = le.transform(cards_train['cards_x_date1'])
cards_test['cards_x_date1'] = le.transform(cards_test['cards_x_date1'])
cards_train['cards_x_date1.1'] = le.transform(cards_train['cards_x_date1.1'])
cards_test['cards_x_date1.1'] = le.transform(cards_test['cards_x_date1.1'])

pivot = pd.pivot_table(cards_train,
                       index='cards_x_date1',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'cards_x_date1_differ_kahao_cnt'
}).reset_index()
cards_train = pd.merge(cards_train, pivot, on='cards_x_date1', how='left')

pivot = pd.pivot_table(cards_test,
                       index='cards_x_date1',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'cards_x_date1_differ_kahao_cnt'
}).reset_index()
cards_test = pd.merge(cards_test, pivot, on='cards_x_date1', how='left')

pivot = pd.pivot_table(cards_train,
                       index='cards_x_cat4',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'cards_x_cat4_differ_kahao_cnt'
}).reset_index()
cards_train = pd.merge(cards_train, pivot, on='cards_x_cat4', how='left')

pivot = pd.pivot_table(cards_test,
                       index='cards_x_cat4',
                       values='卡号',
                       aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={
    '卡号': 'cards_x_cat4_differ_kahao_cnt'
}).reset_index()
cards_test = pd.merge(cards_test, pivot, on='cards_x_cat4', how='left')

train = pd.merge(train, cards_train, on='卡号')
test = pd.merge(test, cards_test, on='卡号')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cat = ['acct_x_cat1', 'tran_x_cat1', 'tran_x_cat2']
num = [
    'acct_x_num1', 'tran_x_num1', 'cust_x_num1', 'tran_x_num2', 'tran_x_num3',
    'tran_x_num7', 'tran_x_num8', 'tran_x_num17', 'tran_x_num22',
    'tran_x_num23'
]
for i in cat:
    for j in num:
        tmp_mean = train.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j: i + '_' + j + '_mean'}, inplace=True)
        train = pd.merge(train, tmp_mean, on=i, how='left')

        tmp_median = train.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j: i + '_' + j + '_median'}, inplace=True)
        train = pd.merge(train, tmp_median, on=i, how='left')

        tmp_std = train.groupby(i)[j].std().reset_index()
        tmp_std.rename(columns={j: i + '_' + j + '_std'}, inplace=True)
        train = pd.merge(train, tmp_std, on=i, how='left')

        mean1 = train.groupby(i)[j].transform('mean')
        median1 = train.groupby(i)[j].transform('median')
        train[i + '/mean_' + j] = train[j] / (mean1 + 1e-5)
        train[i + '/median_' + j] = train[j] / (median1 + 1e-5)

        tmp_minus = tmp_median
        tmp_minus[i + '_' + j +
                  '_median'] = abs(tmp_median[i + '_' + j + '_median'] -
                                   tmp_mean[i + '_' + j + '_mean'])
        tmp_minus.rename(
            columns={i + '_' + j + '_median': i + '_' + j + '_minus'},
            inplace=True)
        train = pd.merge(train, tmp_minus, on=i, how='left')

        tmp_mean = train.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j: i + '_' + j + '_mean'}, inplace=True)
        tmp_median = train.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j: i + '_' + j + '_median'}, inplace=True)
        tmp_divde = tmp_median
        tmp_divde[i + '_' + j +
                  '_median'] = tmp_median[i + '_' + j +
                                          '_median'] / tmp_mean[i + '_' + j +
                                                                '_mean']
        tmp_divde.rename(
            columns={i + '_' + j + '_median': i + '_' + j + '_divde'},
            inplace=True)
        train = pd.merge(train, tmp_divde, on=i, how='left')

for i in cat:
    for j in num:
        tmp_mean = test.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j: i + '_' + j + '_mean'}, inplace=True)
        test = pd.merge(test, tmp_mean, on=i, how='left')

        tmp_median = test.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j: i + '_' + j + '_median'}, inplace=True)
        test = pd.merge(test, tmp_median, on=i, how='left')

        tmp_std = test.groupby(i)[j].std().reset_index()
        tmp_std.rename(columns={j: i + '_' + j + '_std'}, inplace=True)
        test = pd.merge(test, tmp_std, on=i, how='left')

        mean1 = test.groupby(i)[j].transform('mean')
        median1 = test.groupby(i)[j].transform('median')
        test[i + '/mean_' + j] = test[j] / (mean1 + 1e-5)
        test[i + '/median_' + j] = test[j] / (median1 + 1e-5)

        tmp_minus = tmp_median
        tmp_minus[i + '_' + j +
                  '_median'] = abs(tmp_median[i + '_' + j + '_median'] -
                                   tmp_mean[i + '_' + j + '_mean'])
        tmp_minus.rename(
            columns={i + '_' + j + '_median': i + '_' + j + '_minus'},
            inplace=True)
        test = pd.merge(test, tmp_minus, on=i, how='left')

        tmp_mean = test.groupby(i)[j].mean().reset_index()
        tmp_mean.rename(columns={j: i + '_' + j + '_mean'}, inplace=True)
        tmp_median = test.groupby(i)[j].median().reset_index()
        tmp_median.rename(columns={j: i + '_' + j + '_median'}, inplace=True)
        tmp_divde = tmp_median
        tmp_divde[i + '_' + j +
                  '_median'] = tmp_median[i + '_' + j +
                                          '_median'] / tmp_mean[i + '_' + j +
                                                                '_mean']
        tmp_divde.rename(
            columns={i + '_' + j + '_median': i + '_' + j + '_divde'},
            inplace=True)
        test = pd.merge(test, tmp_divde, on=i, how='left')

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

train_label = train['label'].tolist()

param = {
    'booster': 'gbtree',
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
    'scale_pos_weight': 1
}

dtrain = xgb.DMatrix(train.drop(['客户号', '卡号', 'label'], axis=1),
                     label=train['label'])
dtest = xgb.DMatrix(test.drop(['客户号', '卡号'], axis=1))
watchlist = [(dtrain, 'train')]
bst = xgb.train(param, dtrain, num_boost_round=1400, evals=watchlist)
predict = bst.predict(dtest)
predict = pd.DataFrame(predict, columns=['target'])
result = pd.concat([test[['卡号']], predict], axis=1)

feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
feat_importance['feature_name'] = bst.get_score().keys()
feat_importance['importance'] = bst.get_score().values()
feat_importance.sort_values(['importance'], ascending=False, inplace=True)
print(result)
print(feat_importance)
result.to_csv('./pre_result.csv', index=False)
