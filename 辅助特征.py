
#时间特征出判断没用
#cards_train['week1'] = cards_train['date_4'].map(lambda x: x.weekday()) 
#cards_train['is_weekend1'] = cards_train['week1'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#cards_train['week2'] = cards_train['date_6'].map(lambda x: x.weekday()) 
#cards_train['is_weekend2'] = cards_train['week2'].map(lambda x: 0 if x == 5 or x == 6 else 0)

#cards_test['week1'] = cards_test['date_4'].map(lambda x: x.weekday()) 
#cards_test['is_weekend1'] = cards_test['week1'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#cards_test['week2'] = cards_test['date_6'].map(lambda x: x.weekday()) 
#cards_test['is_weekend2'] = cards_test['week2'].map(lambda x: 1 if x == 5 or x == 6 else 0)

#新加的 貌似重要值很大
mean1=train.groupby(i)[j].transform('mean')
median1=train.groupby(i)[j].transform('median')
train[i+'/mean_'+j] = train[j]/(mean1+1e-5)
train[i+'/median_'+j] = train[j]/(median1+1e-5)


tran_train['tran_0_divide_tran_2'] = (tran_train['tran_0']+1e-5)-(tran_train['tran_2']+1e-5)
tran_test['tran_0_divide_tran_2'] = (tran_test['tran_0']+1e-5)-(tran_test['tran_2']+1e-5)

tran_train['tran_9_divide_tran_10'] = (tran_train['tran_9']+1e-5)-(tran_train['tran_10']+1e-5)
tran_test['tran_9_divide_tran_10'] = (tran_test['tran_9']+1e-5)-(tran_test['tran_10']+1e-5)

tran_train['tran_24_divide_tran_23'] = (tran_train['tran_24']+1e-5)-(tran_train['tran_23']+1e-5)
tran_test['tran_24_divide_tran_23'] = (tran_test['tran_24']+1e-5)-(tran_test['tran_23']+1e-5)

#平均值偏离值
tran_train['tran_0-mean'] = tran_train['tran_0'] - (tran_train['tran_0'].mean())
tran_test['tran_0-mean'] = tran_test['tran_0'] - (tran_test['tran_0'].mean())

