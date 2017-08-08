import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.metrics import roc_auc_score, log_loss

RS = 12357
ROUNDS = 10000

print("Started")
np.random.seed(RS)
input_folder = './data/'

# data 
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

x_train_1 = pd.read_csv('xtrain.csv')
del x_train_1['Unnamed: 0']
x_test_1 = pd.read_csv('xtest.csv')
del x_test_1['Unnamed: 0']
print("Feature set 1: X_train: {}, X_test: {}".format(x_train_1.shape,x_test_1.shape))

x_train_2 = pd.read_csv('xtrain_2.csv') 
#del x_train_2['Unnamed: 0'] 
x_test_2 = pd.read_csv('xtest_2.csv')  
#del x_test_2['Unnamed: 0']
print("Feature set 2: X_train: {}, X_test: {}".format(x_train_2.shape, x_test_2.shape))

y_train = df_train['is_duplicate'].values

x_train = pd.concat([x_train_1,x_train_2],axis=1)
x_test = pd.concat([x_test_1,x_test_2],axis=1)
print("Merge: X_train: {}, X_test: {}".format(x_train.shape, x_test.shape))

assert x_train.shape[0] == df_train.shape[0]
assert x_test.shape[0] == df_test.shape[0]

# no resample for re-balancing classes  

# split data set 
print(">> splitting dataset into train and xval ...")
tr_cut = np.int_( np.floor(df_train.shape[0] * 0.8) )
perm = np.random.permutation(df_train.shape[0])
tr_idx = perm[0:tr_cut]
val_ix = perm[tr_cut:]
x_train_train = x_train.ix[tr_idx]
x_val =  x_train.ix[val_ix]
y_train_train = y_train[tr_idx]
y_val =  y_train[val_ix] 

# XGB
print(">> training XGB ...") 
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS


xg_train = xgb.DMatrix(x_train_train, label=y_train_train)
xg_val = xgb.DMatrix(x_val, label=y_val)

clf = xgb.train(params=params,dtrain=xg_train,num_boost_round=ROUNDS)
pred_val = clf.predict(xg_val)
print(">> logloss::",log_loss(y_true=y_val, y_pred=pred_val))

print(">> assembling results ... ")
df_res =  pd.DataFrame({'pred_val':pred_val},index=x_val.index) 
x_all = pd.concat([df_train.ix[val_ix],x_val],axis=1)
x_all2 = pd.concat([x_all,df_res],axis=1)  
x_all2.to_csv("xgb_error_2.csv", index=False)
