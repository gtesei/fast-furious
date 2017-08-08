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

x_train_3 = pd.read_csv('xtrain_3.csv') 
x_test_3 = pd.read_csv('xtest_3.csv')  
print("Feature set 3: X_train: {}, X_test: {}".format(x_train_3.shape, x_test_3.shape))

y_train = df_train['is_duplicate'].values

x_train = pd.concat([x_train_1,x_train_2,x_train_3],axis=1)
x_test = pd.concat([x_test_1,x_test_2,x_test_3],axis=1)
print("Merge: X_train: {}, X_test: {}".format(x_train.shape, x_test.shape))

assert x_train.shape[0] == df_train.shape[0]
assert x_test.shape[0] == df_test.shape[0]

# resample 
if 1: # Now we oversample the negative class - on your own risk of overfitting!
  pos_train = x_train[y_train == 1]
  neg_train = x_train[y_train == 0]
  print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  p = 0.165
  scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
  while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
  neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
  print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  x_train = pd.concat([pos_train, neg_train])
  y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
  del pos_train, neg_train

# XGB
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS

print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
x, X_val, ytrain, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RS)

print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
xg_train = xgb.DMatrix(x, label=ytrain)
xg_val = xgb.DMatrix(X_val, label=y_val)

watchlist  = [(xg_train,'train'), (xg_val,'eval')]

clf = xgb.train(params=params,dtrain=xg_train,num_boost_round=ROUNDS,early_stopping_rounds=200,evals=watchlist)
preds = clf.predict(xgb.DMatrix(x_test))

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv("xgb_feat_seed_3{}_n{}.csv".format(RS, ROUNDS), index=False)

print("Done.")
