import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import re 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import nltk 
import xgboost as xgb

from extract_feat_base import * 

### FUNC ########################################################################
def xgb_feat_importance(model,cols,file_name):
  print('-----> Feature importance ... ')
  feature_importance_dict = model.get_fscore()
  fs = ['f%i' % i for i in range(len(cols))]
  f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()), 'importance': list(feature_importance_dict.values())})
  f2 = pd.DataFrame({'f': fs, 'feature_name': cols})
  feature_importance = pd.merge(f1, f2, how='right', on='f')
  feature_importance = feature_importance.fillna(0)
  feature_importance.sort_values(by='importance', ascending=False)
  print(feature_importance.sort_values)
  feature_importance.to_csv(file_name, index=False) 
#################################################################################


### FEATURE ENG. ################################################################
meta = {'target': 'deal_probability', 
        'test_id': 'item_id', 
       'cols': {
           'item_id': 'REM', 
           'user_id': 'CAT', 
           'region': 'CAT', 
           'city':   'CAT', 
           'parent_category_name': 'CAT',
           'category_name': 'CAT',
           'param_1': 'CAT', 
           'param_2': 'CAT', 
           'param_3': 'CAT', 
           'title': 'LEN',  
           'description': 'LEN' , 
           'price': 'NUM', 
           'item_seq_number': 'NUM', 
           'activation_date': 'DATE',           
           'user_type': 'CAT', 
           'image': 'REM',
           'image_top_1': 'NUM'
       }}

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('--------------> Basic Feature Engineering ... ')
all_data , y_train = encode_dataset(train=train,test=test,meta=meta)
print(all_data.head())
#################################################################################

### MODELING ####################################################################
print('--------------> Modeling ... ')
train_obs = len(y_train)
Xtr, Xv, ytr, yv = train_test_split(all_data[:train_obs].values, y_train, test_size=0.1, random_state=1973)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(all_data[train_obs:].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 50,
            'eta': 0.01,
            'colsample_bytree': 0.5, #0.3
            'max_depth': 15, # 10
            'subsample': 0.5, #0.8
            'lambda': 0.5,
            'nthread': -1,
            'booster' : 'gbtree',
            'silent': 1,
            'eval_metric': 'rmse',
            'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 10000, watchlist, early_stopping_rounds=50,maximize=False, verbose_eval=10)

print('Modeling RMSE %.5f' % model.best_score)

print('--------------> Submission ... ')
test[meta['target']] = model.predict(dtest)
subfn = "base1_eta001_val_"+str(model.best_score)+"__rnd_"+str(model.best_iteration)+".csv"
test[[meta['test_id'], meta['target']]].to_csv(subfn, index=False)

print('--------------> Retrain all data + Feature importance ... ')
dtrain = xgb.DMatrix(all_data[:train_obs].values, label=y_train)
dtest = xgb.DMatrix(all_data[train_obs:].values)
model = xgb.train(xgb_pars, dtrain, model.best_iteration+5, maximize=False, verbose_eval=10)
print('-----> Submission ... ')
test[meta['target']] = model.predict(dtest)
subfn = "base1_eta001_all_data__rnd_"+str(model.best_iteration)+".csv"
test[[meta['test_id'], meta['target']]].to_csv(subfn, index=False)

xgb_feat_importance(model=model,cols=all_data.columns,file_name="feat_importance_base1_eta001.csv")
#################################################################################


