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
import lightgbm as lgb 
from sklearn.metrics import mean_squared_error

from extract_feat_base import * 

### FUNC ########################################################################

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
all_data , y_train = encode_dataset(train=train,test=test,meta=meta,target_model='lightgbm')
print(all_data.head())
print(">>>>>>> shape:",all_data.shape)
print('--------------> Advanced Feature Engineering ... ')
#for f in ['activation_date_is_holiday']:
#    all_data = all_data.drop(f,axis=1)
print(all_data.head())
print(">>>>>>> shape:",all_data.shape)

#categorical_features = []
#for f in meta['cols'].keys():
#    if meta['cols'][f] == 'CAT':
#        for i,col in enumerate(all_data.columns.tolist()):
#            if col == f:
#                categorical_features.append(i)
#
#print(">>> categorical features:",categorical_features)
#################################################################################

### MODELING ####################################################################
print('--------------> Modeling ... ')
train_obs = len(y_train)
Xtr, Xv, ytr, yv = train_test_split(all_data[:train_obs].values, y_train, test_size=0.1, random_state=1973)
# create dataset for lightgbm
lgb_train = lgb.Dataset(Xtr, ytr)
lgb_eval = lgb.Dataset(Xv, yv, reference=lgb_train)
#lgb_test = lgb.Dataset(all_data[train_obs:].values)

# specify your configurations as a dict
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31, #31
        'learning_rate': 0.01, #0.01
        'feature_fraction': 0.7, #0.7, #0.9 
        'bagging_fraction': 0.95, #0.95, #0.8
        'max_bin': 100, #100, # 255
        'max_depth': -1, #5, # -1 
        'bagging_freq': 1, #1, # 5 
        'verbose': 0
    }


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100000,
                valid_sets=lgb_eval,
                feature_name=all_data.columns.tolist(),
                #categorical_feature=categorical_features,
                early_stopping_rounds=50)

print('Save model...')
# save model to file
gbm.save_model('model_lightgbm.txt')

print('Start predicting...')
# predict
y_pred_eval = gbm.predict(Xv, num_iteration=gbm.best_iteration)
# eval
rmse_eval = mean_squared_error(yv, y_pred_eval) ** 0.5
print('The rmse of prediction is:', rmse_eval)

print('--------------> Submission ... ')
pred = gbm.predict(all_data[train_obs:].values, num_iteration=gbm.best_iteration)
pred = (pred<0)*0+(pred>=0)*pred
pred = (pred<1)*pred+(pred>=1)*1
test[meta['target']] = pred
subfn = "base_lightgbm_eta001_val_"+str(rmse_eval)+"__rnd_"+str(gbm.best_iteration)+".csv"
test[[meta['test_id'], meta['target']]].to_csv(subfn, index=False)

print('--------------> Retrain all data + Feature importance ... ')
lgb_train = lgb.Dataset(all_data[:train_obs].values,y_train)
model = lgb.train(params, lgb_train, gbm.best_iteration+5,
                  feature_name=all_data.columns.tolist() ,
                  #categorical_feature=categorical_features
                  )
print('-----> Submission ... ')
pred = model.predict(all_data[train_obs:].values,num_iteration=int(gbm.best_iteration*10/9))
pred = (pred<0)*0+(pred>=0)*pred
pred = (pred<1)*pred+(pred>=1)*1
test[meta['target']] = pred 
subfn = "base_lightgbm_tuned__rnd_"+str(int((gbm.best_iteration*10/9)))+".csv"
test[[meta['test_id'], meta['target']]].to_csv(subfn, index=False)

gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':all_data.columns.tolist(),
                   'split':model.feature_importance('split'),
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(25))
ft.to_csv('base_lightgbm_feat_importance_tuned.csv', index=False)
print("Done.")
#################################################################################


