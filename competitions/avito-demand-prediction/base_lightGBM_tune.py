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
all_data , y_train = encode_dataset(train=train,test=test,meta=meta)
print(all_data.head())
print(">>>>>>> shape:",all_data.shape)
print('--------------> Advanced Feature Engineering ... ')

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
Xtr, Xv, ytr, yv = train_test_split(all_data[:train_obs].values, y_train, test_size=0.6, random_state=1973)
# create dataset for lightgbm
lgb_train = lgb.Dataset(Xtr, ytr)
lgb_eval = lgb.Dataset(Xv, yv, reference=lgb_train)

rmse_cv_mean = []
rmse_cv_std= []
nround_p = []

num_leaves_list = []
learning_rate_list = []
feature_fraction_list = []
bagging_fraction_list = []
max_bin_list = []
max_depth_list = []
bagging_freq_list = []

i = 1
for num_leaves in [31,15]:
    for learning_rate in [0.01,0.005]:
        for feature_fraction in [0.7,0.9,1]:
            for bagging_fraction in [0.8,0.95,1]:
                for max_bin in [100,255]:
                    for max_depth in [-1,5,10]:
                        for bagging_freq in [0,1,5]:
                            params = {
                                        'task': 'train',
                                        'boosting_type': 'gbdt',
                                        'objective': 'regression',
                                        'metric': 'rmse',
                                        'num_leaves': num_leaves,
                                        'learning_rate': learning_rate,
                                        'feature_fraction': feature_fraction,
                                        'bagging_fraction': bagging_fraction,
                                        'max_bin': max_bin,
                                        'max_depth': max_depth,
                                        'bagging_freq': bagging_freq,
                                        'verbose': 0
                                    }
                            print(">>>>",i,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                            print(params)
                            model = lgb.cv(params,
                                           lgb_train,
                                           num_boost_round=100000,
                                           early_stopping_rounds=50,
                                           feature_name=all_data.columns.tolist(),
                                           nfold=4,
                                           stratified=False)
                            num_leaves_list.append(num_leaves)
                            learning_rate_list.append(learning_rate)
                            feature_fraction_list.append(feature_fraction)
                            bagging_fraction_list.append(bagging_fraction)
                            max_bin_list.append(max_bin)
                            max_depth_list.append(max_depth)
                            bagging_freq_list.append(bagging_freq)
                            nround_p.append(len(model['rmse-mean']))
                            rmse_cv_mean.append(model['rmse-mean'][len(model['rmse-mean'])-1])
                            rmse_cv_std.append(model['rmse-stdv'][len(model['rmse-mean'])-1])
                            i = i + 1
                            grid = pd.DataFrame({
                                                'num_leaves': num_leaves_list,
                                                'learning_rate': learning_rate_list,
                                                'feature_fraction': feature_fraction_list,
                                                'bagging_fraction': bagging_fraction_list,
                                                'max_bin': max_bin_list,
                                                'max_depth': max_depth_list,
                                                'bagging_freq': bagging_freq_list,
                                                'nround': nround_p,
                                                'rmse_cv_mean': rmse_cv_mean,
                                                'rmse_cv_std': rmse_cv_std,
                            })
                            grid.to_csv('base_grid_lightGBM_60perc.csv',index=False)
#################################################################################


