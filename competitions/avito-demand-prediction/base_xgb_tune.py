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
# avg_price_usr
#all_data = add_avg_per(df=all_data,what_to_avg='price',on='user_id',new_name='avg_price_usr',include_delta=True,include_perc=True)
# avg_price_cat
#all_data = add_avg_per(df=all_data,what_to_avg='price',on='category_name',new_name='avg_price_cat',include_delta=True,include_perc=True)
# avg_price_usr_cat 
#all_data = add_avg_per(df=all_data,what_to_avg='price',on=['user_id','category_name'],new_name='avg_price_usr_cat',include_delta=True,include_perc=True)
# avg_price_city_cat
#all_data = add_avg_per(df=all_data,what_to_avg='price',on=['city','category_name'],new_name='avg_price_city_cat',include_delta=True,include_perc=True)
# avg_price_region_cat
#all_data = add_avg_per(df=all_data,what_to_avg='price',on=['region','category_name'],new_name='avg_price_region_cat',include_delta=True,include_perc=True)

#for f in ['activation_date_is_holiday']:
#    all_data = all_data.drop(f,axis=1)
print(all_data.head())
print(">>>>>>> shape:",all_data.shape)
#################################################################################

### MODELING ####################################################################
print('--------------> Modeling ... ')
train_obs = len(y_train)
Xtr, Xv, ytr, yv = train_test_split(all_data[:train_obs].values, y_train, test_size=0.6, random_state=1973)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(all_data[train_obs:].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#Try different parameters! My favorite is random search :)

min_child_weight_p = []
eta_p = []
colsample_bytree_p = []
max_depth_p = []
subsample_p = []
lambda_p = []
nround_p = []
rmse_cv_mean = []
rmse_cv_std = []

i = 1
for min_child_weight in [0,0.5,1,15,50]:
    for eta in [0.01,0.005]:
        for colsample_bytree in [0.5,0.7]:
            for max_depth in [6,15]:
                for subsample in [0.5,0.7]:
                    for lambdaa in [0.5,1]:
                        xgb_pars = {'min_child_weight': min_child_weight,
                                    'eta': eta,
                                    'colsample_bytree': colsample_bytree,
                                    'max_depth': max_depth,
                                    'subsample': subsample,
                                    'lambda': lambdaa,
                                    'nthread': -1,
                                    'booster' : 'gbtree',
                                    'silent': 1,
                                    'eval_metric': 'rmse',
                                    'objective': 'reg:linear'}
                        print(">>>>",i,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        print(xgb_pars)
                        model = xgb.cv(xgb_pars, dtrain, 10000,nfold = 4, early_stopping_rounds=50,maximize=False, verbose_eval=10)
                        min_child_weight_p.append(min_child_weight)
                        eta_p.append(eta)
                        colsample_bytree_p.append(colsample_bytree)
                        max_depth_p.append(max_depth)
                        subsample_p.append(subsample)
                        lambda_p.append(lambdaa)
                        nround_p.append(model.shape[0])
                        rmse_cv_mean.append(model['test-rmse-mean'][model.shape[0]-1])
                        rmse_cv_std.append(model['test-rmse-std'][model.shape[0]-1])
                        i = i + 1
                        grid = pd.DataFrame({
                                            'min_child_weight': min_child_weight_p,
                                            'eta': eta_p,
                                            'colsample_bytree': colsample_bytree_p,
                                            'max_depth': max_depth_p,
                                            'subsample': subsample_p,
                                            'lambda': lambda_p,
                                            'nround': nround_p,
                                            'rmse_cv_mean': rmse_cv_mean,
                                            'rmse_cv_std': rmse_cv_std,
                        })
                        grid.to_csv('base_grid_xgb_40perc.csv',index=False)
#################################################################################


