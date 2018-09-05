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

from multiprocessing import Pool, Lock
import pandas as pd 
import numpy as np 
import os 
import glob

global grid 

### FUNC ########################################################################
def create_grid():  
    #
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
                            #print(">>>>",i,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                            min_child_weight_p.append(min_child_weight)
                            eta_p.append(eta)
                            colsample_bytree_p.append(colsample_bytree)
                            max_depth_p.append(max_depth)
                            subsample_p.append(subsample)
                            lambda_p.append(lambdaa)
                            nround_p.append(-1)
                            rmse_cv_mean.append(-1)
                            rmse_cv_std.append(-1)
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
                        'nround': nround_p
                        })
    grid.index = range(len(grid))
    print("Grid:",str(grid.shape))
    print(grid.head())
    #grid.to_csv('base_grid_xgb_40perc.csv',index=False)
    return grid 


grid = create_grid()


def do_compute(x):  
    row = grid.iloc[x,:]   
    eta = row['eta']
    min_child_weight = row['min_child_weight']
    colsample_bytree = row['colsample_bytree']
    max_depth = row['max_depth']
    subsample = row['subsample']
    _lambda = row['lambda']
    nround = row['nround']
    ####
    xgb_pars = {'min_child_weight': min_child_weight,
                'eta': eta,
                'colsample_bytree': colsample_bytree,
                'max_depth': int(max_depth),
                'subsample': subsample,
                'lambda': _lambda,
                'nthread': -1,
                'booster' : 'gbtree',
                'silent': 1,
                'eval_metric': 'rmse',
                'objective': 'reg:linear'}
    #print(xgb_pars)
    model = xgb.cv(xgb_pars, dtrain, 100000,nfold = 4, early_stopping_rounds=50,maximize=False, verbose_eval=10)
    nround = model.shape[0]
    rmse_cv_mean = model['test-rmse-mean'][model.shape[0]-1]
    rmse_cv_std = model['test-rmse-std'][model.shape[0]-1]
    # calculate the square of the value of x
    grid.loc[x,'rmse_cv_mean'] = rmse_cv_mean
    grid.loc[x,'rmse_cv_std'] = rmse_cv_std
    grid.loc[x,'nround'] = nround
    grid.to_csv('base_grid_xgb_40perc__'+str(os.getpid())+'.csv',index=False)
    return rmse_cv_mean
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


#################################################################################

if __name__ == '__main__':
    
    #print("grid created")
    #print(grid.head())

    # Define the dataset
    dataset = range(len(grid))
    agents = 4
    chunksize = int(len(grid)/agents)
    # Output the dataset
    #print ('Dataset: ' , str(dataset) , "chunksize:",str(chunksize))


    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    
    with Pool(processes=agents) as pool:
        result = pool.map(do_compute, dataset, chunksize)

    # Output the result
    print ('Result:  ' + str(result) , "---type:",type(result))
    #grid.to_csv('base_grid_xgb_40perc.csv',index=False)

    print(">>> merge ...")
    agrid = create_grid()
    listing = glob.glob('./base_grid_xgb_40perc__*')
    print(listing)
    for filename in listing:
        print(filename)
        gg = pd.read_csv(filename)
        gg = gg[gg.rmse_cv_mean >=0]
        print(gg.index)
        for i in (gg.index):
            row = gg.loc[i,:] 
            rmse_cv_mean = row['rmse_cv_mean']
            rmse_cv_std = row['rmse_cv_std']
            nround = row['nround']
            agrid.loc[i,'rmse_cv_mean'] = rmse_cv_mean
            agrid.loc[i,'rmse_cv_std'] = rmse_cv_std
            agrid.loc[i,'nround'] = nround
    agrid.to_csv('base_grid_xgb_40perc.csv',index=False)


