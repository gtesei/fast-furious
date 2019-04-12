import lightgbm as lgb
import pandas as pd
import numpy as np

import os 
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

## config 
SEED = 199
TOP_OK = 2

## data 
print(">> loading dataset ... ")
path=Path("data/")
train=pd.read_csv(path/"train.csv")
#train = train[:100]
train_ID_code = train["ID_code"].tolist()
train=train.drop("ID_code",axis=1)

test=pd.read_csv(path/"test.csv")
test_ID_code = test["ID_code"].tolist()
test=test.drop("ID_code",axis=1)

##
print(">> loading cross-validation results ... ")
cv_results=pd.read_csv("base_grid_lightGBM_4.csv")
cv_results = cv_results.sort_values(by='auc_cv_mean' , ascending=False)
cv_results = cv_results[:TOP_OK]
cv_results.reset_index(inplace=True,drop=True)
print(cv_results.head())

##
for i in range(TOP_OK):
    model_name = 'light_gbm_'+format(cv_results.iloc[i]['auc_cv_mean'],'.6')
    print(i,model_name,cv_results.iloc[i])
    valid_df = pd.DataFrame({"ID_code": train_ID_code , 'target':-1})
    result=np.zeros(test.shape[0])
    #
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=SEED)
    for counter,(train_index, valid_index) in enumerate(rskf.split(train, train.target),1):
        print ("fold:",counter)
        #print("*** train_index ***")
        #print(train_index)
        #print("*** valid_index ***")
        #print(valid_index)    
        param = {
            'boost_from_average':'false',
            'bagging_fraction': cv_results.iloc[i]['bagging_fraction'], 
            'boost_from_average':'false',
            #'max_bin': int(cv_results.iloc[i]['max_bin']) , 
            'boost': 'gbdt',
            'bagging_freq': int(cv_results.iloc[i]['bagging_freq']) , 
            'feature_fraction': cv_results.iloc[i]['feature_fraction']  ,
            'learning_rate': cv_results.iloc[i]['learning_rate']  ,
            'max_depth': int(cv_results.iloc[i]['max_depth'])  ,
            'metric':'auc',
            'min_data_in_leaf': int(cv_results.iloc[i]['min_data_in_leaf'])  ,
            'min_sum_hessian_in_leaf': cv_results.iloc[i]['min_sum_hessian_in_leaf'] ,
            'num_leaves': int(cv_results.iloc[i]['num_leaves']) ,
            'n_jobs': 30,
            'tree_learner': 'serial',
            'objective': 'binary',
            'verbosity': -1
        }
        #Train data
        t=train.iloc[train_index]
        trn_data = lgb.Dataset(t.drop("target",axis=1), label=t.target)
        #
        model = lgb.train(param, trn_data, int(cv_results.iloc[i]['nround']), 
                          feature_name=train.columns.tolist()[1:], 
                          verbose_eval=500)
        valid_df.loc[valid_index , 'target'] = model.predict(train.iloc[valid_index].drop("target",axis=1))
        result += model.predict(test)
    print(">> ensembling")
    assert np.sum(valid_df['target'] < 0) == 0 
    assert np.sum(result < 0) == 0 
    ##
    os.makedirs('ensemb/'+model_name)
    valid_df.to_csv('ensemb/'+model_name+'/train_pred.csv',index=False)
    submission = pd.read_csv(path/'sample_submission.csv')
    submission['target'] = result/counter
    submission.to_csv('ensemb/'+model_name+'/submission.csv', index=False)
    
