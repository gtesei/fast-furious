import lightgbm as lgb
import pandas as pd
import numpy as np
import sys 
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

path=Path("data/")
train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)
test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)

param = {
    'boost_from_average':'false',
    'bagging_fraction': 0.5,
    'boost': 'gbdt',
    'feature_fraction': 0.02,
    'learning_rate': 0.001,
    'max_depth': 6,
    'metric':'auc',
    'min_data_in_leaf': 100,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'n_jobs': 30,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
    }

result=np.zeros(test.shape[0])

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=10)
best_iteration , best_valid_auc = 0, 0 
for counter,(train_index, valid_index) in enumerate(rskf.split(train, train.target),1):
    print ("Rep-Fold:",counter)
    sys.stdout.flush()
    #Train data
    t=train.iloc[train_index]
    trn_data = lgb.Dataset(t.drop("target",axis=1), label=t.target)
    #Validation data
    v=train.iloc[valid_index]
    val_data = lgb.Dataset(v.drop("target",axis=1), label=v.target)
    #Training
    model = lgb.train(param, trn_data, 1000000, feature_name=train.columns.tolist()[1:], valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 4000)
    result +=model.predict(test)
    ## feat imp
    gain = model.feature_importance('gain')
    ft = pd.DataFrame({'feature':train.columns.tolist()[1:],'split':model.feature_importance('split'),'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print("************ FEAT IMPORTANCE *****************")
    print(ft.head(25))
    print()
    ##
    _best_valid_auc = model.best_score['valid_1']['auc']
    _best_iteration = model.best_iteration
    print("best_iteration:",_best_iteration,"- best_valid_auc:",_best_valid_auc )
    best_valid_auc +=_best_valid_auc
    best_iteration += _best_iteration

submission = pd.read_csv(path/'sample_submission.csv')
submission['target'] = result/counter
filename="{:%Y-%m-%d_%H_%M}_sub_after_tune.csv".format(datetime.now())
submission.to_csv(filename, index=False)

## feat importance
best_valid_auc = best_valid_auc/counter
best_iteration = best_iteration/counter
fh = open("base_light_gbm1.log","w")
print("best_iteration_avg:",best_iteration,"- best_valid_auc_avg:",best_valid_auc,file=fh)
fh.close()
 




    
    
    
