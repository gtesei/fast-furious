import lightgbm as lgb
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split 

path=Path("data/")
train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)
test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)

param = {
    'boost_from_average':'false',
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    #'num_leaves': 2**12-1,
    'n_jobs': 30,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
    }

y = train['target']
X = train.drop('target',axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.15)

trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_valid, label=y_valid)

model = lgb.train(param, trn_data, 1000000, feature_name=X_train.columns.tolist(), valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 4000)
result = model.predict(test)

## feat imp
gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':X_train.columns.tolist(),'split':model.feature_importance('split'),'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print("************ FEAT IMPORTANCE *****************")
print(ft.head(25))
print()
##
best_valid_auc = model.best_score['valid_1']['auc']
best_iteration = model.best_iteration
print("best_iteration:",best_iteration,"- best_valid_auc:",best_valid_auc )
fh = open("base_light_gbm0.log","w")
print("best_iteration_avg:",best_iteration,"- best_valid_auc:",best_valid_auc,file=fh)
fh.close()

submission = pd.read_csv(path/'sample_submission.csv')
submission['target'] = result
filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())
submission.to_csv(filename, index=False)

 




    
    
    
