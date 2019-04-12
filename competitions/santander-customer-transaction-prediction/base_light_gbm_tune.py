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
#test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)

#train = train[:200]

y = train['target']
X = train.drop('target',axis=1)

trn_data = lgb.Dataset(X, label=y)

##
auc_cv_mean = []
auc_cv_std= []
nround_p = []

num_leaves_list = []
learning_rate_list = []
feature_fraction_list = []
bagging_fraction_list = []
max_bin_list = []
max_depth_list = []
bagging_freq_list = []
min_data_in_leaf_list = []
min_sum_hessian_in_leaf_list = [] 

i = 1
for num_leaves in [13,10,20]:
    for learning_rate in [0.0083,0.005]:
        for feature_fraction in [0.041,0.03,0.05]:
            for bagging_fraction in [0.335,0.5,0.25]:
                for max_bin in [100,255]:
                    for max_depth in [-1,10]:
                        for bagging_freq in [0,1,5]:
                            for min_data_in_leaf in [80,50,100]:
                                for min_sum_hessian_in_leaf in [10.0,8.0,13.0]:
                                    param = {
                                        #'device': 'gpu', 
                                        'boost_from_average':'false',
                                        'bagging_fraction': bagging_fraction,
                                        'boost_from_average':'false',
                                        'max_bin': max_bin, 
                                        'boost': 'gbdt',
                                        'bagging_freq': bagging_freq, 
                                        'feature_fraction': feature_fraction,
                                        'learning_rate': learning_rate,
                                        'max_depth': max_depth,
                                        'metric':'auc',
                                        'min_data_in_leaf': min_data_in_leaf,
                                        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                                        'num_leaves': num_leaves,
                                        'n_jobs': 30,
                                        'tree_learner': 'serial',
                                        'objective': 'binary',
                                        'verbosity': -1
                                    }
                                    print(">>>>",i,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                                    print(param)
                                    model = lgb.cv(param, trn_data, 1000000,
                                               feature_name=X.columns.tolist(),
                                               verbose_eval=500,
                                               early_stopping_rounds = 4000,
                                               nfold=4)
                                    num_leaves_list.append(num_leaves)
                                    learning_rate_list.append(learning_rate)
                                    feature_fraction_list.append(feature_fraction)
                                    bagging_fraction_list.append(bagging_fraction)
                                    max_bin_list.append(max_bin)
                                    max_depth_list.append(max_depth)
                                    bagging_freq_list.append(bagging_freq)
                                    min_data_in_leaf_list.append(min_data_in_leaf)
                                    min_sum_hessian_in_leaf_list.append(min_sum_hessian_in_leaf)
                                    nround_p.append(len(model['auc-mean']))
                                    auc_cv_mean.append(model['auc-mean'][len(model['auc-mean'])-1])
                                    auc_cv_std.append(model['auc-stdv'][len(model['auc-mean'])-1])
                                    i = i + 1
                                    grid = pd.DataFrame({
                                        'num_leaves' : num_leaves_list,
                                        'learning_rate' : learning_rate_list,
                                        'feature_fraction' : feature_fraction_list,
                                        'bagging_fraction' : bagging_fraction_list,
                                        'max_bin' : max_bin_list,
                                        'max_depth' : max_depth_list,
                                        'bagging_freq' : bagging_freq_list,
                                        'min_data_in_leaf' : min_data_in_leaf_list,
                                        'min_sum_hessian_in_leaf' : min_sum_hessian_in_leaf_list,
                                        'nround' : nround_p,
                                        'auc_cv_mean' : auc_cv_mean,
                                        'auc_cv_std' : auc_cv_std
                                    })
                                    grid.to_csv('base_grid_lightGBM.csv',index=False)

                                        
                                        
                                        
                                    

                                    
                                   



model['auc-mean'][len(model['rmse-mean'])-10:]

 




    
    
    
