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

print(">> loading training set ... ")
y = train['target']
X = train.drop('target',axis=1)

y_1 = np.sum(y == 1)
y_0 = np.sum(y == 0)

assert len(y) == y_1 + y_0 

print(">train size:",len(y),"  0s:",y_0,"  1s:",y_1)
scale_pos_weight_train = y_0 / y_1
print(">scale_pos_weight_train:",scale_pos_weight_train)

trn_data = lgb.Dataset(X, label=y)

##
auc_cv_mean = []
auc_cv_std= []
nround_p = []

num_leaves_list = []
learning_rate_list = []
feature_fraction_list = []
bagging_fraction_list = []
#max_bin_list = []
max_depth_list = []
bagging_freq_list = []
min_data_in_leaf_list = []
min_sum_hessian_in_leaf_list = []
scale_pos_weight_list = [] 

i = 1
for num_leaves in [13]:
    for learning_rate in [0.002,0.001]:
        for feature_fraction in [0.02]:
            for bagging_fraction in [0.5]:
                for scale_pos_weight in [1.0]:
                    for max_depth in [4,6]:
                        for bagging_freq in [5]:
                            for min_data_in_leaf in [100]:
                                for min_sum_hessian_in_leaf in [10.0]:
                                    param = {
                                        #'device': 'gpu', 
                                        'boost_from_average':'false',
                                        'bagging_fraction': bagging_fraction,
                                        'boost_from_average':'false',
                                        #'max_bin': max_bin,
                                        'scale_pos_weight': scale_pos_weight, 
                                        'boost': 'gbdt',
                                        'bagging_freq': bagging_freq, 
                                        'feature_fraction': feature_fraction,
                                        'learning_rate': learning_rate,
                                        'max_depth': max_depth,
                                        'metric':'auc',
                                        'min_data_in_leaf': min_data_in_leaf,
                                        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                                        'num_leaves': num_leaves,
                                        'n_jobs': 20,
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
                                    #max_bin_list.append(max_bin)
                                    scale_pos_weight_list.append(scale_pos_weight)
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
                                        #'max_bin' : max_bin_list,
                                        'scale_pos_weight' : scale_pos_weight_list, 
                                        'max_depth' : max_depth_list,
                                        'bagging_freq' : bagging_freq_list,
                                        'min_data_in_leaf' : min_data_in_leaf_list,
                                        'min_sum_hessian_in_leaf' : min_sum_hessian_in_leaf_list,
                                        'nround' : nround_p,
                                        'auc_cv_mean' : auc_cv_mean,
                                        'auc_cv_std' : auc_cv_std
                                    })
                                    grid.to_csv('base_grid_lightGBM_4.csv',index=False)

                                        
                                        
print(">>>>>>>>>>>>>>>>>>>> END <<<<<<<<<<<<<<<<<<<<<<<")

 




    
    
    
