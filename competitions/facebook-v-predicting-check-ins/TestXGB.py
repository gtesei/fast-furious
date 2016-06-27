import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

X = np.random.rand(1000,5)
y = np.random.random_integers(low=0, high=5, size=1000) * 100

X_val = np.random.rand(100,5)
y_val = np.random.random_integers(low=0, high=5, size=100) * 100

X_test = np.random.rand(100,5)
y_test = np.random.random_integers(low=0, high=5, size=100) * 100

le = LabelEncoder()
_y = le.fit_transform(y)

xg_train = xgb.DMatrix(X, label=y)
xg_val = xgb.DMatrix(X_val, label=y_val)
xg_test = xgb.DMatrix(X_test)

watchlist  = [ (xg_train,'train'),(xg_val,'eval')]

#### setup parameters for xgboost
param = {}
# use softmax multi-class classification
#param['objective'] = 'multi:softmax'
param['objective'] = 'multi:softprob'
param["eta"] = 0.1
param["min_child_weight"] = 10
param["subsample"] = 0.8
param["colsample_bytree"] = 0.8
param["scale_pos_weight"] = 1.0
param["silent"] = 1
param["max_depth"] = 7
param["nthread"] = 4
param['num_class'] = 6
param['eval_metric'] = 'merror'

params = list(param.items())

#### train
num_rounds = 20000
model = xgb.train(params,
                  xg_train,
                  num_rounds,
                  watchlist,
                  #feval=mapk,
                  early_stopping_rounds=50)

pred = model.predict(xg_test)
pred_label = np.argsort(pred, axis=1)[:, ::-1][:, :3]



