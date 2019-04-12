import pandas as pd
import numpy as np

import os 
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys


# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
# def auc_roc_old(y_true, y_pred):
#      # any tensorflow metric
#     value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
#     # find all variables created for this metric
#     metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value


def auc_roc(y_true,y_pred):
    value, update_op = tf.metrics.auc(predictions=y_pred, labels=y_true,num_thresholds=1000)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    #print(metric_vars)
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value    
    
    
def create_model(init_dim,n0,n1,n2,act='relu'):
    model_name = "ffnn_"+str(n0)+"_"+str(n1)+"_"+str(n2)+"_"+str(act)+""
    model = Sequential()
    model.add(Dense(n0, input_dim=init_dim, activation=act))
    model.add(Dense(n1, activation=act))
    model.add(Dense(n2, activation=act))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=[auc_roc])
    # Fit the model
    return model , model_name

## config 
SEED = 199

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
valid_df = pd.DataFrame({"ID_code": train_ID_code , 'target':-1})
result=np.zeros(test.shape[0])
#
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=SEED)
for counter,(train_index, valid_index) in enumerate(rskf.split(train, train.target),1):
    K.clear_session()
    model = None # Clearing the NN.
    model , model_name = create_model(init_dim=200,n0=200,n1=100,n2=50,act='relu')
    print ("fold:",counter, "   -- model name:",model_name)
    sys.stdout.flush()
    #Train data
    t=train.iloc[train_index]
    v = train.iloc[valid_index]
    early_stopping = EarlyStopping(monitor='val_auc_roc', patience=2 , mode='max')
    model_path = model_name + '.h5'
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_auc_roc'  , mode='max', save_best_only=True, verbose=1)
    results = model.fit(t.drop("target",axis=1),
                        t.target,
                        validation_data=(v.drop("target",axis=1),v.target),
                        epochs=20000, batch_size=2048,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    _auc_roc_score = results.history['auc_roc']
    _val_auc_roc_score = results.history['val_auc_roc']
    _val_auc_roc_score_best = max(_val_auc_roc_score)
    nround_best = sorted(range(len(_val_auc_roc_score)), key=lambda i: _val_auc_roc_score[i])[-1] + 1 
    print("> val_auc_roc_score_best:",_val_auc_roc_score_best," - nround_best:",nround_best,'  - len(val_auc_roc_score)',str(len(_val_auc_roc_score)))
    model.load_weights(model_path)
    valid_df.loc[valid_index , 'target'] = model.predict(train.iloc[valid_index].drop("target",axis=1), batch_size=2048)
    auc_valid = roc_auc_score(y_true=v.target, y_score=valid_df.loc[valid_index , 'target'])
    print(">> Validation - auc_valid:",auc_valid)
    result += model.predict(test, batch_size=2048).reshape((test.shape[0]))
    raise Exception("boom")
    
print(">> ensembling")
assert np.sum(valid_df['target'] < 0) == 0 
assert np.sum(result < 0) == 0 
##
os.makedirs('ensemb/'+model_name)
valid_df.to_csv('ensemb/'+model_name+'/train_pred.csv',index=False)
submission = pd.read_csv(path/'sample_submission.csv')
submission['target'] = result/counter
submission.to_csv('ensemb/'+model_name+'/submission.csv', index=False)
    
