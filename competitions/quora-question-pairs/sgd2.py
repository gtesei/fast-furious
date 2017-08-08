import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score

RS = 12357
print("Started")
np.random.seed(RS)
input_folder = './data/'

# data 
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

x_train_1 = pd.read_csv('xtrain.csv')
del x_train_1['Unnamed: 0']
x_test_1 = pd.read_csv('xtest.csv')
del x_test_1['Unnamed: 0']
print("Feature set 1: X_train: {}, X_test: {}".format(x_train_1.shape,x_test_1.shape))

x_train_2 = pd.read_csv('xtrain_2.csv') 
#del x_train_2['Unnamed: 0'] 
x_test_2 = pd.read_csv('xtest_2.csv')  
#del x_test_2['Unnamed: 0']
print("Feature set 2: X_train: {}, X_test: {}".format(x_train_2.shape, x_test_2.shape))

#x_train_3 = pd.read_csv('xtrain_3.csv') 
#x_test_3 = pd.read_csv('xtest_3.csv')  
#print("Feature set 3: X_train: {}, X_test: {}".format(x_train_3.shape, x_test_3.shape))

y_train = df_train['is_duplicate'].values

x_train = pd.concat([x_train_1,x_train_2],axis=1)
x_test = pd.concat([x_test_1,x_test_2],axis=1)
print("Merge: X_train: {}, X_test: {}".format(x_train.shape, x_test.shape))

assert x_train.shape[0] == df_train.shape[0]
assert x_test.shape[0] == df_test.shape[0]

# resample 
if 1: # Now we oversample the negative class - on your own risk of overfitting!
  pos_train = x_train[y_train == 1]
  neg_train = x_train[y_train == 0]
  print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  p = 0.165
  scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
  while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
  neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
  print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  x_train = pd.concat([pos_train, neg_train])
  y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
  del pos_train, neg_train

# classification 
x_train = x_train.fillna(value=0)
x_test = x_test.fillna(value=0)
clf1 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
clf1.fit(x_train, np.array(y_train))


#clf2 = RandomForestClassifier(n_estimators = 1000)
#clf2.fit(x_train, np.array(y_train))

p1 = clf1.predict_proba(x_test)[:, 1]
#p2 = clf2.predict_proba(x_test)[:, 1]

#preds = 0.6*p1+0.2*p2

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p1
sub.to_csv("sgd_mul.csv", index=False)

print("Done.")
