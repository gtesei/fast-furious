import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
import sys
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score , roc_auc_score , log_loss
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import Stemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss
from numpy import linalg as LA
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from bs4 import BeautifulSoup
#import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import fbeta_score, make_scorer

def split_cat(text):
    try: return text.split("/")
    except: return ("Other", "Other", "Other")






train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
sample_submission = pd.read_csv("sample_submission.csv")

#train = train.iloc[:1000,:]
#test = test.iloc[:1000,:]

print(train.shape)
print(test.shape)

#
train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))

# 
y_train = np.log1p(train['price'])
train = train.drop(['price'],axis=1)
#train['category_name'] = train['category_name'].fillna('Other').astype(str)
train['name'] = train['name'].fillna('Other').astype(str)
train['general_cat'] = train['general_cat'].fillna('Other').astype(str)
train['subcat_1'] = train['subcat_1'].fillna('Other').astype(str)
train['subcat_2'] = train['subcat_2'].fillna('Other').astype(str)
train['brand_name'] = train['brand_name'].fillna('Other').astype(str)
train['shipping'] = train['shipping'].fillna('None').astype(str)
train['item_condition_id'] = train['item_condition_id'].astype(str)
train['item_description'] = train['item_description'].fillna('None').astype(str)


test['name'] = test['name'].fillna('Other').astype(str)
test['general_cat'] = test['general_cat'].fillna('Other').astype(str)
test['subcat_1'] = test['subcat_1'].fillna('Other').astype(str)
test['subcat_2'] = test['subcat_2'].fillna('Other').astype(str)
test['brand_name'] = test['brand_name'].fillna('Other').astype(str)
test['shipping'] = test['shipping'].fillna('None').astype(str)
test['item_condition_id'] = test['item_condition_id'].astype(str)
test['item_description'] = test['item_description'].fillna('None').astype(str)
all_data = pd.concat([train,test],axis=0)
# we need a custom pre-processor to extract correct field,
# but want to also use default scikit-learn preprocessing (e.g. lowercasing)
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(all_data.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('general_cat', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('general_cat'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])
all_data_train = vectorizer.fit_transform(all_data.values)

train_obs = train.shape[0]
Xtrain = all_data_train[:train_obs,:]
Xtest = all_data_train[train_obs:,:]
assert Xtrain.shape[0] == train.shape[0]
assert Xtest.shape[0] == test.shape[0]

del train, test, all_data_train 



#def my_rmsle(y_true, y_pred):
#	    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))
	    #return np.sqrt(mean_squared_log_error(y_true, y_pred))
	#rmsle_scorer = make_scorer(my_rmsle)

def my_rmsle(y_true,y_pred):
   #y_true = np.expm1(y_true)
   #y_pred = np.expm1(y_pred)
   assert len(y_true) == len(y_pred)
   return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true), 2)))
   #return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


## Whether score_func is a score function (default), meaning high is good, or a loss function, meaning low is good. 
## In the latter case, the scorer object will sign-flip the outcome of the score_func.
rmsle_scorer = make_scorer(my_rmsle, greater_is_better=False) 	

model = Ridge(alpha=0.5)
grid_clf = GridSearchCV(estimator=model, param_grid={'alpha': [0.1,0.3,0.5,0,8,1,1.2,5,7,10,50],'fit_intercept': [True, False]},cv=4, scoring=rmsle_scorer, n_jobs=10, iid=False)

print(np.expm1(y_train).head())
grid_clf.fit(Xtrain, y_train)
print("best params:",grid_clf.best_params_)

print(grid_clf.cv_results_['mean_test_score'])

best_idx = grid_clf.best_index_ 
print("best rmsle:", -1*grid_clf.cv_results_['mean_test_score'][best_idx] , " - std:",grid_clf.cv_results_['std_test_score'][best_idx])
sys.stdout.flush()

grid = pd.DataFrame({'alpha' : [grid_clf.best_params_['alpha']], 
	'fit_intercept': [grid_clf.best_params_['fit_intercept']],
	'best_rmsle': [-1*grid_clf.cv_results_['mean_test_score'][best_idx]] , 
	'best_rmsle_std': [grid_clf.cv_results_['std_test_score'][best_idx]]})

grid.to_csv('best_rmsle_ridge.csv',index=False)

print(">> predicting on test set ... ")
pred_test = grid_clf.predict(Xtest)
sample_submission['price'] = np.expm1(pred_test) 
print(">> sample submission")
print(sample_submission.head())
grid.to_csv('ridge_sub.csv.gz',index=False,compression='gzip')











