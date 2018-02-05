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
import datetime as dt
from sklearn.cross_validation import train_test_split
import xgboost as xgb

# StemmedTfidfVectorizer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))


def clean_text( text ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. lowercase
    len_txt = len(text)
    upper_perc = sum(1 for c in text if c.isupper())/len(text)
    text = text.lower()
    #
    # 2. Remove non-letters
    text = re.sub(r"\"\"\"\"", " _Q_ ", text)
    text = re.sub(r"\"\"\"", " _Q_ ", text)
    text = re.sub(r"\"\"", " _Q_ ", text)
    text = re.sub(r"\.\.\.\.", " _P_ ", text)
    text = re.sub(r"\.\.\.", " _P_ ", text)
    text = re.sub(r"!", " _exclamationmark_ ", text)
    text = re.sub(r"\?", " _questionmark_ ", text)
    text = re.sub(r"f u c k   o f f", " fuck off ", text)
    #
    text = re.sub("[^A-za-z0-9^,?!.\/'+-=]"," ", text)
    text = re.sub(r"what's", " what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", " i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"f\*cking", "  fuck ", text)
    text = re.sub(r"prost!tute", " prostitute ", text)
    text = re.sub(r"motherfuckin", " mother fuck ", text)
    text = re.sub(r"fucker", " fuck ", text)
    #
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    #
    return text , upper_perc, len_txt


def build_data_set(ngram=3,stem=False,max_features=2000,min_df=2,remove_stopwords=True):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test.fillna('__NA__',inplace=True)
    ##
    Y_train = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    ## 
    clean_train_comments = []
    upper_percs = []
    lens_txt = [] 
    for i in range(train.shape[0]):
        ttext , upper_perc, len_txt = clean_text(train["comment_text"][i])
        lens_txt.append( len_txt )
        upper_percs.append( upper_perc )
        clean_train_comments.append( ttext )
    #print(">>> processing test set ...")
    train_obs = train.shape[0]
    del train
    for i in range(test.shape[0]):
        ttext , upper_perc, len_txt = clean_text(test["comment_text"][i])
        lens_txt.append( len_txt )
        upper_percs.append( upper_perc )
        clean_train_comments.append( ttext )
    del test 
    qs = pd.Series(clean_train_comments).astype(str)
    del clean_train_comments
    upper_percs_arr = np.array(upper_percs)
    lens_txt_arr = np.array(lens_txt)
    del upper_percs,lens_txt 
    if not stem:
        # 1-gram / no-stem
        vect = TfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect = vect.fit_transform(qs)
        #print("ifidf_vect:", ifidf_vect.shape)
        X = ifidf_vect.toarray()
        X = np.hstack([X,upper_percs_arr.reshape((X.shape[0],1)), lens_txt_arr.reshape((X.shape[0],1)), ])
        del upper_percs_arr, lens_txt_arr
        X_train = X[:train_obs]
        X_test = X[train_obs:]
    else:
        vect_stem = StemmedTfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect_stem = vect_stem.fit_transform(qs)
        #print("ifidf_vect_stem:", ifidf_vect_stem.shape)
        X = ifidf_vect_stem.toarray()
        X = np.hstack([X,upper_percs_arr.reshape((X.shape[0],1)), lens_txt_arr.reshape((X.shape[0],1)), ])
        del upper_percs_arr, lens_txt_arr
        X_train = X[:train_obs]
        X_test = X[train_obs:]
    assert Y_train.shape[0] == X_train.shape[0]
    return X_train,X_test,Y_train


#--------------------------- Main()
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
params = {
    'toxic': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } , 
    'threat': {'ngrams': 1, 'stem': False, 'max_features': 10000, 'C': 10 } , 
    'severe_toxic': {'ngrams': 1, 'stem': True, 'max_features': 5000, 'C': 1.2 } , 
    'obscene': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } , 
    'insult': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 1.2 } , 
    'identity_hate': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } 
}

# XGB
RS = 123
ROUNDS = 100000
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['eval_metric'] = 'logloss'
xgb_params['eta'] = 0.01
xgb_params['silent'] = 1
xgb_params['seed'] = RS

sample_submission = pd.read_csv("baseline_linear.csv")

# proc
t0 = dt.datetime.now()

for label in labels:
    print(">>> processing ",label)
    X_train,X_test,Y_train = build_data_set(ngram=params[label]['ngrams'],
                                            stem=params[label]['stem'],
                                            #max_features=params[label]['max_features'],
                                            max_features=5000,
                                            min_df=2,remove_stopwords=True)
    Y_train_lab = Y_train[label]
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    xtrain, xval, ytrain, yval = train_test_split(X_train, Y_train_lab, test_size=0.2, random_state=RS)
    xg_train = xgb.DMatrix(xtrain, label=ytrain)
    xg_val = xgb.DMatrix(xval, label=yval)
    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    clf = xgb.train(params=xgb_params,dtrain=xg_train,num_boost_round=ROUNDS,early_stopping_rounds=200,evals=watchlist)
    pred_proba = clf.predict(xgb.DMatrix(X_test))
    sample_submission[label] = pred_proba
    sample_submission.to_csv("baseline_xgb.csv.gz", index=False, compression='gzip')
    del X_train, Y_train_lab, X_test, xtrain, xval, ytrain, yval, xg_train, xg_val, watchlist

t1 = dt.datetime.now()
print('Total time: %i seconds' % (t1 - t0).seconds)
 
