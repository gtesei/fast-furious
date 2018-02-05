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

# StemmedTfidfVectorizer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))


def text_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    text = BeautifulSoup(review,'html.parser').get_text()
    #
    # 2. Remove non-letters
    text = re.sub("[^A-za-z0-9^,?!.\/'+-=]"," ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    #
    # 3. Convert words to lower case and split them
    words = text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list
    return(words)

def clean_text( text ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #text = BeautifulSoup(review,'html.parser').get_text()
    #
    # 2. Remove non-letters
    text = re.sub("[^A-za-z0-9^,?!.\/'+-=]"," ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " _exclamationmark_ ", text)
    text = re.sub(r"\?", " _questionmark_ ", text)
    #
    return text


def build_data_set(ngram=3,stem=False,max_features=2000,min_df=2,remove_stopwords=True):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test.fillna('__NA__',inplace=True)
    ## 
    clean_train_comments = []
    for i in range(train.shape[0]):
        clean_train_comments.append( clean_text(train["comment_text"][i]) )
    #print(">>> processing test set ...")
    for i in range(test.shape[0]):
        clean_train_comments.append( clean_text(test["comment_text"][i]) )
    qs = pd.Series(clean_train_comments).astype(str)
    if not stem:
        # 1-gram / no-stem
        vect = TfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect = vect.fit_transform(qs)
        #print("ifidf_vect:", ifidf_vect.shape)
        X = ifidf_vect.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    else:
        vect_stem = StemmedTfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect_stem = vect_stem.fit_transform(qs)
        #print("ifidf_vect_stem:", ifidf_vect_stem.shape)
        X = ifidf_vect_stem.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    Y_train = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
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

sample_submission = pd.read_csv("data/sample_submission.csv")

# proc
t0 = dt.datetime.now()

for label in labels:
    print(">>> processing ",label)
    X_train,X_test,Y_train = build_data_set(ngram=params[label]['ngrams'],
                                            stem=params[label]['stem'],
                                            max_features=params[label]['max_features'],
                                            min_df=2,remove_stopwords=True)
    Y_train_lab = Y_train[label]
    clf = lm.LogisticRegression(C=params[label]['C'])
    clf.fit(X_train, Y_train_lab)
    pred_proba = clf.predict_proba(X_test)[:, 1]
    sample_submission[label] = pred_proba
    sample_submission.to_csv("baseline_linear.csv", index=False)


t1 = dt.datetime.now()
print('Total time: %i seconds' % (t1 - t0).seconds)
 
