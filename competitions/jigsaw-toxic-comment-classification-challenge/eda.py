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


def build_data_set(ngram=3,stem=False,max_features=2000,min_df=2,remove_stopwords=True,holdout_perc=0.25,seed=123,debug=True):
    train_data = pd.read_csv('data/train.csv')
    if debug:
        df = train_data.loc[:5000]
    else:
        print(">>> loading test set ...")
        test = pd.read_csv('data/test.csv')
        test.fillna('__NA__',inplace=True)
        df = train_data
    np.random.seed(seed)
    perm = np.random.permutation(df.shape[0])
    df = df.sample(frac=1).reset_index(drop=True)
    ## 
    clean_train_comments = []
    for i in range(df.shape[0]):
        #clean_train_comments.append(" ".join(text_to_wordlist(df["comment_text"][i], remove_stopwords)))
        clean_train_comments.append( clean_text(df["comment_text"][i]) )
    if not debug:
        print(">>> processing test set ...")
        for i in range(test.shape[0]):
            clean_train_comments.append( clean_text(test["comment_text"][i]) )
    qs = pd.Series(clean_train_comments).astype(str)
    if not stem:
        # 1-gram / no-stem
        vect = TfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect = vect.fit_transform(qs)
        #print("ifidf_vect:", ifidf_vect.shape)
        X = ifidf_vect.toarray()
        if not debug:
            X = X[:df.shape[0]]
    else:
        vect_stem = StemmedTfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect_stem = vect_stem.fit_transform(qs)
        #print("ifidf_vect_stem:", ifidf_vect_stem.shape)
        X = ifidf_vect_stem.toarray()
        if not debug:
            X = X[:df.shape[0]]
    Y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    assert Y.shape[0] == X.shape[0]
    ## split 
    hold_out_obs = int(df.shape[0] * holdout_perc)
    train_obs = df.shape[0] - hold_out_obs
    # X 
    X_train = X[:train_obs]
    X_holdout = X[train_obs:]
    # Y_toxic
    Y_train = Y[:train_obs]
    Y_holdout = Y[train_obs:]
    return X_train,X_holdout,Y_train,Y_holdout




#--------------------------- Main()
# conf
debug = False
if debug:
    kfolds = 2 
    print(">>> Debug mode .")
else:
    print(">>> Production mode .")
    kfolds = 5 

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# models
columns = ['1_gram',
           '2_gram',
           '3_gram',
           'Stem',
           'Max_Features',
           'Classifier',
           'Best_Parameters_CV',
           'Best_LogLoss_CV',
           'STD_CV',
           'LogLoss_HOLDOUT',
           'Resampling_Procedure']

perf_panels = { 'toxic': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns),
                'severe_toxic': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns),
                'obscene': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns),
                'threat': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns),
                'insult': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns),
                'identity_hate': pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
                }

models = ['LogisticRegression','MultinomialNB']
parameters = {
    'RandomForest': {"n_estimators": [100, 1000, 10000],
                     "max_depth": [3, 1, None],
                     "criterion": ["gini", "entropy"]},
    'SVC': {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1, 5, 10, 50, 100]},
    'LogisticRegression': {'C': [0.1,0.8,1,1.2,10]},
    'MultinomialNB': {'alpha': [0.1, 0.5, 0.9, 1]},
    'KNeighborsClassifier': {'n_neighbors': [5, 10, 20, 50], 'weights': ['uniform', 'distance']},
    'MLPClassifier': {'hidden_layer_sizes': [(1000, 50),(2000,100),(3000,200),(3000,1000,100)]}
}

#--------------------------- Assumptions to be tuned during next rounds
#    - removed stopwords
#    - removed numbers ... instead of replacing with _number_
#    - min_df = 2 (min. freq)
#    - no lemmatization
#-----------------------------------------------------------------

#--------------------------- pre-processing options
stem_options = [True,False]
grams = [1,2,3]
max_features = [500,2000,5000,10000]
#-----------------------------------------------------------------
# proc
t0 = dt.datetime.now()
iter = 1
for stem_option in stem_options:
    for max_feat in max_features:
        for gram in grams:
            X_train,X_holdout,Y_train,Y_holdout = build_data_set(ngram=gram, stem=stem_option,max_features=max_feat,debug=debug)
            for label in labels:
                Y_train_lab = Y_train[label]
                Y_holdout_lab = Y_holdout[label]
                perf_panel_fn = 'perf_panel_'+ label + '.csv'
                for model in models:
                    if model == "XGB":
                        raise ValueError('TODO: ' + model)
                    elif model == 'RandomForest':
                        clf = RandomForestClassifier(n_estimators = 1000)
                    elif model == 'SVC':
                        clf = SVC(kernel='linear',C=10)
                    elif model == 'LogisticRegression':
                        clf = lm.LogisticRegression(C=1e5)
                    elif model == 'MultinomialNB':
                        clf =  MultinomialNB(alpha=0.0005)
                    elif model == 'KNeighborsClassifier':
                        clf = neighbors.KNeighborsClassifier(10, weights='distance')
                    elif model == 'MLPClassifier':
                        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 50))
                    else:
                        raise ValueError('unkwown model: ' + model)
                    #
                    print(iter,"************************************************")
                    print("label:",label,"--gram:",gram,"--stem_option:",stem_option,"--max_features:",max_feat,"--model:",model)
                    grid_clf = GridSearchCV(estimator=clf, param_grid=parameters[model],
                                            cv=kfolds, scoring='neg_log_loss', n_jobs=10, iid=False)
                    grid_clf.fit(X_train, Y_train_lab)
                    pred_proba = grid_clf.predict_proba(X_holdout)
                    ll_holdout = log_loss(y_true=Y_holdout_lab,y_pred=pred_proba[:, 1])
                    sys.stdout.flush()
                    print("best params:",grid_clf.best_params_)
                    best_idx = np.argmax(grid_clf.cv_results_['mean_test_score'])
                    print("best LogLoss:", grid_clf.cv_results_['mean_test_score'][best_idx] , " - std:",grid_clf.cv_results_['std_test_score'][best_idx])
                    print("LogLoss holdout:",ll_holdout)
                    sys.stdout.flush()
                    print()
                    sys.stdout.flush()
                    # finally
                    perf_panel = perf_panels[label]
                    perf_panel = perf_panel.append(pd.DataFrame(np.array([[gram >= 1, gram >= 2, gram >= 3, stem_option, max_feat,
                                                                           model, str(grid_clf.best_params_), str(grid_clf.cv_results_['mean_test_score'][best_idx]),
                                                                           str(grid_clf.cv_results_['std_test_score'][best_idx]), str(ll_holdout),
                                                                           '1 Repeat 5-fold cross validation']]),
                                                                columns=columns))
                    perf_panel.sort_values(by='LogLoss_HOLDOUT', ascending=True , inplace=True)
                    perf_panel.to_csv(perf_panel_fn , index=False)
                    perf_panels[label] = perf_panel

                    iter = iter + 1

t1 = dt.datetime.now()
print('Total time: %i seconds' % (t1 - t0).seconds)
 
