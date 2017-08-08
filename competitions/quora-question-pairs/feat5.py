import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
##x%matplotlib inline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss
from numpy import linalg as LA
import re 
import Stemmer

import gensim 
import numpy as np
import nltk 
from nltk.corpus import wordnet as wn
from numpy import linalg as LA


# some stats 
print('# File sizes')
for f in os.listdir('./data'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('./data/' + f) / 1000000, 2)) + 'MB')

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

print(">> df_train:",df_train.shape)
print(">> df_test:",df_test.shape)

# Initial Feature Analysis
stops = set(stopwords.words("english"))


# TF-IDF 
def string_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(review , "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review)
    #
    # 3. Convert words to lower case and split them
    #review_text = review
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


print("string_to_wordlist)) -- src:",df_train.ix[1,'question1']) 
print("string_to_wordlist)) -- dest:",string_to_wordlist(df_train.ix[1,'question1']))


#docs = ['']*df_train.shape[0]
#for i in range(len(docs)):
#  docs[i] = " ".join(string_to_wordlist(str(df_train.ix[:,'question1'])))
qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist() + df_test['question1'].tolist()  + df_test['question2'].tolist() ).astype(str)


# 1/4 2-gram / no-stem
print("***  1-gram / no-stem ")
vect =  TfidfVectorizer(analyzer=u'word',
                                 stop_words='english',
                                 #stop_words=None,
                                 min_df=2,
                                 ngram_range=(1, 1),
                                 max_features=5000)
ifidf_vect = vect.fit_transform(qs)

# dict word-weigth 
ww = dict(zip(vect.get_feature_names(), vect.idf_))
print(ww['youtube'])


# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
type(model['computer'])
model['computer'].shape




 
tr_csv.to_csv("xtrain_4.csv", index=False)
te_csv.to_csv("xtest_4.csv", index=False)

