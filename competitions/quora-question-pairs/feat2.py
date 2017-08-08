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

# StemmedTfidfVectorizer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(TfidfVectorizer, self).build_analyzer()
    return lambda doc: english_stemmer.stemWords(analyzer(doc))

def compcos(q1_vect,q2_vect,isTrain=False):
    trcs = np.zeros(q1_vect.shape[0])
    for i in range(0,q1_vect.shape[0]):
      if i % 10000 == 0:
          print(i,end=" ",flush=True)
      v1 = q1_vect[i].toarray()
      v2 = q2_vect[i].toarray() 
      if LA.norm(v1) == 0 or LA.norm(v2) == 0:
          trcs[i] = 0 
      else: 
          trcs[i] = np.inner(v1,v2)/(LA.norm(v1)*LA.norm(v2))
    if isTrain: 
      print('\n>> cosine AUC:', roc_auc_score(df_train['is_duplicate'], trcs))
      print('>> cosine logloss:', log_loss(y_true=df_train['is_duplicate'], y_pred=trcs))
    return trcs




# 1/4 2-gram / no-stem
print("*** [1/4]  2-gram / no-stem ")
vect =  TfidfVectorizer(analyzer=u'word',
                                 stop_words='english',
                                 #stop_words=None,
                                 min_df=2,
                                 ngram_range=(2, 2),
                                 max_features=5000)
ifidf_vect = vect.fit_transform(qs)
#feat = ifidf_vect.toarray()

# dict word-weigth 
ww = dict(zip(vect.get_feature_names(), vect.idf_))
print(ww['youtube channel'])

tr_qs1_tfidf = ifidf_vect[0:df_train.shape[0]]  
tr_qs2_tfidf = ifidf_vect[df_train.shape[0]:2*df_train.shape[0]]
te_qs1_tfidf = ifidf_vect[2*df_train.shape[0]:(2*df_train.shape[0]+df_test.shape[0])]
te_qs2_tfidf = ifidf_vect[(2*df_train.shape[0]+df_test.shape[0]):]
 
cstr_2_nostem = compcos(tr_qs1_tfidf,tr_qs2_tfidf,isTrain=True)
cste_2_nostem = compcos(te_qs1_tfidf,te_qs2_tfidf)

# 2/4 3-gram / no-stem
print("*** [2/4]  2-gram / no-stem ")
vect =  TfidfVectorizer(analyzer=u'word',
                                 stop_words='english',
                                 #stop_words=None,
                                 min_df=2,
                                 ngram_range=(3, 3),
                                 max_features=5000)
ifidf_vect = vect.fit_transform(qs)
#feat = ifidf_vect.toarray()

tr_qs1_tfidf = ifidf_vect[0:df_train.shape[0]]  
tr_qs2_tfidf = ifidf_vect[df_train.shape[0]:2*df_train.shape[0]]
te_qs1_tfidf = ifidf_vect[2*df_train.shape[0]:(2*df_train.shape[0]+df_test.shape[0])]
te_qs2_tfidf = ifidf_vect[(2*df_train.shape[0]+df_test.shape[0]):]
 
cstr_3_nostem = compcos(tr_qs1_tfidf,tr_qs2_tfidf,isTrain=True)
cste_3_nostem = compcos(te_qs1_tfidf,te_qs2_tfidf)

# 3/4 2-gram / stem
print("*** [3/4]  2-gram / stem ")
vect =  StemmedTfidfVectorizer(analyzer=u'word',
                                 stop_words='english',
                                 #stop_words=None,
                                 min_df=2,
                                 ngram_range=(2, 2),
                                 max_features=5000)
ifidf_vect = vect.fit_transform(qs)
#feat = ifidf_vect.toarray()

tr_qs1_tfidf = ifidf_vect[0:df_train.shape[0]]  
tr_qs2_tfidf = ifidf_vect[df_train.shape[0]:2*df_train.shape[0]]
te_qs1_tfidf = ifidf_vect[2*df_train.shape[0]:(2*df_train.shape[0]+df_test.shape[0])]
te_qs2_tfidf = ifidf_vect[(2*df_train.shape[0]+df_test.shape[0]):]
 
cstr_2_stem = compcos(tr_qs1_tfidf,tr_qs2_tfidf,isTrain=True)
cste_2_stem = compcos(te_qs1_tfidf,te_qs2_tfidf)

# 4/4 2-gram / stem
print("*** [4/4]  3-gram / stem ")
vect =  StemmedTfidfVectorizer(analyzer=u'word',
                                 stop_words='english',
                                 #stop_words=None,
                                 min_df=2,
                                 ngram_range=(3, 3),
                                 max_features=5000)
ifidf_vect = vect.fit_transform(qs)
#feat = ifidf_vect.toarray()

tr_qs1_tfidf = ifidf_vect[0:df_train.shape[0]]  
tr_qs2_tfidf = ifidf_vect[df_train.shape[0]:2*df_train.shape[0]]
te_qs1_tfidf = ifidf_vect[2*df_train.shape[0]:(2*df_train.shape[0]+df_test.shape[0])]
te_qs2_tfidf = ifidf_vect[(2*df_train.shape[0]+df_test.shape[0]):]
 
cstr_3_stem = compcos(tr_qs1_tfidf,tr_qs2_tfidf,isTrain=True)
cste_3_stem = compcos(te_qs1_tfidf,te_qs2_tfidf)

# write on disk 
tr_csv = pd.DataFrame({'nostem_2':cstr_2_nostem,'no_stem3':cstr_3_nostem,'stem_2':cstr_2_stem,'stem_3':cstr_3_stem})
te_csv = pd.DataFrame({'nostem_2':cste_2_nostem,'no_stem3':cste_3_nostem,'stem_2':cste_2_stem,'stem_3':cste_3_stem})

tr_csv.to_csv("xtrain_2.csv", index=False)
te_csv.to_csv("xtest_2.csv", index=False)

