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



p = sns.color_palette()

print('# File sizes')
for f in os.listdir('./data'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('./data/' + f) / 1000000, 2)) + 'MB')


### Training set 
df_train = pd.read_csv('./data/train.csv')
print(df_train.head())

print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

#plt.hist(qids.value_counts(), bins=50)
#plt.yscale('log', nonposy='clip')
#plt.title('Log-Histogram of question appearance counts')
#plt.xlabel('Number of occurences of question')
#plt.ylabel('Number of questions')
#plt.show()
print()


## trivial model 
p = df_train['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))

df_test = pd.read_csv('./data/test.csv')
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
sub.to_csv('naive_submission.csv', index=False)
print(sub.head())

p = df_train['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))

df_test = pd.read_csv('./data/test.csv')
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
sub.to_csv('naive_submission.csv', index=False)


## Text analysis
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))

# Initial Feature Analysis
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

print('>>> train_word_match: min={:.2f} max={:.2f} mean={:.2f} std={:.2f}'.format(np.min(train_word_match),np.max(train_word_match),np.mean(train_word_match),np.std(train_word_match)))


print('>> train_word_match AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
print('>> train_word_match logloss:', log_loss(y_true=df_train['is_duplicate'], y_pred=train_word_match))

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


vectorizer =  TfidfVectorizer(analyzer=u'word',
                                 #stop_words='english',
                                 stop_words=None,
                                 min_df=2,
                                 ngram_range=(1, 1),
                                 max_features=5000)
ifidf_vect = vectorizer.fit_transform(qs)
#feat = ifidf_vect.toarray()

tr_qs1_tfidf = ifidf_vect[0:df_train.shape[0]]  
tr_qs2_tfidf = ifidf_vect[df_train.shape[0]:2*df_train.shape[0]]
te_qs1_tfidf = ifidf_vect[2*df_train.shape[0]:(2*df_train.shape[0]+df_test.shape[0])]
te_qs2_tfidf = ifidf_vect[(2*df_train.shape[0]+df_test.shape[0]):]
 

cx1 = np.zeros(df_train.shape[0])
for i in range(0,df_train.shape[0]):
  if i % 10000 == 0:
    print(i)
  v1 = tr_qs1_tfidf[i].toarray()
  v2 = tr_qs2_tfidf[i].toarray() 
  if LA.norm(v1) == 0 or LA.norm(v2) == 0:
    cx1[i] = 0 
  else: 
    cx1[i] = np.inner(v1,v2)/(LA.norm(v1)*LA.norm(v2)) 



print('>> cosine AUC:', roc_auc_score(df_train['is_duplicate'], cx1))
print('>> cosine logloss:', log_loss(y_true=df_train['is_duplicate'], y_pred=cx1))


