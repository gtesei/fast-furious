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
#import Stemmer
import nltk 
from nltk.corpus import wordnet as wn
import gensim
from numpy import linalg as LA


# average embedding 
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# sym 
def synsym(s1,s2):
    ts0 = nltk.pos_tag(nltk.word_tokenize(s1))
    ts1 = nltk.pos_tag(nltk.word_tokenize(s2))
    # adj
    jj0 = [x for x,y in ts0 if y=='JJ' or y=='JJR' or y=='JJS']
    jj1 = [x for x,y in ts1 if y=='JJ' or y=='JJR' or y=='JJS']
    if len(jj0) == 0 or len(jj1) ==0:
      jjps = 0
    else:
      v1 = makeFeatureVec(jj0,model,300)
      v2 = makeFeatureVec(jj1,model,300)
      jjps = np.inner(v1,v2)/(LA.norm(v1)*LA.norm(v2))
    # noum
    jj0 = [x for x,y in ts0 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS' or y=='DT']
    jj1 = [x for x,y in ts1 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS' or y=='DT']
    if len(jj0) == 0 or len(jj1) ==0:
      nps = 0
    else:
      v1 = makeFeatureVec(jj0,model,300)
      v2 = makeFeatureVec(jj1,model,300)
      nps =  np.inner(v1,v2)/(LA.norm(v1)*LA.norm(v2))
    # verb
    jj0 = [x for x,y in ts0 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    jj1 = [x for x,y in ts1 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    if len(jj0) == 0 or len(jj1) ==0:
      vps = 0
    else:
      v1 = makeFeatureVec(jj0,model,300)
      v2 = makeFeatureVec(jj1,model,300)
      vps =  np.inner(v1,v2)/(LA.norm(v1)*LA.norm(v2))
    return [jjps,nps,vps]


# data  
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  


print(">> df_train:",df_train.shape)
print(">> df_test:",df_test.shape)

print("-- src:",df_train.ix[1,'question1']) 

qs1 = pd.Series(df_train['question1'].tolist() + df_test['question1'].tolist()).astype(str)
qs2 = pd.Series(df_train['question2'].tolist() + df_test['question2'].tolist()).astype(str)

feat = np.zeros((df_train.shape[0]+df_test.shape[0],3))

for i in range(0,feat.shape[0]):
  if i % 100000 ==0: 
    print(str(i),end=' ',flush=True) 
  feat[i,:] =  synsym(qs1[i],qs2[i])
  if i == 0:
    print(str(feat[i,:]))
  
tr_feat = feat[0:df_train.shape[0]]  
te_feat = feat[df_train.shape[0]:]
 
# write on disk 
tr_csv = pd.DataFrame({'adj_sym_we':tr_feat[:,0],'noun_sym_we':tr_feat[:,1],'verb_sym_we':tr_feat[:,2]})
te_csv = pd.DataFrame({'adj_sym_we':te_feat[:,0],'noun_sym_we':te_feat[:,1],'verb_sym_we':te_feat[:,2]})

tr_csv.to_csv("xtrain_4.csv", index=False)
te_csv.to_csv("xtest_4.csv", index=False)

