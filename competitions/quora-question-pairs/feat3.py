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
import nltk 
from nltk.corpus import wordnet as wn


# sym 
def synsym(s1,s2):
    ts0 = nltk.pos_tag(nltk.word_tokenize(s1))
    ts1 = nltk.pos_tag(nltk.word_tokenize(s2))
    # adj  
    jj0 = [x for x,y in ts0 if y=='JJ' or y=='JJR' or y=='JJS']
    jj1 = [x for x,y in ts1 if y=='JJ' or y=='JJR' or y=='JJS']
    jj0w = [wn.synsets(xx,pos=wn.ADJ) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.ADJ) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    jjps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    jjps = [x for x in jjps if x != None]
    if len(jjps)==0:
      jjps = [0]
    # noum  
    jj0 = [x for x,y in ts0 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS']
    jj1 = [x for x,y in ts1 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS']
    jj0w = [wn.synsets(xx,pos=wn.NOUN) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.NOUN) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    nps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    nps = [x for x in nps if x != None]
    if len(nps)==0:
      nps = [0]
    # verb  
    jj0 = [x for x,y in ts0 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    jj1 = [x for x,y in ts1 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    jj0w = [wn.synsets(xx,pos=wn.VERB) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.VERB) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    vps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    vps = [x for x in vps if x != None]
    if len(vps)==0:
      vps = [0]
    # adverb  
    jj0 = [x for x,y in ts0 if y=='RB' or y=='RBR' or y=='RBS' or y=='WRB']
    jj1 = [x for x,y in ts1 if y=='RB' or y=='RBR' or y=='RBS' or y=='WRB']
    jj0w = [wn.synsets(xx,pos=wn.ADV) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.ADV) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    aps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    aps = [x for x in aps if x != None]
    if len(aps)==0:
      aps = [0]
    return [jjps,nps,vps,aps]

# data  
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

print(">> df_train:",df_train.shape)
print(">> df_test:",df_test.shape)

print("-- src:",df_train.ix[1,'question1']) 

qs1 = pd.Series(df_train['question1'].tolist() + df_test['question1'].tolist()).astype(str)
qs2 = pd.Series(df_train['question2'].tolist() + df_test['question2'].tolist()).astype(str)

feat = np.zeros((df_train.shape[0]+df_test.shape[0],4))

for i in range(0,feat.shape[0]):
  if i % 100000 ==0: 
    print(str(i),end=' ',flush=True) 
  l =  synsym(qs1[i],qs2[i])
  feat[i,0] = max(l[0])
  feat[i,1] = max(l[1])
  feat[i,2] = max(l[2])
  feat[i,3] = max(l[3])
  
tr_feat = feat[0:df_train.shape[0]]  
te_feat = feat[df_train.shape[0]:]
 
# write on disk 
tr_csv = pd.DataFrame({'adj_sym':tr_feat[:,0],'noun_sym':tr_feat[:,1],'verb_sym':tr_feat[:,2],'adv_sym':tr_feat[:,3]})
te_csv = pd.DataFrame({'adj_sym':te_feat[:,0],'noun_sym':te_feat[:,1],'verb_sym':te_feat[:,2],'adv_sym':te_feat[:,3]})

tr_csv.to_csv("xtrain_3.csv", index=False)
te_csv.to_csv("xtest_3.csv", index=False)

