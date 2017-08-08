import gensim 
import numpy as np
import nltk 
from nltk.corpus import wordnet as wn
from numpy import linalg as LA

bs = ['I had my normal periods 10 days after on my regular date however now I am one week late. Could I be pregnant?','I had sex two months ago. I had a period last month after it was 4-5 days late. This month, I am now 8 days late. What does this mean?']

gs = ['Just completed my B.Tech with an aggregate of 66%, X and XII 75, 57 respectively. The eligibilty criteria in most of companies is 60% and above in X, XII and graduation. Now that I am helpless and cannot do anything about my XII score, does this mean that I wont get a job anywhere?',"I'm not an IITian and graduated in 2016. I have CGPA around 8.4 and pretty good scores in X and XII. Can you advice considering following details?"]

bs2 =['How do I start my life?','How do I start all over again?']


tb = [nltk.pos_tag(nltk.word_tokenize(t)) for t in bs]
tb2 = [nltk.pos_tag(nltk.word_tokenize(t)) for t in bs2]

tg = [nltk.pos_tag(nltk.word_tokenize(t)) for t in gs]



# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  

type(model['computer'])
model['computer'].shape



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

