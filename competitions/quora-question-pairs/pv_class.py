'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import gensim 
from numpy import linalg as LA


import operator
from sklearn.cross_validation import train_test_split

########################################
## set directories and parameters
########################################
BASE_DIR = './data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
EMBEDDING_DIM = 400
TEXT_WINDOW = 9 

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))

print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])

print('Found %s texts in test.csv' % len(test_texts_1))

# "[..] In our experiments, we cross validate the window size using the validation set, and the optimal window size is 8. The vector presented to the classifier is a concatenation of two vectors, one from PV-DBOW and one from PV-DM. In PV-DBOW, the learned vector representations have 400 dimensions. In PV-DM, the learned vector representations have 400 dimensions for both words and paragraphs. To predict the 8-th word, we concatenate the paragraph vectors and 7 word vectors. Special characters such as ,.!? are treated as a normal word. If the paragraph has less than 9 words, we pre-pad with a special NULL word symbol.[..]"

print('Loading PV models ...')
model_pv_dbow = gensim.models.doc2vec.Doc2Vec.load('doc2vec_pv_dbow')
model_pv_dm = gensim.models.doc2vec.Doc2Vec.load('doc2vec_pv_dm')

# xtrain
x_train_pv = np.zeros([len(texts_1),4*EMBEDDING_DIM])
#x_train_pv2 = np.zeros([len(texts_1),4*EMBEDDING_DIM])
for i, line in enumerate(texts_1):
  #line = ' '.join(line) 
  if (i == 0):
        print('0:',line)
  # 1 
  x_train_pv[i,:EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(line))
  x_train_pv[i,EMBEDDING_DIM:2*EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(line))
  x_train_pv[i,2*EMBEDDING_DIM:3*EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(texts_2[i]))
  x_train_pv[i,3*EMBEDDING_DIM:] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(texts_2[i]))
  # 2 
  #x_train_pv2[i,:EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(line))
  #x_train_pv2[i,EMBEDDING_DIM:2*EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(line))
  #x_train_pv2[i,2*EMBEDDING_DIM:3*EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(texts_2[i]))
  #x_train_pv2[i,3*EMBEDDING_DIM:] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(texts_2[i]))
print(i,"Finished train set")

# xtest
x_test_pv = np.zeros([len(test_texts_1),4*EMBEDDING_DIM])
#x_test_pv2 = np.zeros([len(test_texts_1),4*EMBEDDING_DIM])
for i, line in enumerate(test_texts_1):
  #line = ' '.join(line) 
  if (i == 0):
        print('0:',line)
  # 1 
  x_test_pv[i,:EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(line))
  x_test_pv[i,EMBEDDING_DIM:2*EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(line))
  x_test_pv[i,2*EMBEDDING_DIM:3*EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(test_texts_2[i]))
  x_test_pv[i,3*EMBEDDING_DIM:] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(test_texts_2[i]))
  # 2 
  #x_test_pv2[i,:EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(line))
  #x_test_pv2[i,EMBEDDING_DIM:2*EMBEDDING_DIM] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(line))
  #x_test_pv2[i,2*EMBEDDING_DIM:3*EMBEDDING_DIM] = model_pv_dm.infer_vector(gensim.utils.simple_preprocess(test_texts_2[i]))
  #x_test_pv2[i,3*EMBEDDING_DIM:] = model_pv_dbow.infer_vector(gensim.utils.simple_preprocess(test_texts_2[i]))


del  model_pv_dbow, model_pv_dm, texts_1, texts_2, test_texts_1, test_texts_2
print(i,"Finished test set")

print('>> splitting train set ...')
labels_train = np.array(labels)
del labels
x_train, x_val, y_train, y_val = train_test_split(x_train_pv, labels_train, test_size=0.2, random_state=1234)
del x_train_pv, labels_train

print("Training ...")
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(50, input_shape=(4*EMBEDDING_DIM,)))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model_checkpoint = ModelCheckpoint('PV.h5', save_best_only=True, save_weights_only=True)
hist = model.fit([x_train], y_train, 
        validation_data=([x_val], y_val), 
        epochs=200, batch_size=2048, shuffle=True, 
        callbacks=[early_stopping, model_checkpoint])

########################################
## make the prediction
########################################
print('Making the submission before fine-tuning')

pred = model.predict([x_test_pv], batch_size=2048, verbose=1)
#pred = pred + model.predict([x_test_pv2], batch_size=2048, verbose=1)
#pred = pred / 2 

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':pred.ravel()})
submission.to_csv('pv.csv', index=False)

