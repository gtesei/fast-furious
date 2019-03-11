import sys
import pickle
import os

import numpy as np
import gensim
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, Input, Dense, Dropout
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

from sklearn.preprocessing import LabelEncoder

from keras.layers import Reshape


import re
from keras.preprocessing.text import Tokenizer

import warnings
from operator import itemgetter 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split


##
LOWER_CASE = False  
DATA_DIR = 'data/'
Goog_w2v = DATA_DIR+"embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
dropout_prob=0.5
N_EPOCHS = 200
PREFIX="cnn_"
VALIDATION_SPLIT=0.001
##

def process_text(text):
    text = re.sub(r"\'s", " is ", text) 
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    return text
## 
print(os.listdir(DATA_DIR))
train_df = pd.read_csv(DATA_DIR+"train.csv")
test_df = pd.read_csv(DATA_DIR+"test.csv")

model_word_embed = gensim.models.KeyedVectors.load_word2vec_format(Goog_w2v,binary=True)
    
SEQ_LEN_TR = len(max(train_df['question_text'], key=len).split())
SEQ_LEN_TS = len(max(test_df['question_text'], key=len).split())
SEQ_LEN = max(SEQ_LEN_TR,SEQ_LEN_TS)
print("SEQ_LEN:",SEQ_LEN)
assert SEQ_LEN == 45 

##
train_cat_list, train_text_list, train_questions = [], [], [] 
test_text_list, test_questions = [], []

for i in range(len(train_df)):
    quest = train_df.loc[i,'question_text']
    train_questions.append(quest)
    train_cat_list.append(train_df.loc[i,'target'])
    train_text_list.append(text_to_word_sequence(process_text(quest),lower=LOWER_CASE))

for i in range(len(test_df)):
    quest = test_df.loc[i,'question_text']
    test_questions.append(quest)
    test_text_list.append(text_to_word_sequence(process_text(quest),lower=LOWER_CASE))
    
assert len(train_cat_list) == len(train_text_list)
assert len(train_cat_list) == len(train_questions)
assert len(test_questions) == len(test_text_list)

print(">> train_size:",len(train_cat_list))
print(">> train sample:",train_cat_list[44] , train_text_list[44], train_questions[44])
print(">> test_size:",len(test_questions))
print(">> test sample:", test_text_list[44] , test_questions[44])

##
tokenizer = Tokenizer(num_words=None,char_level=False,lower=False)
tokenizer.fit_on_texts(train_text_list + test_text_list) 
sequences_train = tokenizer.texts_to_sequences(train_text_list) # ... train , test .. 
sequences_test = tokenizer.texts_to_sequences(test_text_list) # ... train , test .. 
data_train = pad_sequences(sequences_train, maxlen=SEQ_LEN,padding='post')
data_test = pad_sequences(sequences_test, maxlen=SEQ_LEN,padding='post')
labels = np.array(train_cat_list)

nb_words = len(tokenizer.word_index)+1

print(">> Number of words:",nb_words)
print(">> data_train:",data_train.shape)
print(">> train sample:",sequences_train[44] , data_train[44] , train_text_list[44] , train_questions[44])
print(">> data_test:",data_test.shape)
print(">> test sample:",sequences_test[44] , data_test[44] , test_text_list[44] , test_questions[44])

########################################
## sample train/validation data
########################################
np.random.seed(17)
perm = np.random.permutation(len(data_train))
idx_train = perm[:int(len(data_train)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_train)*(1-VALIDATION_SPLIT)):]

data_tr = data_train[idx_train]
data_val = data_train[idx_val]
labels_tr = labels[idx_train]
labels_val = labels[idx_val]

del data_train

##
embedding_matrix = np.zeros((nb_words, 300))
print('>>>>>>>>>>> OUT LOG:',file=open('out.log','w'))
for word, i in tokenizer.word_index.items():
    if word in model_word_embed.vocab:
        #print('IN:',word)
        embedding_matrix[i] = model_word_embed.word_vec(word)
    else:
        print('>>> OUT <<<:',word,file=open('out.log','a'))
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

## Null word embeddings: 116,788 (LOWER_CASE = False)
## Null word embeddings: 141,480 (LOWER_CASE = True) e.g. autria, gennifer
EMBEDDING_DIM = len(embedding_matrix[1])
print("EMBEDDING_DIM:",EMBEDDING_DIM)

#### Model
embedding_layer = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM,weights=[embedding_matrix],input_length=SEQ_LEN,trainable=False)
sequence_input = Input(shape=(SEQ_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
embedded_sequences_rh = Reshape((SEQ_LEN,EMBEDDING_DIM,1))(embedded_sequences)

### -------------------------- MAIN CAT
# 2-gram
conv_1 = Conv2D(500, (2, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
max_pool_1 = MaxPooling2D(pool_size=(SEQ_LEN-2, 1 ))(conv_1) # 30
# 3-gram
conv_2 = Conv2D(500, (3, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
max_pool_2 = MaxPooling2D(pool_size=(SEQ_LEN-3, 1 ))(conv_2) # 29
# 4-gram
conv_3 = Conv2D(500, (4, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
max_pool_3 = MaxPooling2D(pool_size=(SEQ_LEN-4, 1 ))(conv_3) # 28
# 5-gram
conv_4 = Conv2D(500, (5, EMBEDDING_DIM), activation="relu") (embedded_sequences_rh)
max_pool_4 = MaxPooling2D(pool_size=(SEQ_LEN-5, 1))(conv_4) # 27
# concat 
merged = concatenate([max_pool_1, max_pool_2, max_pool_3,max_pool_4])
#merged = Reshape((1,-1))(merged)
#flatten = Attention_CNN(1)(merged)
flatten = Flatten()(merged)
# full-connect -- MAIN  
full_conn = Dense(128, activation= 'tanh')(flatten)
dropout_1 = Dropout(dropout_prob)(full_conn)
full_conn_2 = Dense(64, activation= 'tanh')(dropout_1)
dropout_2 = Dropout(dropout_prob)(full_conn_2)
output = Dense(1, activation= 'sigmoid')(dropout_2) 
model = Model(sequence_input,output)

########
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics= ['accuracy'])
model.summary()
earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_acc',mode='max')
checkpointer = ModelCheckpoint(PREFIX+'model.h5', verbose=1, save_best_only=True,monitor='val_acc',mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_acc',mode='max')

results = model.fit(data_tr,labels_tr,
                    validation_data=[data_val,labels_val],
                    batch_size=66, epochs=N_EPOCHS,
                    callbacks=[earlystopper, checkpointer,reduce_lr])

#learning_curve_df = plot_learn_curve(results,do_plot=False)
#learning_curve_df.to_csv(PREFIX+'learning_curve.csv')
print(">> TEST ...")
model = load_model(PREFIX+'model.h5')
print("> Sub category:")

th_best=0.35
f1_best=0
for th in np.linspace(0.1,0.9,20).tolist():
    pred_val = model.predict(data_val)
    pred_val = (np.array(pred_val) > th).astype(np.int)
    f1 = f1_score(labels_val,pred_val)
    if f1 > f1_best:
        f1_best = f1
        th_best = th
print("f1_best:",f1_best,"   --- th_best:",th_best)

###
pred = model.predict(data_test)
pred = (np.array(pred) > th_best).astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": np.squeeze(pred)})
submit_df.to_csv("submission.csv", index=False)