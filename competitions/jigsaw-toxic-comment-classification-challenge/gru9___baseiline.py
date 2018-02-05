import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from keras.models import Model
from keras.layers import Dense, Embedding, Input , Activation
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten , Conv1D , GlobalMaxPooling1D , GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import re , os 
import logging, gensim , random
from gensim.models import word2vec
from keras.layers.merge import concatenate

# conf 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_features = 20000
#ax_features = 15000


######## ARMONY #####################################
# maxlen 200                           (2x)
# EMBEDDING_DIM 100                    (x)   <--- 
# GRU           100 (layers = 1)       (x) 
# num_dense     100                    (x) 
#####################################################

## Epochs on x-val: 1 

maxlen = 600
 
EMBEDDING_DIM = 300
we_fn='glove.840B.300d.txt'

num_lstm = 300 
lstm_layers = 1
rate_drop_dense = 0.1
num_dense = 300

batch_size = 32
epochs = 10

# load data 
train = pd.read_csv("data/train.csv")
#train = train[:2000]
test = pd.read_csv("data/test.csv")
#test = test[:2000]
train = train.sample(frac=1)


# pre-processing 
def pre_process_pre_trained_embed(train,test,we_fn='glove.6B.300d.txt'):
	print('>> Indexing word vectors ...')
	embeddings_index = {}
	f = open(os.path.join('data', we_fn))
	for line in f:
	   values = line.split(' ')
	   word = values[0] #print("values:",values)
	   coefs = np.asarray(values[1:], dtype='float32')
	   embeddings_index[word] = coefs
	f.close()
	#model = gensim.models.Word2Vec.load(os.path.join('data', we_fn))
	#for k,v in model.wv.vocab.items():
  #              embeddings_index[k] = model[k]
	print('Found %s word vectors.' % len(embeddings_index))

	print(">> pre-processing ... ")
	list_sentences_train = train["comment_text"].fillna("__NA__").values
	y = train[list_classes].values
	list_sentences_test = test["comment_text"].fillna("__NA__").values
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
	list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
	list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))
	word_index = tokenizer.word_index
	X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
	X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

	# prepare embedding matrix
	print('>> Preparing embedding matrix...')
	num_words = min(max_features, len(word_index))
	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
	for word, i in word_index.items():
	    if i >= max_features:
	        continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	return X_t, X_te, y , embedding_matrix

def pre_process(train,test):
	print(">> pre-processing ... ")
	list_sentences_train = train["comment_text"].fillna("__NA__").values
	y = train[list_classes].values
	list_sentences_test = test["comment_text"].fillna("__NA__").values
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
	list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
	list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))
	X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
	X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
	return X_t, X_te, y

def get_gru_bidirectional_avg(embed_size = 200 , 
			  embedding_matrix = None, 
              num_lstm = 50 , 
              rate_drop_dense = 0.1,
              num_dense = 50):
    
    if embedding_matrix is None: 
	    print(">> get_model_bidirectional_avg [no pre-trained word embeddings]<<")
	    inp = Input(shape=(maxlen, ))
	    x = Embedding(max_features, embed_size)(inp)
    else:
        print(">> get_model_bidirectional_avg [pre-trained word embeddings]<<")
        embedding_layer = Embedding(max_features,embed_size,weights=[embedding_matrix],input_length=maxlen,trainable=False)
        inp = Input(shape=(maxlen, ) , dtype='int32')
        x = embedding_layer(inp)
    x = Bidirectional(GRU(num_lstm, return_sequences=True))(x)
    x = Dropout(rate_drop_dense)(x)

    #add a GlobalAveragePooling1D, which will average the embeddings of all words in the document
    x = GlobalAveragePooling1D()(x)

    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    #x = BatchNormalization()(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_bidirectional(embed_size = 200 , 
        embedding_matrix = None, 
              num_lstm = 50 , 
              rate_drop_dense = 0.1,
              num_dense = 50):
    
    if embedding_matrix is None: 
      print(">> get_model_bidirectional_avg [no pre-trained word embeddings]<<")
      inp = Input(shape=(maxlen, ))
      x = Embedding(max_features, embed_size)(inp)
    else:
        print(">> get_model_bidirectional_avg [pre-trained word embeddings]<<")
        #embedding_layer = Embedding(max_features,embed_size,weights=[embedding_matrix],input_length=maxlen,trainable=True)
        embedding_layer = Embedding(max_features,embed_size,weights=[embedding_matrix],input_length=maxlen)
        inp = Input(shape=(maxlen, ) , dtype='int32')
        x = embedding_layer(inp)
    #x = Bidirectional(LSTM(num_lstm, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_dense))(x)
    for i in range(lstm_layers):
        x = Bidirectional(GRU(num_lstm, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_dense,trainable=True))(x)
    #x = Dropout(rate_drop_dense)(x)

    #add a GlobalAveragePooling1D, which will average the embeddings of all words in the document
    #x1 = GlobalAveragePooling1D()(x)
    #x2 = GlobalMaxPool1D()(x)
    #x = concatenate([x1, x2])

    x = GlobalMaxPool1D()(x)
    
    #x = BatchNormalization()(x)
    #x = Dropout(rate_drop_dense)(x)

    ## 1 layer
    #x = Dense(num_dense, activation="relu")(x)

    #x = BatchNormalization()(x)
    #x = Dropout(rate_drop_dense)(x)

    # 2 layer 
    x = Dense(num_dense, activation="relu")(x)

    #x = BatchNormalization()(x)
    x = Dropout(rate_drop_dense)(x)

    # output 
    #x = Dropout(rate_drop_dense)(x)
    #x = BatchNormalization()(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  #optimizer='nadam',
                  metrics=['accuracy'])

    return model


def get_model_conv(embed_size = 200 , 
              rate_drop_dense = 0.2,
              filters = 250, 
              kernel_size = 3, 
              num_dense = 50):
    print(">> Conv1D <<")

    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(rate_drop_dense))
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
    
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    model.add(Dense(num_dense))
    model.add(Dropout(rate_drop_dense))
    model.add(Activation('relu'))

    # output layer 
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_model_bidirectional(embed_size = 200 , 
              num_lstm = 64 , 
              rate_drop_lstm = 0, 
              rate_drop_dense = 0.1,
              num_dense = 50):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Bidirectional(GRU(num_lstm)))
    model.add(Dropout(rate_drop_dense))
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dropout(rate_drop_dense))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_cnn_lstm(embed_size = 200 , 
              rate_drop_dense = 0.2,
              filters = 64, 
              lstm_output_size = 70, 
              kernel_size = 5, 
              num_dense = 50):
    print(">>> cnn + gru <<")

    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(rate_drop_dense))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(GRU(lstm_output_size)))

    model.add(Dense(num_dense, activation='relu'))
    model.add(Dropout(rate_drop_dense))

    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# train 
#X_t, X_te, y = pre_process(train=train,test=test)
X_t, X_te, y , embedding_matrix = pre_process_pre_trained_embed(train=train,test=test,we_fn=we_fn)
model = get_bidirectional(embed_size = EMBEDDING_DIM , embedding_matrix = embedding_matrix,num_lstm = num_lstm,rate_drop_dense = rate_drop_dense,num_dense = num_dense)
print(model.summary())
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=0)
callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

# predict
print(">>> predicting on test set ... ")
model.load_weights(file_path)
y_test = model.predict(X_te)

#sub
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("sub_gru9_300e_in600_feat20000_poolMax_dense300_adam_5.csv.gz", index=False , compression='gzip')
