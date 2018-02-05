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
from keras.layers.merge import concatenate
from gensim.models import KeyedVectors

# conf 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_features = 20000
maxlen = 100

EMBEDDING_DIM = 300
we_fn='glove.6B.300d.txt'

EMBEDDING_DIM_2 = 300
we_fn2='GoogleNews-vectors-negative300.bin'

batch_size = 32
epochs = 10

# load data 
train = pd.read_csv("data/train.csv")
#train = train[:2000]
test = pd.read_csv("data/test.csv")
#test = test[:2000]
train = train.sample(frac=1)


# pre-processing 
def pre_process_pre_trained_embed(train,test,we_fn='glove.6B.300d.txt',we_fn2='sswe-u.txt'):
  print('>> Indexing word vectors',we_fn,"...")
  embeddings_index = {}
  f = open(os.path.join('data', we_fn))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  print('Found %s word vectors.' % len(embeddings_index))

  print('>> Indexing word vectors 2',we_fn2,"...")
  #embeddings_index_2 = {}
  embeddings_index_2 = KeyedVectors.load_word2vec_format(os.path.join('data', we_fn2), binary=True)
  print('Found %s word vectors of word2vec' % len(embeddings_index_2.vocab))
  #f = open(os.path.join('data', we_fn2))
  #for line in f:
  #    values = line.split()
  #    word = values[0]
  #    coefs = np.asarray(values[1:], dtype='float32')
  #    embeddings_index_2[word] = coefs
  #f.close()
  #print('Found %s word vectors.' % len(embeddings_index_2))

  print(">> pre-processing ... ")
  list_sentences_train = train["comment_text"].fillna("__NA__").values
  y = train[list_classes].values
  list_sentences_test = test["comment_text"].fillna("__NA__").values
  tokenizer = text.Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
  list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
  list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))
  word_index = tokenizer.word_index
  #print("> tokenizer.word_index:"+str(type(tokenizer.word_index)))
  #assert '?' in word_index.keys()
  #assert '!' in word_index.keys()
  #print("> tokenizer.word_index(?):"+str(word_index.get('?')))
  #print("> tokenizer.word_index(!):"+str(word_index.get('!')))
  X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
  X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

  # prepare embedding matrix 1 
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

  # prepare embedding matrix 2
  print('>> Preparing embedding matrix 2...')
  embedding_matrix_2 = np.zeros((num_words, EMBEDDING_DIM_2))
  #for i in range(num_words):
  #  embedding_matrix_2[i+1] = embeddings_index_2.get('<unk>')
  #embedding_matrix_2[0] = embeddings_index_2.get('<s>')
  for word, i in word_index.items():
    if i >= max_features:
          continue
    if word in embeddings_index_2.vocab:
        embedding_matrix_2[i] = embeddings_index_2.word_vec(word)

  return X_t, X_te, y , embedding_matrix , embedding_matrix_2

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

def get_model_bidirectional_avg(embed_size = 200 , embed_size_2 = 50 , 
			  embedding_matrix = None, 
        embedding_matrix_2 = None , 
              num_lstm = 50 , 
              rate_drop_dense = 0.1,
              num_dense = 50):
    
    if embedding_matrix is None: 
	    print(">> get_model_bidirectional_avg [no pre-trained word embeddings]<<")
	    inp = Input(shape=(maxlen, ))
	    x = Embedding(max_features, embed_size)(inp)
    else:
        print(">> get_model_bidirectional_avg [pre-trained word embeddings]<<")
        embedding_layer = Embedding(max_features,embed_size,weights=[embedding_matrix],input_length=maxlen,trainable=True)
        inp = Input(shape=(maxlen, ) , dtype='int32')
        x1 = embedding_layer(inp)
        x1 = Bidirectional(GRU(num_lstm, return_sequences=True))(x1)
        x1 = Dropout(rate_drop_dense)(x1)
        #add a GlobalAveragePooling1D, which will average the embeddings of all words in the document
        x1 = GlobalAveragePooling1D()(x1)
        if embedding_matrix_2 is not None:
          embedding_layer_2 = Embedding(max_features,embed_size_2,weights=[embedding_matrix_2],input_length=maxlen,trainable=True)
          x2 = embedding_layer_2(inp)
          x2 = Bidirectional(GRU(num_lstm, return_sequences=True))(x2)
          x2 = Dropout(rate_drop_dense)(x2)
          #add a GlobalAveragePooling1D, which will average the embeddings of all words in the document
          x2 = GlobalAveragePooling1D()(x2)
          x = concatenate([x1, x2])
        else:
          x = x1 
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    #x = BatchNormalization()(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
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
X_t, X_te, y , embedding_matrix, embedding_matrix_2 = pre_process_pre_trained_embed(train=train,test=test,we_fn=we_fn,we_fn2=we_fn2)
model = get_model_bidirectional_avg(embed_size = EMBEDDING_DIM , embed_size_2 = EMBEDDING_DIM_2 , embedding_matrix = embedding_matrix , embedding_matrix_2 = embedding_matrix_2)
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
sample_submission.to_csv("sub_gru6_emb_2.csv.gz", index=False , compression='gzip')