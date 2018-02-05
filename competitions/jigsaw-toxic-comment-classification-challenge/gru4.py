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
import re 

# conf 
max_features = 20000
maxlen = 100

# load data 
train = pd.read_csv("data/train.csv")
#train = train[:1000]
test = pd.read_csv("data/test.csv")
#test = test[:1000]
train = train.sample(frac=1)


# pre-processing 
print(">> pre-processing ... ")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"].fillna("__NA__").values
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("__NA__").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model_bidirectional_avg(embed_size = 200 , 
              num_lstm = 50 , 
              rate_drop_lstm = 0, 
              rate_drop_dense = 0.1,
              num_dense = 50):
    
    print(">> get_model_bidirectional_avg <<")

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
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
batch_size = 32
epochs = 10
model = get_cnn_lstm()
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
sample_submission.to_csv("sub_gru5_.csv.gz", index=False , compression='gzip')