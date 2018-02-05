import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
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

def get_model(embed_size = 200 , 
              num_lstm = 50 , 
              rate_drop_lstm = 0, 
              rate_drop_dense = 0.1,
              num_dense = 50):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    #x = Bidirectional(LSTM(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm ))(x)
    x = Bidirectional(GRU(num_lstm, return_sequences=True))(x)
    x = Dropout(rate_drop_dense)(x)
    #x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(2*num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    #x = BatchNormalization()(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# train 
batch_size = 32
epochs = 10
model = get_model()
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
sample_submission.to_csv("sub_gru3_.csv.gz", index=False , compression='gzip')