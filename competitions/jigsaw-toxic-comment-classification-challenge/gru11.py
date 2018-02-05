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
import nltk 
from collections import OrderedDict
import sys
from nltk.tokenize import word_tokenize

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out.
        lower: Whether to convert the input to lowercase.
        split: Sentence split marker (string).
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,) and isinstance(text, unicode):
        translate_map = dict((ord(c), unicode(split)) for c in filters)
    else:
        translate_map = maketrans(filters, split * len(filters))

    text = text.translate(translate_map)
    #seq = text.split(split)
    seq = text.split()
    
    #seq = word_tokenize(text)
    #print("text:",seq)
    
    #pos_seq = nltk.pos_tag(text)
    #return [i for i in seq if i]
    return nltk.pos_tag(seq)

class TokenizerPOS(text.Tokenizer):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...
    # Arguments
        num_words: the maximum number of words to keep, based
            on word frequency. Only the most common `num_words` words will
            be kept.
        filters: a string where each element is a character that will be
            filtered from the texts. The default is all punctuation, plus
            tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: character or string to use for token splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls
    By default, all punctuation is removed, turning the texts into
    space-separated sequences of words
    (words maybe include the `'` character). These sequences are then
    split into lists of tokens. They will then be indexed or vectorized.
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 **kwargs):
        # Legacy support
        if 'nb_words' in kwargs:
            warnings.warn('The `nb_words` argument in `Tokenizer` '
                          'has been renamed `num_words`.')
            num_words = kwargs.pop('nb_words')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.pos_counts = OrderedDict()
        self.pos_docs = {}
        
        self.filters = filters
        self.split = split
        self.lower = lower
        
        self.num_words = num_words
        
        self.document_count = 0
        self.char_level = char_level
        self.oov_token = oov_token
    
    def texts_to_sequences(self, texts):
        """Transforms each text in texts in a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Returns
            A list of sequences.
        """
        res = []
        res_pos = [] 
        for vect,vect_pos in self.texts_to_sequences_generator(texts):
            res.append(vect)
            res_pos.append(vect_pos)
        return res , res_pos

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in texts in a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        for text in texts:
            seq = text_to_word_sequence(text,self.filters,self.lower,self.split)
            vect = []
            res_pos = [] 
            for w,p in seq:
                i = self.word_index.get(w)
                j = self.word_index_pos.get(p)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
                        res_pos.append(j)
                elif self.oov_token is not None:
                    i = self.word_index.get(self.oov_token)
                    j = self.word_index_pos.get(self.oov_token)
                    if i is not None:
                        vect.append(i)
                        res_pos.append(j)
            yield vect , res_pos
    
    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        Required before using `texts_to_sequences` or `texts_to_matrix`.
        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        """
        self.document_count = 0
        print(len(texts))
        for text in texts:
            self.document_count += 1
            seq = text_to_word_sequence(text,self.filters,self.lower,self.split)
            #print(str(seq))
            for w,pos in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
                
                if pos in self.pos_counts:
                    self.pos_counts[pos] += 1
                else:
                    self.pos_counts[pos] = 1
                    
                    
            for w in set([w for w,pos in seq]):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1
            
            for pos in set([pos for w,pos in seq]):
                if pos in self.pos_docs:
                    self.pos_docs[pos] += 1
                else:
                    self.pos_docs[pos] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        
        pcounts = list(self.pos_counts.items())
        pcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc_pos = [wc[0] for wc in pcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index_pos = dict(list(zip(sorted_voc_pos, list(range(1, len(sorted_voc_pos) + 1)))))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1
            i = self.word_index_pos.get(self.oov_token)
            if i is None:
                self.word_index_pos[self.oov_token] = len(self.word_index_pos) + 1

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c
        
        self.index_docs_pos = {}
        for w, c in list(self.pos_docs.items()):
            self.index_docs_pos[self.word_index_pos[w]] = c




### --------------------> conf 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_features = 20000


######## ARMONY #####################################
# maxlen 200                           (2x)
# EMBEDDING_DIM 100                    (x)   <--- 
# GRU           100 (layers = 1)       (x) 
# num_dense     100                    (x) 
#####################################################


maxlen = 600
 
EMBEDDING_DIM_1 = 300
we_fn_1='glove.840B.300d.txt'

EMBEDDING_DIM_2 = 100

rate_drop_dense = 0.2
num_dense = EMBEDDING_DIM_1 + EMBEDDING_DIM_2

batch_size = 32
epochs = 10

### -------------------------->  load data 
train = pd.read_csv("data/train.csv")
#train = train[:2000]
test = pd.read_csv("data/test.csv")
#test = test[:2000]
train = train.sample(frac=1)


# pre-processing 
def pre_process_pre_trained_embed(train,test):
	print('>> Indexing word vectors ...')
	embeddings_index_1 = {}
	f = open(os.path.join('data', we_fn_1))
	for line in f:
	        values = line.split(' ')
	        word = values[0] #print("values:",values)
	        coefs = np.asarray(values[1:], dtype='float32')
	        embeddings_index_1[word] = coefs
	f.close()
	print('Found %s word vectors. [1]' % len(embeddings_index_1))

	print(">> pre-processing ... ")
	list_sentences_train = train["comment_text"].fillna("__NA__").values
	y = train[list_classes].values
	list_sentences_test = test["comment_text"].fillna("__NA__").values


  # TokenizerPOS
	tokenizer = TokenizerPOS(num_words=max_features)
	tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
	list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
	list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))

  ### ------------ word 
	word_index = tokenizer.word_index
	X_t = sequence.pad_sequences(list_tokenized_train[0], maxlen=maxlen)
	X_te = sequence.pad_sequences(list_tokenized_test[0], maxlen=maxlen)

	# prepare embedding matrix
	print('>> Preparing embedding matrix 1...')
	num_words = min(max_features, len(word_index))
	embedding_matrix_1 = np.zeros((num_words, EMBEDDING_DIM_1))
	for word, i in word_index.items():
	    if i >= max_features:
	        continue
	    embedding_vector = embeddings_index_1.get(word)
	    if embedding_vector is not None:
	            # words not found in embedding index will be all-zeros.
	            embedding_matrix_1[i] = embedding_vector

	### ------------ POS 
	X_t_POS = sequence.pad_sequences(list_tokenized_train[1], maxlen=maxlen)
	X_te_POS = sequence.pad_sequences(list_tokenized_test[1], maxlen=maxlen)

	return X_t, X_te, y , embedding_matrix_1 , X_t_POS , X_te_POS



def get_bidirectional(embed_size_1 = 200 , embedding_matrix_1 = None, embed_size_2 = 200 , 
              #num_lstm = 50 , 
              rate_drop_dense = 0.1,
              num_dense = 50):
        
    print(">> get_model_bidirectional_avg [pre-trained word embeddings]<<")
    #embedding_layer = Embedding(max_features,embed_size,weights=[embedding_matrix],input_length=maxlen,trainable=True)
    embedding_layer_1 = Embedding(max_features,embed_size_1,weights=[embedding_matrix_1],input_length=maxlen)
    embedding_layer_2 = Embedding(max_features,embed_size_2,input_length=maxlen)


    inp1 = Input(shape=(maxlen, ) , dtype='int32')
    inp2 = Input(shape=(maxlen, ) , dtype='int32')

    x1 = embedding_layer_1(inp1)
    x2 = embedding_layer_2(inp2)
    
    x = concatenate([x1, x2],axis=2)

    x = Bidirectional(GRU(num_dense, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_dense,trainable=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=[inp1,inp2], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  #optimizer='nadam',
                  metrics=['accuracy'])

    return model

# train 
X_t, X_te, y , embedding_matrix_1 , X_t_POS , X_te_POS = pre_process_pre_trained_embed(train=train,test=test)

model = get_bidirectional(embed_size_1 = EMBEDDING_DIM_1 , embedding_matrix_1 = embedding_matrix_1, embed_size_2 = EMBEDDING_DIM_2 , rate_drop_dense = rate_drop_dense,num_dense = num_dense)
print(model.summary())
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=0)
callbacks_list = [checkpoint, early] #early
model.fit([X_t,X_t_POS], y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list, shuffle=True)

# predict
print(">>> predicting on test set ... ")
model.load_weights(file_path)
y_test = model.predict([X_te,X_te_POS])

#sub
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("sub_gru11_Embed_POS_dropout02_2.csv.gz", index=False , compression='gzip')
