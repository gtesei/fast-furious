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
import re , os , sys , argparse 
import logging, gensim , random
from gensim.models import word2vec
from smart_open import smart_open
from numpy import linalg as LA

logger = logging.getLogger(__name__)

def glove2word2vec(glove_input_file, word2vec_output_file):
    """Convert `glove_input_file` in GloVe format into `word2vec_output_file` in word2vec format."""
    num_lines, num_dims = get_glove_info(glove_input_file)
    logger.info("converting %i vectors from %s to %s", num_lines, glove_input_file, word2vec_output_file)
    with smart_open(word2vec_output_file, 'wb') as fout:
        fout.write("{0} {1}\n".format(num_lines, num_dims).encode('utf-8'))
        with smart_open(glove_input_file, 'rb') as fin:
            for line in fin:
                fout.write(line)
    return num_lines, num_dims

def get_glove_info(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for _ in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## conf 
EMBEDDING_DIM = 300
we_fn='data/glove.6B.'+str(EMBEDDING_DIM)+'d.txt'
we_fn_conv='data/glove.6B.'+str(EMBEDDING_DIM)+'d.gensim.txt'
we_fn_up='data/glove.6B.'+str(EMBEDDING_DIM)+'d.gensim.up.txt'


# convert glove2gensim
num_lines, num_dims = glove2word2vec(glove_input_file=we_fn, word2vec_output_file=we_fn_conv)
logger.info('Converted model with %i vectors and %i dimensions', num_lines, num_dims)

# load data 
train = pd.read_csv("data/train.csv")
#train = train[:2000]
test = pd.read_csv("data/test.csv")
#test = test[:2000]
print("> train:",train.shape)
print("> test:",test.shape)

#
list_sentences_train = train["comment_text"].fillna("__NA__").values
list_sentences_test = test["comment_text"].fillna("__NA__").values
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list(list_sentences_train))
list_tokenized_test = tokenizer.texts_to_sequences(list(list_sentences_test))
word_index = tokenizer.word_index
print(len(word_index) , len(list_tokenized_train) , len(list_tokenized_test))

rev = {v: k for k, v in word_index.items()}
#


# Set values for various parameters                  
min_word_count = 5     # Minimum word count                        
num_workers = 4        # Number of threads to run in parallel
context = 5            # Context window size                                                                                    
downsampling = 1e-5    # Downsample setting for frequent words

alpha_val = 0.025         # Initial learning rate
min_alpha_val = 0.001     # Minimum for linear learning rate decay
passes = 20               # Number of passes of one document during training
alpha_delta = (alpha_val - min_alpha_val) / passes

# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(workers=num_workers, 
                          size=EMBEDDING_DIM, 
                          min_count = min_word_count, 
                          window = context, 
                          sample = downsampling)



list_tokenized_train_w = [[rev[i] for i in list_tokenized_train[j]] for j in range(len(list_tokenized_train)) ]
list_tokenized_test_w = [[rev[i] for i in list_tokenized_test[j]] for j in range(len(list_tokenized_test)) ]
list_tokenized_test_w.extend(list_tokenized_train_w) 
print(list_tokenized_test_w[0])
print(list_tokenized_test_w[11])
print("sentences:",len(list_tokenized_test_w))

model.build_vocab(list_tokenized_test_w)
print(">> .. vocab:",len(model.wv.vocab))
model.intersect_word2vec_format(we_fn_conv, binary=False , lockf=1.0)
arr_origin = np.array(model['blocked'], copy=True)   
#print(arr_origin)
arr_origin2 = np.array(model['down'], copy=True)  

#model.train(list_tokenized_test_w, total_examples=model.corpus_count, epochs=20)

for epoch in range(passes):
  # Shuffling gets better results
  random.shuffle(list_tokenized_test_w)
  # Train
  model.alpha, model.min_alpha = alpha_val, alpha_val
  model.train(list_tokenized_test_w,total_examples=len(list_tokenized_test_w), epochs=1)
  # Logs
  print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))
  # Next run alpha
  alpha_val -= alpha_delta


print(">>> .. checking serialization[word=blocked] cos:",np.inner(arr_origin,model['blocked'] )/(LA.norm(arr_origin)*LA.norm(model['blocked'] )))
# print("orig++++++++++++")
# print(arr_origin)
# print("====== new" )
# print(model['blocked'])
print(">>> .. checking serialization[word=down] cos:",np.inner(arr_origin2,model['down'] )/(LA.norm(arr_origin2)*LA.norm(model['down'] )))
#
print(">> saving ...")
model.save(we_fn_up)
#model.wv.save_word2vec_format(we_fn_up)

model = gensim.models.Word2Vec.load(we_fn_up)
for k,v in model.wv.vocab.items():
    print(k,v)
    print(type(model[k]))
    print(model[k].shape)
    break



