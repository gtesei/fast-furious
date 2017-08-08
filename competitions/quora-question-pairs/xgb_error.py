import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.metrics import roc_auc_score, log_loss


RS = 12357
ROUNDS = 500

print("Started")
np.random.seed(RS)
input_folder = './data/'

def train_xgb(X, y, params):
  print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
  x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

  xg_train = xgb.DMatrix(x, label=y_train)
  xg_val = xgb.DMatrix(X_val, label=y_val)

  watchlist  = [(xg_train,'train'), (xg_val,'eval')]

  clf = xgb.train(params=params,dtrain=xg_train,num_boost_round=70000,early_stopping_rounds=200,evals=watchlist)

#  return xgb.train(params, xg_train, ROUNDS, watchlist)
  return clf

def predict_xgb(clr, X_test):
	return clr.predict(xgb.DMatrix(X_test))

def create_feature_map(features):
  outfile = open('xgb.fmap', 'w')
  i = 0
  for feat in features:
    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    i = i + 1
  outfile.close()

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.05
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS

df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

print("Features processing, be patient...")

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
  return 0 if count < min_count else 1 / (count + eps)

  train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
  words = (" ".join(train_qs)).lower().split()
  counts = Counter(words)
  weights = {word: get_weight(count) for word, count in counts.items()}

  stops = set(stopwords.words("english"))
  def word_shares(row):
    q1 = set(str(row['question1']).lower().split())
    q1words = q1.difference(stops)
    if len(q1words) == 0:
      return '0:0:0:0:0:0'

    q2 = set(str(row['question2']).lower().split())
    q2words = q2.difference(stops)
    if len(q2words) == 0:
      return '0:0:0:0:0:0'

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    shared_words = q1words.intersection(q2words)
    diff_words = q1words.difference(q2words).union(q2words.difference(q1words))
    shared_weights = [weights.get(w, 0) for w in shared_words] 
    diff_weights = [weights.get(w, 0) for w in diff_words] 
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    RD = np.sum(diff_weights) / np.sum(total_weights) #tfidf diff
    return '{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, RD)

  df = pd.concat([df_train, df_test])
  df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

  x = pd.DataFrame()

  x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
  x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
  x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

  x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
  x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
  x['tfidf_diff']       = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
  x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

  x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
  x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
  x['diff_len'] = x['len_q1'] - x['len_q2']

  x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
  x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
  x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

  x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
  x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
  x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

  #... YOUR FEATURES HERE ...
	
  feature_names = list(x.columns.values)
  create_feature_map(feature_names)
  print("Features: {}".format(feature_names))

  x_train = x[:df_train.shape[0]]
  x_test  = x[df_train.shape[0]:]
  y_train = df_train['is_duplicate'].values
  #del x, df_train ##; x_train.to_csv("xtrain.csv",index=True); x_test.to_csv("xtest.csv",index=True)
  # if 1: # Now we oversample the negative class - on your own risk of overfitting!
  #   pos_train = x_train[y_train == 1]
  #   neg_train = x_train[y_train == 0]

  #   print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  #   p = 0.165
  #   scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
  #   while scale > 1:
  #     neg_train = pd.concat([neg_train, neg_train])
  #     scale -=1
  #   neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
  #   print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

  #   x_train = pd.concat([pos_train, neg_train])
  #   y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
  #   del pos_train, neg_train
	        
  # train 
  print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
  #clr = train_xgb(x_train, y_train, params)
  print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
  
  tr_cut = np.int_( np.floor(df_train.shape[0] * 0.8) )
  perm = np.random.permutation(df_train.shape[0])
  tr_idx = perm[0:tr_cut]
  val_ix = perm[tr_cut:]
  x_train_train = x_train.ix[tr_idx]
  x_val =  x_train.ix[val_ix]
  y_train_train = y_train[tr_idx]
  y_val =  y_train[val_ix] 
  #x, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

  xg_train = xgb.DMatrix(x_train_train, label=y_train_train)
  xg_val = xgb.DMatrix(x_val, label=y_val)

#  watchlist  = [(xg_train,'train'), (xg_val,'eval')]

  clf = xgb.train(params=params,dtrain=xg_train,num_boost_round=500)

  pred_val = clf.predict(xg_val)

  print(">> logloss::",log_loss(y_true=y_val, y_pred=pred_val))

  df_res =  pd.DataFrame({'pred_val':pred_val},index=x_val.index) 

  x_all = pd.concat([df_train.ix[val_ix],x_val],axis=1)
  x_all2 = pd.concat([x_all,df_res],axis=1)
  
  x_all2.to_csv("xgb_error.csv", index=False)


  # pred 
  #preds = predict_xgb(clr, x_test)

  # print("Writing output...")
  # sub = pd.DataFrame()
  # sub['test_id'] = df_test['test_id']
  # sub['is_duplicate'] = preds
  # sub.to_csv("xgb_seed_2{}_n{}.csv".format(RS, ROUNDS), index=False)

  # print("Features importances...")
  # importance = clr.get_fscore(fmap='xgb.fmap')
  # importance = sorted(importance.items(), key=operator.itemgetter(1))
  # ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

  # ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
  # plt.gcf().savefig('features_importance.png')

print("Done.")
