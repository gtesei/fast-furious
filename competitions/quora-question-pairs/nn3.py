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

import math 
import numpy as np
import tensorflow as tf

num_steps = 6001
RS = 1234

print("Started")
np.random.seed(RS)
input_folder = './data/'

# data 
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

x_train_1 = pd.read_csv('xtrain.csv')
del x_train_1['Unnamed: 0']
x_test_1 = pd.read_csv('xtest.csv')
del x_test_1['Unnamed: 0']
print("Feature set 1: X_train: {}, X_test: {}".format(x_train_1.shape,x_test_1.shape))

x_train_2 = pd.read_csv('xtrain_2.csv') 
#del x_train_2['Unnamed: 0'] 
x_test_2 = pd.read_csv('xtest_2.csv')  
#del x_test_2['Unnamed: 0']
print("Feature set 2: X_train: {}, X_test: {}".format(x_train_2.shape, x_test_2.shape))

x_train_3 = pd.read_csv('xtrain_3.csv') 
x_test_3 = pd.read_csv('xtest_3.csv')  
print("Feature set 3: X_train: {}, X_test: {}".format(x_train_3.shape, x_test_3.shape))

y_train = df_train['is_duplicate'].values

x_train = pd.concat([x_train_1,x_train_2,x_train_3],axis=1)
x_test = pd.concat([x_test_1,x_test_2,x_test_3],axis=1)
print("Merge: X_train: {}, X_test: {}".format(x_train.shape, x_test.shape))

assert x_train.shape[0] == df_train.shape[0]
assert x_test.shape[0] == df_test.shape[0]

# resample 
if 0: # Now we oversample the negative class - on your own risk of overfitting!
  pos_train = x_train[y_train == 1]
  neg_train = x_train[y_train == 0]
  print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  p = 0.165
  scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
  while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
  neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
  print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
  x_train = pd.concat([pos_train, neg_train])
  y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
  del pos_train, neg_train

#NN 
def create_nn3_model_and_run(graph,
                         train_dataset,
                         train_labels,
                         valid_dataset,
                         valid_labels,
                         test_dataset,
                         test_labels,
                         beta,
                         num_steps,
                         hidden_size = 24, 
                         num_labels=2,batch_size = 30000):
    
    uniMax = 1/math.sqrt(hidden_size)
    
    with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.cast(tf.placeholder(tf.float32,shape=(batch_size,train_dataset.shape[1])),tf.float32)
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

      tf_valid_labels = tf.cast(tf.constant(valid_labels),tf.float32)
      tf_valid_dataset = tf.cast(tf.constant(valid_dataset),tf.float32)
      tf_test_dataset = tf.cast(tf.constant(test_dataset),tf.float32)
      
      # Hidden 1
      weights_1 = tf.cast(tf.Variable(tf.random_uniform([train_dataset.shape[1], hidden_size], minval=-uniMax, maxval=uniMax),name='weights_1'),tf.float32)
      biases_1 = tf.cast(tf.Variable(tf.random_uniform([hidden_size],minval=-uniMax, maxval=uniMax),name='biases_1'),tf.float32)
      hidden_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
        
      # Hidden 2
      weights_2 = tf.cast(tf.Variable(tf.random_uniform([hidden_size, hidden_size], minval=-uniMax, maxval=uniMax),name='weights_2'),tf.float32)
      biases_2 = tf.cast(tf.Variable(tf.random_uniform([hidden_size],minval=-uniMax, maxval=uniMax),name='biases_2'),tf.float32)
      hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights_2) + biases_2)
    
      # Hidden 3
      weights_3 = tf.cast(tf.Variable(tf.random_uniform([hidden_size, hidden_size], minval=-uniMax, maxval=uniMax),name='weights_3'),tf.float32)
      biases_3 = tf.cast(tf.Variable(tf.random_uniform([hidden_size],minval=-uniMax, maxval=uniMax),name='biases_3'),tf.float32)
      hidden_3 = tf.nn.relu(tf.matmul(hidden_2, weights_3) + biases_3)

      # Softmax 
      weights_4 = tf.cast(tf.Variable(tf.random_uniform([hidden_size, num_labels],minval=-uniMax, maxval=uniMax), name='weights_4'),tf.float32)
      biases_4 = tf.cast(tf.Variable(tf.random_uniform([num_labels],minval=-uniMax, maxval=uniMax),name='biases_4'),tf.float32)
      logits = tf.matmul(hidden_3, weights_4) + biases_4

      # 
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels) )+(1/batch_size)*beta*(tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4))


      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      
      valid_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2),
              weights_3)+biases_3),weights_4)+biases_4

      valid_prediction = tf.nn.softmax(valid_logits)


      test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2),
              weights_3)+biases_3),weights_4)+biases_4)

      loss_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits, labels=tf_valid_labels) )+(1/batch_size)*beta*(tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4))

      # Optimizer.
      #global_step = tf.Variable(0)  # count the number of steps taken.
      #learning_rate = tf.train.exponential_decay(0.5, global_step, 100000, 0.96, staircase=True)
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


    test_accuracy = 0
    with tf.Session(graph=graph) as session, tf.device('/gpu:0'):
#    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
    
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]
            
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#            _, l  = session.run([optimizer, loss], feed_dict=feed_dict)
            
            if (step % 500 == 0): 
              _, l,trp,xvp,tep  = session.run([optimizer, loss,train_prediction,valid_prediction,test_prediction], feed_dict=feed_dict)  
              print("Minibatch loss at step %d: %f" % (step, l))
              print("Minibatch valid loss at step %d: %f" % (step, loss_valid.eval(feed_dict=feed_dict)))
              print("Minibatch accuracy: %f" % accuracy(trp, batch_labels))
              print(trp[0:5])
              print(logits.eval(feed_dict=feed_dict))
              print(weights_1.eval(feed_dict=feed_dict))
              #print(weights_2.eval(feed_dict=feed_dict))
              #print(weights_3.eval(feed_dict=feed_dict))
              print("Validation accuracy: %f" % accuracy(xvp, valid_labels))
              test_accuracy = accuracy(tep, test_labels)
              print("Test accuracy: %f" % test_accuracy)
              print("")
            else:
              _, l  = session.run([optimizer, loss], feed_dict=feed_dict) 
    return test_accuracy


print("Will train NN3 for {} rounds, RandomSeed: {}".format(num_steps, RS))
x_train = x_train.fillna(value=0)
x_test = x_test.fillna(value=0)

x, X_val, ytrain, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RS)
y_val =  np.array(y_val)
ytrain = np.array(ytrain)

def reformat(labels,num_labels):
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return labels

y_val = reformat(y_val,2)
ytrain = reformat(ytrain,2)

print("*ytrain*")
print(ytrain[0:5])
print("*y_val*")
print(y_val[0:5])



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
betas = [0, 0.001,0.01,0.1,1,10]
test_accuracy = np.zeros(len(betas))
i = 0
for beta in betas:
  print("\n>>>>>>>>>> Beta: %f%%" % beta)
  graph = tf.Graph()
  test_accuracy[i] = create_nn3_model_and_run(graph,
                         x.as_matrix(),
                         ytrain,
                         X_val.as_matrix(),
                         y_val,
                         X_val.as_matrix(),
                         y_val,
                         beta,
                         num_steps)
   
  i = i +1



# preds = clf.predict(xgb.DMatrix(x_test))

# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds
# sub.to_csv("xgb_feat_seed_3{}_n{}.csv".format(RS, ROUNDS), index=False)

print("Done.")
