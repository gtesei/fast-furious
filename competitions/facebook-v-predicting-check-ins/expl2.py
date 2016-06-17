import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print('Loading data ...')
df_train = pd.read_csv \
    (
        'C:\\Users\\gtesei\\Desktop\\Deloitte\\C_Folder\\Cognitive_Technologies\\Machine_Learning\\git\\fast-furious\dataset\\facebook-v-predicting-check-ins\\train.csv',
        dtype={'x': np.float32,
               'y': np.float32,
               'accuracy': np.int16,
               'time': np.int,
               'place_id': np.int},
        index_col=0)
df_test = pd.read_csv \
    (
        'C:\\Users\\gtesei\\Desktop\\Deloitte\\C_Folder\\Cognitive_Technologies\\Machine_Learning\\git\\fast-furious\dataset\\facebook-v-predicting-check-ins\\test.csv',
        dtype={'x': np.float32,
               'y': np.float32,
               'accuracy': np.int16,
               'time': np.int,
               'place_id': np.int},
        index_col=0)

place_counts = df_train.place_id.value_counts()
place_ids = place_counts.keys()
counts = place_counts.values
print('place_counts:'+str(len(place_counts)))
plt.hist(counts)
print('count mean:'+str(np.mean(counts)))
print('place_counts:'+str(np.std(counts)))
print('count min:'+str(np.min(counts)))
print('count max:'+str(np.max(counts)))

for th in range(15):
    mask = (place_counts >= (10* th)).values
    print('count th ='+str(th*10)+" --> "+str(sum(counts > (10*th)))+" --- "+str(sum(counts > (10*th))/len(counts)))



x_dist = df_train[df_train['place_id']==place_ids[0]].x.values
x_max = np.max(x_dist)
x_min = np.min(x_dist)
x_delta = x_max - x_min
x_mean = np.mean(x_dist)
x_std = np.std(x_dist)

y_dist = df_train[df_train['place_id']==place_ids[0]].y.values
y_max = np.max(y_dist)
y_min = np.min(y_dist)
y_delta = y_max - y_min
y_mean = np.mean(y_dist)
y_std = np.std(y_dist)

## x binning --> 15
x_means =  df_train.loc[:,['place_id','x']].groupby('place_id').mean()
x_stds =  df_train.loc[:,['place_id','x']].groupby('place_id').std()

x_means_mean = np.mean(x_means.values)
x_stds = x_stds.values
x_stds_mean = np.mean(x_stds[~np.isnan(x_stds)]).mean()
print("x_stds_mean:"+str(x_stds_mean)+"---> number of bins:"+str(round((10/x_stds_mean))))

## y binning --> 480
y_means =  df_train.loc[:,['place_id','y']].groupby('place_id').mean()
y_stds =  df_train.loc[:,['place_id','y']].groupby('place_id').std()

y_means_mean = np.mean(y_means.values)
y_stds = y_stds.values
y_stds_mean = np.mean(y_stds[~np.isnan(y_stds)]).mean()
print("x_stds_mean:"+str(y_stds_mean)+"---> number of bins:"+str(round((10/y_stds_mean))))


## x
std =  np.mean(df.loc[:,['place_id','x']].groupby('place_id').std()) #0.68076
min = 0
max = 10

wx = (max-min) / std ##14.689474

## y
std = np.mean(df.loc[:,['place_id','y']].groupby('place_id').std()) #0.020819
min = 0
max = 10

wy = (max-min) / std ## 480.326019

## hour
std = np.mean(df.loc[:,['place_id','hour']].groupby('place_id').std()) #5.632953
min = 0
max = 23

wh = (max-min) / std #4.083116

## weekday
std = np.mean(df.loc[:,['place_id','weekday']].groupby('place_id').std()) #1.845209
min = 0
max = 6

wwd = (max-min) / std #3.251665

## day
std = np.mean(df.loc[:,['place_id','day']].groupby('place_id').std()) #89.172616
min = 1
max = 365

wd = (max-min) / std #4.081971


## month
std = np.mean(df.loc[:,['place_id','month']].groupby('place_id').std()) #2.922695
min = 1
max = 12

wm = (max-min) / std #3.763649

## year
std = np.mean(df.loc[:,['place_id','year']].groupby('place_id').std()) #0.416436
min = 2014
max = 2015

wy = (max-min) / std #2.40133




