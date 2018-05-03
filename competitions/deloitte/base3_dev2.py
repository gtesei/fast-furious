import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans


def process_am(x):
    aa = ''
    if type(x) == pd.core.series.Series:
        x = x.values
        aa = [aa + x[i] for i in range(len(x))]
        aa = aa[0]
        aa = re.sub('"'," ", aa)
    elif  type(x) == str:
        aa = x
        aa = re.sub('"'," ", aa)
    aal = []
    _aal = aa.split(',')
    for aa in _aal:
        aa = re.sub("{"," ", aa)
        aa = re.sub("}"," ", aa)
        aa = re.sub(","," ", aa)
        aa = re.sub(":"," ", aa)
        aa = re.sub('’n',"", aa)
        aa = aa.strip()
        aa = re.sub('\s+',"_", aa)
        aa = aa.lower()
        if len(aa)>0: 
            aal.append(aa)
    return dict.fromkeys(set(aal), 1)

def perc2float(x):
    return float(x.strip('%'))/100


########################
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("train:",train.shape)
print("test:",test.shape)

# 1. log_price
print("1. log_price")
y_train = train['log_price']
train = train.drop(['log_price'],axis=1)
assert train.shape[1] == test.shape[1]
for i in range(train.shape[1]):
    assert train.columns[i] == test.columns[i]

train_obs = len(train)
all_data = pd.concat([train,test],axis=0)

# 2. property_type, room_type, bed_type
print('--------------> Feature Engineering ... ')
print("2. property_type, room_type, bed_type")
encoder = LabelEncoder()
encoder.fit(all_data['property_type']) 
all_data['property_type'] = encoder.transform(all_data['property_type'])

all_data['room_type'] = all_data['room_type'].map( {'Entire home/apt':5, 'Private room':3, 'Shared room':1})

all_data.bed_type = all_data.bed_type.fillna('missing')
encoder = LabelEncoder()
encoder.fit(all_data['bed_type']) 
all_data['bed_type'] = encoder.transform(all_data['bed_type'])

# 3. amenities 
print("3. amenities")
am_list = [process_am( all_data.iloc[i]['amenities']) for i in range(len(all_data))]
assert len(am_list) == len(all_data)
v = DictVectorizer(sparse=False)
X = v.fit_transform(am_list)
amenities_df = pd.DataFrame(data=X,columns=v.feature_names_)
amenities_df.index = all_data.index
all_data = pd.concat([all_data,amenities_df],axis=1)
all_data = all_data.drop(['amenities'],axis=1)
del amenities_df

#4. accommodates , bathrooms

#5. cancellation_policy, cleaning_fee
print("5. cancellation_policy, cleaning_fee")
all_data['cancellation_policy'] = all_data['cancellation_policy'].map( {
    'super_strict_60':20, 
    'super_strict_30':30, 
    'strict':50,
    'moderate':10,
    'flexible':5,
    'long_term':1,
})

all_data['cleaning_fee'] = all_data['cleaning_fee'].map( {
    True:1, 
    False:0
})

# 6. city
print("6. city")
encoder = LabelEncoder()
encoder.fit(all_data['city']) 
all_data['city'] = encoder.transform(all_data['city'])

# 7. description TODO
print("7. description ... TODO")
all_data['description'] = all_data['description'].fillna('')
all_data = all_data.drop(['description'],axis=1)


# 8. first_review ,  last_review , number_of_reviews , review_scores_rating
print("7. 8. first_review ,  last_review , number_of_reviews , review_scores_rating ... TODO better")
all_data['is_last_review_na'] = all_data['last_review'].isnull().map({True: 1 , False: 0 }) 
all_data['is_first_review_na'] = all_data['first_review'].isnull().map({True: 1 , False: 0 }) 

last_review_year = pd.to_datetime(all_data['last_review']).dt.year
last_review_year = last_review_year.fillna(1900)
all_data['last_review_year'] = last_review_year
last_review_month = pd.to_datetime(all_data['last_review']).dt.month
last_review_month = last_review_month.fillna(-1)
all_data['last_review_month'] = last_review_month

first_review_year = pd.to_datetime(all_data['first_review']).dt.year
first_review_year = first_review_year.fillna(1900)
all_data['first_review_year'] = first_review_year
first_review_month = pd.to_datetime(all_data['first_review']).dt.month
first_review_month = first_review_month.fillna(-1)
all_data['first_review_month'] = first_review_month

most_recent_review = pd.to_datetime(all_data.last_review).max()
delta_last_review = most_recent_review - pd.to_datetime(all_data.last_review)
delta_last_review = delta_last_review.fillna(-1)
delta_last_review = delta_last_review.map(lambda x: x.total_seconds()/(60*60*24))
all_data['delta_most_recent_review'] = delta_last_review

delta_rev = pd.to_datetime(all_data.last_review) - pd.to_datetime(all_data.first_review)
delta_rev = delta_rev.fillna(-1)
delta_rev = delta_rev.map(lambda x: x.total_seconds()/(60*60*24))
all_data['delta_rev'] = delta_rev

delta_rev_density = all_data.number_of_reviews+0.0000000000000001 / delta_rev
delta_rev_density = delta_rev_density.fillna(0)
all_data['delta_rev_density'] = delta_rev_density

all_data = all_data.drop(['first_review','last_review'],axis=1)
all_data['review_scores_rating'] = all_data['review_scores_rating'].fillna(-1)

# 9. host_has_profile_pic, host_identity_verified, host_since
print("9. host_has_profile_pic, host_identity_verified, host_since ")
all_data['host_has_profile_pic'] = all_data['host_has_profile_pic'].fillna('f')
all_data['host_identity_verified'] = all_data['host_identity_verified'].fillna('f')
all_data['host_has_profile_pic'] = all_data['host_has_profile_pic'].map({'t':1,'f':0})
all_data['host_identity_verified'] = all_data['host_identity_verified'].map({'t':1,'f':0})

all_data['is_host_since_na'] = all_data['host_since'].isnull().map({True: 1 , False: 0 }) 
host_oldest = pd.to_datetime(all_data.host_since).min()
delta_host = pd.to_datetime(all_data.host_since) - host_oldest 
delta_host = delta_host.fillna(-1)
delta_host = delta_host.map(lambda x: x.total_seconds()/(60*60*24))
all_data['delta_host'] = delta_host

delta_host_lev = np.zeros(len(all_data))
for i in range(len(all_data)):
    if all_data.iloc[i]['is_host_since_na'] == 1:
        delta_host_lev[i] = -1
    elif all_data.iloc[i]['delta_host'] < 1871.0:
        delta_host_lev[i] = 1
    elif all_data.iloc[i]['delta_host'] < 2398.0:    
        delta_host_lev[i] = 2
    else:
        delta_host_lev[i] = 3
all_data['delta_host_lev'] = delta_host_lev 

host_since_year = pd.to_datetime(all_data['host_since']).dt.year
host_since_year = host_since_year.fillna(2018)
all_data['host_since_year'] = host_since_year
host_since_month = pd.to_datetime(all_data['host_since']).dt.month
host_since_month = host_since_month.fillna(-1)
all_data['host_since_month'] = host_since_month

all_data = all_data.drop(['host_since'],axis=1)

# 10. host_response_rate , instant_bookable
print("10. host_response_rate , instant_bookable ")
all_data['instant_bookable'] = all_data['instant_bookable'].map({'t':1,'f':0})
all_data.host_response_rate = all_data.host_response_rate.fillna('0%')
all_data.host_response_rate = all_data.host_response_rate.apply(perc2float)


# 11. latitude,longitude  TODO ... leave as-is for now 
print("11. latitude,longitude  .......... TODO ")
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_100'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_1000'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=1500, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_1500'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=2000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_2000'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=3000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_3000'] = kmeans.predict(all_data[['latitude','longitude']])

# 12. name, neighbourhood, thumbnail_url, zipcode 
print("11. name, neighbourhood, thumbnail_url, zipcode   .......... TODO better ")
all_data['thumbnail_url_ok'] = 0 
all_data['thumbnail_url_ok'] [all_data.thumbnail_url.isnull() == False ] = 1

# neighbourhood
all_data['is_neighbourhood_na'] = all_data['neighbourhood'].isnull().map({True: 1 , False: 0 }) 
all_data['neighbourhood'] = all_data['neighbourhood'].fillna('UKN')
encoder = LabelEncoder()
encoder.fit(all_data['neighbourhood']) 
all_data['neighbourhood'] = encoder.transform(all_data['neighbourhood'])

# zipcode
all_data['is_zipcode_na'] = all_data['zipcode'].isnull().map({True: 1 , False: 0 }) 
all_data['zipcode'] = all_data['zipcode'].fillna('UKN')
encoder = LabelEncoder()
encoder.fit(all_data['zipcode']) 
all_data['zipcode'] = encoder.transform(all_data['zipcode'])


all_data = all_data.drop(['name','thumbnail_url',],axis=1)


# 12. bedrooms, beds , bed_type , bathrooms
all_data['is_bedrooms_na'] = all_data['bedrooms'].isnull().map({True: 1 , False: 0 }) 
all_data['is_beds_na'] = all_data['beds'].isnull().map({True: 1 , False: 0 }) 
all_data['is_bathrooms_na'] = all_data['bathrooms'].isnull().map({True: 1 , False: 0 }) 
all_data.bedrooms = all_data.bedrooms.fillna(0)
all_data.beds = all_data.beds.fillna(0)
all_data.bathrooms = all_data.bathrooms.fillna(0)

## rem sequnece 
all_data = all_data.drop(['id'],axis=1)

assert np.sum(all_data.isnull()).sum() == 0 

##################  
print('--------------> Modeling ... ')
Xtr, Xv, ytr, yv = train_test_split(all_data[:train_obs].values, y_train, test_size=0.1, random_state=1973)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(all_data[train_obs:].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 50,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'max_depth': 10,
            'subsample': 0.8,
            'lambda': 0.5,
            'nthread': -1,
            'booster' : 'gbtree',
            'silent': 1,
            'eval_metric': 'rmse',
            'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 10000, watchlist, early_stopping_rounds=50,maximize=False, verbose_eval=10)

print('Modeling RMSE %.5f' % model.best_score)

print('--------------> Submission ... ')
test['log_price'] = model.predict(dtest)
subfn = "base3dev1__val_"+str(model.best_score)+"__rnd_"+str(model.best_iteration)+".csv"
test[['id', 'log_price']].to_csv(subfn, index=False)

print('--------------> Retrain all data + Feature importance ... ')
dtrain = xgb.DMatrix(all_data[:train_obs].values, label=y_train)
dtest = xgb.DMatrix(all_data[train_obs:].values)
model = xgb.train(xgb_pars, dtrain, model.best_iteration+5, maximize=False, verbose_eval=10)
print('-----> Submission ... ')
test['log_price'] = model.predict(dtest)
subfn = "base3dev1__all_data__rnd_"+str(model.best_iteration)+".csv"
test[['id', 'log_price']].to_csv(subfn, index=False)

print('-----> Feature importance ... ')
feature_names = all_data.columns
feature_importance_dict = model.get_fscore()
fs = ['f%i' % i for i in range(len(feature_names))]
f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()), 'importance': list(feature_importance_dict.values())})
f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
feature_importance = pd.merge(f1, f2, how='right', on='f')
feature_importance = feature_importance.fillna(0)
feature_importance.sort_values(by='importance', ascending=False)
print(feature_importance.sort_values)
subfn = "error__feat_importance_base3dev1.csv" 
feature_importance.to_csv(subfn, index=False) 