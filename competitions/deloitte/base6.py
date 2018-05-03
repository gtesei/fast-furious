import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import nltk 
from nltk.corpus import stopwords
import os 
from sklearn.decomposition import PCA

stops = set(stopwords.words("english"))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_w2v_model(dir = './' , we_fn = 'glove.840B.300d.txt'):
    print('    >> Indexing word vectors ...')
    embeddings_index = {}
    f = open(os.path.join(dir, we_fn))
    for line in f:
        values = line.split(' ')
        word = values[0] #print("values:",values)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('    >> Found %s word vectors. [1]' % len(embeddings_index))
    return embeddings_index

def count_desc_len(x):
    return len(review_to_sentences(x, tokenizer=tokenizer, stops=stops, remove_stopwords=True))

def string_to_wordlist( review, stops, remove_stopwords ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review,'html.parser').get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        #stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, stops , remove_stopwords ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.extend( string_to_wordlist( raw_sentence, stops = stops , remove_stopwords=remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


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
all_data.bathrooms = all_data.bathrooms.fillna(0)

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

all_data['description_len'] = all_data['description'].apply(count_desc_len)

embeddings_index = load_w2v_model()

featureVec = np.zeros((len(all_data),300),dtype="float32")
warn_w2v = 0 
for i in range(len(all_data)):
    words = review_to_sentences(all_data.iloc[i]['description'], tokenizer=tokenizer, stops=stops, remove_stopwords=True)
    featureVec_i = np.zeros((300),dtype="float32")
    #
    nwords = 0.
    # 
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in embeddings_index.keys(): 
            nwords = nwords + 1.
            featureVec_i = np.add(featureVec_i,embeddings_index[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords > 0: 
        featureVec_i = np.divide(featureVec_i,nwords)
    else:
        #print(">>> WARNING <<< No words in vocaublary")
        warn_w2v = warn_w2v + 1 
        #print(str(words))
    featureVec[i] = featureVec_i

print("    >> No words in vocaublary for ",warn_w2v,"cases")

#desc_w2v = pd.DataFrame(data=featureVec , columns=['desc_w2v_'+str(i) for i in range(300)])
#desc_w2v.index = all_data.index
#all_data = pd.concat([all_data,desc_w2v],axis=1)

pca = PCA().fit(featureVec)
w2v_desc_pca_transf = pca.transform(featureVec)
all_data['w2v_desc_pca0'] = w2v_desc_pca_transf[:, 0]
all_data['w2v_desc_pca1'] = w2v_desc_pca_transf[:, 1]
all_data['w2v_desc_pca2'] = w2v_desc_pca_transf[:, 2]
all_data['w2v_desc_pca3'] = w2v_desc_pca_transf[:, 3]
all_data['w2v_desc_pca4'] = w2v_desc_pca_transf[:, 4]
all_data['w2v_desc_pca5'] = w2v_desc_pca_transf[:, 5]
all_data['w2v_desc_pca6'] = w2v_desc_pca_transf[:, 6]
all_data['w2v_desc_pca7'] = w2v_desc_pca_transf[:, 7]
all_data['w2v_desc_pca8'] = w2v_desc_pca_transf[:, 8]
all_data['w2v_desc_pca9'] = w2v_desc_pca_transf[:, 9]
all_data['w2v_desc_pca10'] = w2v_desc_pca_transf[:, 10]
all_data['w2v_desc_pca11'] = w2v_desc_pca_transf[:, 11]
all_data['w2v_desc_pca12'] = w2v_desc_pca_transf[:, 12]
all_data['w2v_desc_pca13'] = w2v_desc_pca_transf[:, 13]
all_data['w2v_desc_pca14'] = w2v_desc_pca_transf[:, 14]
all_data['w2v_desc_pca15'] = w2v_desc_pca_transf[:, 15]
all_data['w2v_desc_pca16'] = w2v_desc_pca_transf[:, 16]
all_data['w2v_desc_pca17'] = w2v_desc_pca_transf[:, 17]
all_data['w2v_desc_pca18'] = w2v_desc_pca_transf[:, 18]
all_data['w2v_desc_pca19'] = w2v_desc_pca_transf[:, 19]
all_data['w2v_desc_pca20'] = w2v_desc_pca_transf[:, 20]
all_data['w2v_desc_pca21'] = w2v_desc_pca_transf[:, 21]
all_data['w2v_desc_pca22'] = w2v_desc_pca_transf[:, 22]
all_data['w2v_desc_pca23'] = w2v_desc_pca_transf[:, 23]
all_data['w2v_desc_pca24'] = w2v_desc_pca_transf[:, 24]
all_data['w2v_desc_pca25'] = w2v_desc_pca_transf[:, 25]
all_data['w2v_desc_pca26'] = w2v_desc_pca_transf[:, 26]
all_data['w2v_desc_pca27'] = w2v_desc_pca_transf[:, 27]
all_data['w2v_desc_pca28'] = w2v_desc_pca_transf[:, 28]
all_data['w2v_desc_pca29'] = w2v_desc_pca_transf[:, 29]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(featureVec)  ## TODO: tune the number of cluster  
all_data.loc[:, 'w2v_desc_cluster_100'] = kmeans.predict(featureVec)

kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=10000).fit(featureVec)  ## TODO: tune the number of cluster  
all_data.loc[:, 'w2v_desc_cluster_1000'] = kmeans.predict(featureVec)

kmeans = MiniBatchKMeans(n_clusters=3000, batch_size=10000).fit(featureVec)  ## TODO: tune the number of cluster  
all_data.loc[:, 'w2v_desc_cluster_3000'] = kmeans.predict(featureVec)

all_data = all_data.drop(['description'],axis=1)


# 8. first_review ,  last_review , number_of_reviews , review_scores_rating
print("8. first_review ,  last_review , number_of_reviews , review_scores_rating ... TODO better")
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

host_oldest = pd.to_datetime(all_data.host_since).min()
delta_host = pd.to_datetime(all_data.host_since) - host_oldest 
delta_host = delta_host.fillna(-1)
delta_host = delta_host.map(lambda x: x.total_seconds()/(60*60*24))
all_data['delta_host'] = delta_host

all_data = all_data.drop(['host_since'],axis=1)

# 10. host_response_rate , instant_bookable
print("10. host_response_rate , instant_bookable ")
all_data['instant_bookable'] = all_data['instant_bookable'].map({'t':1,'f':0})
all_data.host_response_rate = all_data.host_response_rate.fillna('0%')
all_data.host_response_rate = all_data.host_response_rate.apply(perc2float)


# 11. latitude,longitude  TODO ... leave as-is for now 
print("11. latitude,longitude  .......... TODO ")
# pca = PCA().fit(all_data[['latitude','longitude']])
# lalo_pca_transf = pca.transform(all_data[['latitude','longitude']])
# all_data['latitude'] = lalo_pca_transf[:, 0]
# all_data['longitude'] = lalo_pca_transf[:, 1]


kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_1000'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=3000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_3000'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=5000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_5000'] = kmeans.predict(all_data[['latitude','longitude']])
kmeans = MiniBatchKMeans(n_clusters=7000, batch_size=10000).fit(all_data[['latitude','longitude']])  ## TODO: tune the number of cluster  
all_data.loc[:, 'geo_cluster_7000'] = kmeans.predict(all_data[['latitude','longitude']])

# 12. name, neighbourhood, thumbnail_url, zipcode 
print("11. name, neighbourhood, thumbnail_url, zipcode   .......... TODO better ")
all_data['thumbnail_url_ok'] = 0 
all_data['thumbnail_url_ok'] [all_data.thumbnail_url.isnull() == False ] = 1

all_data['neighbourhood'] = all_data['neighbourhood'].fillna('UKN')
encoder = LabelEncoder()
encoder.fit(all_data['neighbourhood']) 
all_data['neighbourhood'] = encoder.transform(all_data['neighbourhood'])

all_data['zipcode'] = all_data['zipcode'].fillna('UKN')
encoder = LabelEncoder()
encoder.fit(all_data['zipcode']) 
all_data['zipcode'] = encoder.transform(all_data['zipcode'])

# name 
all_data['name'] = all_data['name'].fillna('')
featureVec = np.zeros((len(all_data),300),dtype="float32")
warn_w2v = 0 
for i in range(len(all_data)):
    words = review_to_sentences(all_data.iloc[i]['name'], tokenizer=tokenizer, stops=stops, remove_stopwords=True)
    featureVec_i = np.zeros((300),dtype="float32")
    #
    nwords = 0.
    # 
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in embeddings_index.keys(): 
            nwords = nwords + 1.
            featureVec_i = np.add(featureVec_i,embeddings_index[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords > 0: 
        featureVec_i = np.divide(featureVec_i,nwords)
    else:
        #print(">>> WARNING <<< No words in vocaublary")
        warn_w2v = warn_w2v + 1 
        #print(str(words))
    featureVec[i] = featureVec_i

print("    >> No words in vocaublary for ",warn_w2v,"cases")

pca = PCA().fit(featureVec)
w2v_name_pca_transf = pca.transform(featureVec)
all_data['w2v_name_pca0'] = w2v_name_pca_transf[:, 0]
all_data['w2v_name_pca1'] = w2v_name_pca_transf[:, 1]
all_data['w2v_name_pca2'] = w2v_name_pca_transf[:, 2]
all_data['w2v_name_pca3'] = w2v_name_pca_transf[:, 3]
all_data['w2v_name_pca4'] = w2v_name_pca_transf[:, 4]
all_data['w2v_name_pca5'] = w2v_name_pca_transf[:, 5]
all_data['w2v_name_pca6'] = w2v_name_pca_transf[:, 6]
all_data['w2v_name_pca7'] = w2v_name_pca_transf[:, 7]
all_data['w2v_name_pca8'] = w2v_name_pca_transf[:, 8]
all_data['w2v_name_pca9'] = w2v_name_pca_transf[:, 9]
all_data['w2v_name_pca10'] = w2v_name_pca_transf[:, 10]
all_data['w2v_name_pca11'] = w2v_name_pca_transf[:, 11]
all_data['w2v_name_pca12'] = w2v_name_pca_transf[:, 12]

all_data = all_data.drop(['name','thumbnail_url',],axis=1)


# 12. bedrooms, beds , bed_type 
all_data.bedrooms = all_data.bedrooms.fillna(0)
all_data.beds = all_data.beds.fillna(0)

## cut
# all_data = all_data.drop(['well-lit_path_to_entrance','smartlock','garden_or_backyard','window_guards','high_chair','hot_water_kettle','pocket_wifi','babysitter_recommendations',
#     'private_bathroom','accessible-height_bed','flat','waterfront','baby_bath','free_parking_on_street','wide_entryway','beach_essentials','accessible-height_toilet','handheld_shower_head','other_pet(s)',
#     'wide_hallway_clearance','smooth_pathway_to_front_door','wide_clearance_to_bed','changing_table','baby_monitor','other','wide_clearance_to_shower_&_toilet','table_corner_guards','air_purifier',
#     'bath_towel','bathtub_with_shower_chair','beachfront','body_soap','disabled_parking_spot','ev_charger','firm_matress','firm_mattress','fixed_grab_bars_for_shower_&_toilet','flat_smooth_pathway_to_front_door',
#     'grab-rails_for_shower_and_toilet','ground_floor_access','hand_or_paper_towel','hand_soap','lake_access','paid_parking_off_premises','path_to_entrance_lit_at_night','roll-in_shower_with_chair',
#     'ski_in/ski_out','toilet_paper','washer_/_dryer','wide_clearance_to_shower_and_toilet'],axis=1)

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
subfn = "base6_eta0005__val_"+str(model.best_score)+"__rnd_"+str(model.best_iteration)+".csv"
test[['id', 'log_price']].to_csv(subfn, index=False)

print('--------------> Retrain all data + Feature importance ... ')
dtrain = xgb.DMatrix(all_data[:train_obs].values, label=y_train)
dtest = xgb.DMatrix(all_data[train_obs:].values)
model = xgb.train(xgb_pars, dtrain, model.best_iteration+5, maximize=False, verbose_eval=10)
print('-----> Submission ... ')
test['log_price'] = model.predict(dtest)
subfn = "base60005__all_data__rnd_"+str(model.best_iteration)+".csv"
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
subfn = "error__feat_importance_base6eta0005.csv" 
feature_importance.to_csv(subfn, index=False) 





