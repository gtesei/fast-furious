import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import re 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import nltk 

try: 
    from nltk.corpus import stopwords
except:
    print('> trying to download stopwords...')
    nltk.download('stopwords')
    from nltk.corpus import stopwords

#stops = set(stopwords.words("english"))
stops = set(stopwords.words('russian'))

### FUNC ########################################################################
def xgb_feat_importance(model,cols,file_name):
    print('-----> Feature importance ... ')
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(cols))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()), 'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': cols})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    feature_importance.sort_values(by='importance', ascending=False)
    print(feature_importance.sort_values)
    feature_importance.to_csv(file_name, index=False)

def add_avg_per(df,what_to_avg,on,new_name,include_delta=True,include_perc=True):
    if type(on) == str:
        _full = [on,what_to_avg]
        _fulla = [on,new_name]
    elif type(on) == list:
        _full = on.copy()
        _full.append(what_to_avg)
        _fulla = on.copy()
        _fulla.append(new_name)
    else:
        raise Exception('what type is on!')
    _avg = df.groupby(on)[_full].mean()
    _avg.columns = _fulla
    prev_len = len(df)
    df = df.merge(_avg,how='inner' , on=on)
    assert len(df) == prev_len
    if include_delta:
        df[str(new_name+'_delta')] = df[what_to_avg] - df[new_name]
    if include_perc:
        df[str(new_name+'_perc')] = (df[what_to_avg] - df[new_name])/df[new_name]
    return df 
    
def count_desc_len(x,remove_stopwords=False,stops=stops):
    if not x:
        return 0
    if type(x) != str:
        return 0 
    if len(x) == 0:
        return 0 
    if not remove_stopwords:
        return len(x.split())
    else:
        return len([w for w in [i.lower() for i in x.split()] if not w in stops])
        
def encode_dataset(train,test,meta,target_model='xgb'):
    y_train = train[meta['target']]
    train = train.drop([meta['target']],axis=1)
    assert train.shape[1] == test.shape[1]
    for i in range(train.shape[1]):
        assert train.columns[i] == test.columns[i]
    train_obs = len(train)
    #
    all_data = pd.concat([train,test],axis=0)
    for i,f in enumerate(meta['cols'].keys()):
        print(i,f,meta['cols'][f])
        if meta['cols'][f] == 'CAT':
            all_data[f] = all_data[f].fillna('missing')
            encoder = LabelEncoder()
            encoder.fit(all_data[f])
            if target_model == 'xgb':
                all_data[f] = encoder.transform(all_data[f])
            else:
                all_data[f] = encoder.transform(all_data[f]).astype(int)
        elif meta['cols'][f] == 'NUM':
            all_data[f] = all_data[f].fillna(-1)
        elif meta['cols'][f] == 'DATE':
            tmp = pd.to_datetime(all_data[f])
            all_data[f] = tmp.dt.weekday
            cal = calendar()
            #holidays = cal.holidays(start=tmp.min(), end=tmp.max())
            #$all_data[f+'_is_holiday'] = 1*tmp.isin(holidays)
        elif meta['cols'][f] == 'REM':
            all_data = all_data.drop(f,axis=1)
        elif meta['cols'][f] == 'LEN':
            all_data[f+'_len'] = all_data[f].apply(count_desc_len)
            all_data = all_data.drop(f,axis=1)
        else:
            raise Exception(str(meta['cols'][f])+":unknown mapping")
    assert train_obs == len(y_train)
    return all_data , y_train
#################################################################################




