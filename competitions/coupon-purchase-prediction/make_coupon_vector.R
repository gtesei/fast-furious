library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 
check_coupon_data = function(coupon_data) {
  stopifnot(ncol(coupon_data)==166)
  
  idx.LIST.CAPSULE_TEXT = grep(pattern = 'LIST.CAPSULE_TEXT_',x = colnames(coupon_data))
  idx.LIST.GENRE_NAME = grep(pattern = 'LIST.GENRE_NAME_',x = colnames(coupon_data))
  idx.LIST.LARGE_AREA_NAME = grep(pattern = 'LIST.LARGE_AREA_NAME_',x = colnames(coupon_data))
  
  idx.AREA.SMALL_AREA_NAME = grep(pattern = 'AREA.SMALL_AREA_NAME_',x = colnames(coupon_data))
  idx.AREA.PREF_NAME = grep(pattern = 'AREA.PREF_NAME_',x = colnames(coupon_data))
  
  idx.LIST.BID_CATALOG_PRICE = grep(pattern = 'LIST.BID_CATALOG_PRICE_',x = colnames(coupon_data))
  idx.LIST.BID_PRICE_RATE = grep(pattern = 'LIST.BID_PRICE_RATE_',x = colnames(coupon_data))
  
  idx.LIST.WDISPPERIOD = grep(pattern = 'LIST.WDISPPERIOD_',x = colnames(coupon_data))
  
  ## these components must have norm = 1 
  comp_sum = apply(X = coupon_data[,idx.LIST.CAPSULE_TEXT] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  comp_sum = apply(X = coupon_data[,idx.LIST.CAPSULE_TEXT] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  comp_sum = apply(X = coupon_data[,idx.LIST.GENRE_NAME] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  comp_sum = apply(X = coupon_data[,idx.LIST.LARGE_AREA_NAME] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  comp_sum = apply(X = coupon_data[,idx.LIST.BID_CATALOG_PRICE] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  comp_sum = apply(X = coupon_data[,idx.LIST.BID_PRICE_RATE] ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(comp_sum)==nrow(coupon_data))
  
  ## other components (idx.AREA.SMALL_AREA_NAME,idx.AREA.PREF_NAME) must be treated differently  
}

normalize_coupon_data = function(coupon_data) {
  stopifnot(ncol(coupon_data)==166)
  
  idx.LIST.CAPSULE_TEXT = grep(pattern = 'LIST.CAPSULE_TEXT_',x = colnames(coupon_data))
  idx.LIST.GENRE_NAME = grep(pattern = 'LIST.GENRE_NAME_',x = colnames(coupon_data))
  idx.LIST.LARGE_AREA_NAME = grep(pattern = 'LIST.LARGE_AREA_NAME_',x = colnames(coupon_data))
  
  idx.AREA.SMALL_AREA_NAME = grep(pattern = 'AREA.SMALL_AREA_NAME_',x = colnames(coupon_data))
  idx.AREA.PREF_NAME = grep(pattern = 'AREA.PREF_NAME_',x = colnames(coupon_data))
  
  idx.LIST.BID_CATALOG_PRICE = grep(pattern = 'LIST.BID_CATALOG_PRICE_',x = colnames(coupon_data))
  idx.LIST.BID_PRICE_RATE = grep(pattern = 'LIST.BID_PRICE_RATE_',x = colnames(coupon_data))
  
  idx.LIST.WDISPPERIOD = grep(pattern = 'LIST.WDISPPERIOD_',x = colnames(coupon_data))
  
  ##### >>>>  nomalize 
  norm_int = function(c_data,indexList) {
    # e.g. indexList = list(idx.LIST.CAPSULE_TEXT)
    aa = lapply(indexList , function(idx) {
      sub_mat = c_data[,idx]
      weights = apply(X = sub_mat , MARGIN = 1 , FUN = sum)
      sub_mat_norm = sub_mat / weights
      c_data[,idx] <<- sub_mat_norm
    })
    return(c_data)
  }
  
  ## call to norm 
  coupon_data = norm_int(c_data = coupon_data , 
                         indexList = list(idx.LIST.CAPSULE_TEXT, idx.LIST.GENRE_NAME, idx.LIST.LARGE_AREA_NAME,
                                          idx.AREA.SMALL_AREA_NAME, idx.AREA.PREF_NAME,
                                          idx.LIST.BID_CATALOG_PRICE, idx.LIST.BID_PRICE_RATE, 
                                          idx.LIST.WDISPPERIOD))
  
  ## check 
  check_coupon_data(coupon_data)
  
  return(coupon_data)
}

isCouponVectorWellFormed = function(vect) {
  # vect = train_vect[train_vect$USER_ID_hash==uid,-1]
  stopifnot(length(vect)==165)
  idx.LIST.CAPSULE_TEXT = grep(pattern = 'LIST.CAPSULE_TEXT_',x = colnames(vect))
  idx.LIST.GENRE_NAME = grep(pattern = 'LIST.GENRE_NAME_',x = colnames(vect))
  idx.LIST.LARGE_AREA_NAME = grep(pattern = 'LIST.LARGE_AREA_NAME_',x = colnames(vect))
  
  idx.AREA.SMALL_AREA_NAME = grep(pattern = 'AREA.SMALL_AREA_NAME_',x = colnames(vect))
  idx.AREA.PREF_NAME = grep(pattern = 'AREA.PREF_NAME_',x = colnames(vect))
  
  idx.LIST.BID_CATALOG_PRICE = grep(pattern = 'LIST.BID_CATALOG_PRICE_',x = colnames(vect))
  idx.LIST.BID_PRICE_RATE = grep(pattern = 'LIST.BID_PRICE_RATE_',x = colnames(vect))
  
  ##
  stopifnot( sum(vect[,idx.LIST.CAPSULE_TEXT]) > 1-10^-9  ,  sum(vect[,idx.LIST.CAPSULE_TEXT]) < 1+10^-9 )
  stopifnot( sum(vect[,idx.LIST.GENRE_NAME]) > 1-10^-9  ,  sum(vect[,idx.LIST.GENRE_NAME]) < 1+10^-9 )
  stopifnot( sum(vect[,idx.LIST.LARGE_AREA_NAME]) > 1-10^-9  ,  sum(vect[,idx.LIST.LARGE_AREA_NAME]) < 1+10^-9 )
  
  stopifnot( sum(vect[,idx.AREA.SMALL_AREA_NAME]) > 1-10^-9  ,  sum(vect[,idx.AREA.SMALL_AREA_NAME]) < 1+10^-9 )
  #stopifnot( sum(vect[,idx.AREA.PREF_NAME]) > 1-10^-9  ,  sum(vect[,idx.AREA.PREF_NAME]) < 1+10^-9 )
  
  stopifnot( sum(vect[,idx.LIST.BID_CATALOG_PRICE]) > 1-10^-9  ,  sum(vect[,idx.LIST.BID_CATALOG_PRICE]) < 1+10^-9 )
  stopifnot( sum(vect[,idx.LIST.BID_PRICE_RATE]) > 1-10^-9  ,  sum(vect[,idx.LIST.BID_PRICE_RATE]) < 1+10^-9 )
}

getWeekBoundary <- function (week_number) {
  
  ## 2011-06-27 = min(coupon_list_train.meta$DISPFROM) monday
  ## getWeekBoundary(52)$first = "2011-06-26"
  
  ## 2012-06-30 = max(coupon_list_test.meta$DISPFROM) saturday 
  ## getWeekBoundary(0)$last = "2012-06-30"
  
  stopifnot(week_number>=0 , is.numeric(week_number))
  day0.in = as.Date('2012-06-24')
  day0.last = as.Date('2012-06-30')
  return(list(
    first = day0.in-(7*week_number), 
    last = day0.last-(7*week_number)
  ))
}

getWeekNumber = function (day) {
  day.from = as.Date('2012-06-24')
  return(ceiling(as.numeric(day.from-as.Date(day))/7))
} 

getLabels <- function (week_number , verbose=T) {
  stopifnot(week_number>=0 , is.numeric(week_number))
  
  labels = NULL
  coupons = NULL
  if (week_number > 0) {
    w1 = coupon_list_train.meta[coupon_list_train.meta$DISPFROM >= getWeekBoundary(week_number)$first & 
                                  coupon_list_train.meta$DISPFROM <= getWeekBoundary(week_number)$last,]
    w1_id = w1$COUPON_ID_hash
    coupons = w1_id
    if (verbose) cat(">>> labels for week ",week_number,":",length(coupons),"\n") 
    t1 = coupon_detail_train.meta[ coupon_detail_train.meta$COUPON_ID_hash %in% w1_id & 
                                     coupon_detail_train.meta$I_DATE >= getWeekBoundary(week_number)$first & 
                                     coupon_detail_train.meta$I_DATE <= getWeekBoundary(week_number)$last , ]
    
    t1_user_cp = ddply(t1 , .(USER_ID_hash,COUPON_ID_hash) , function(x) c(num=nrow(x)) )
    
    tmp = ddply(t1_user_cp, .(USER_ID_hash) , function(x) c(num=nrow(x)) )
    max_cp_pur = max(tmp$num)
    if (verbose) cat(">>> max number of different coupons purchased by a user in week ",week_number,":",max_cp_pur,"\n") 
    
    #     t1_user = ddply(t1_user_cp, .(USER_ID_hash) , function(x) {
    #       bs = unique(x$COUPON_ID_hash)
    #       ret = c(bs,rep('NA_COUPON',(max_cp_pur-length(bs))))
    #       return(setNames(object = ret , nm = paste0('PURCHASED_COUPONS_',1:max_cp_pur)))
    #     } )
    
    t1_user = ddply(t1_user_cp, .(USER_ID_hash) , function(x) {
      bs = unique(x$COUPON_ID_hash)
      ret = paste(bs , collapse = ' ')
      return(setNames(object = ret , nm = 'PURCHASED_COUPONS'))
    } )
    
    
    labels = merge(x = user_data[,'USER_ID_hash' , drop=F] , y = t1_user , by='USER_ID_hash' , all = T)
    labels[is.na(labels)] = ''
  } else {
    ## this is the test area 
    labels = NULL ## predict this is the goal of the competition :-)) 
    coupons = coupon_list_test.meta$COUPON_ID_hash
  }
  
  return(list(
    labels = labels, 
    coupons = coupons
  ))
}



### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')

### CONFIG 
#debug = T

### DATA
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')
coupon_visit = as.data.frame( fread(paste(ff.getPath("data") , "coupon_visit_train.csv" , sep='')))
source(paste0(ff.getPath('code'),'enc_data_base.R'))

### RESAMPLING: 
### 0-train set / test set 
###        - train lables: find coupons that last week of train set were activated and make them train labels 
###        - train features: compute train features without these coupons and using transaction data till a week before the end of train period 
###        - test lables: test labels are the 310 provided 
###        - test features: compute test features using all train coupons and using transaction data till the end of train period  
### 1-train set / croos-validation set / test set
###        - train lables: find coupons activated between 2 and 1 weeks before the end of train period and make them train labels 
###        - train features: compute features without these coupons and using transaction data till 2 weeks before the end of train period  
###        - xval labels: find coupons that were activated the week before the end of train period and make them cross-validation labels 
###        - xval features: compute cross validation features without these coupons and using transaction data till 1 week before the end of train period  
###        - use cross validation to tune models (e.g. nrounds) and asses model performance
###        - pred-train lables: find coupons that last week of train set were activated and make them train labels 
###        - pred-train features: compute train features without these coupons and using transaction data till a week before the end of train period 
###        - test lables: test labels are the 310 provided 
###        - test features: compute test features using all train coupons and using transaction data till the end of train period  
### x(>1)-train set / croos-validation set / test set
###        - train lables: find coupon activated between (x+1) and x weeks before the end of train period and make them train labels 
###        - train features: compute features without these coupons and using transaction data till (x+1) weeks before the end of train period  
###        - xval labels: find coupons that were activated the x weeks before the end of train period and make them cross-validation labels 
###        - xval features: compute cross validation features without these coupons and using transaction data till x weeks before the end of train period  
###        - use cross validation to tune models (e.g. nrounds) and asses model performance
###        - pred-train lables: find coupons that last week of train set were activated and make them train labels 
###        - pred-train features: compute train features without these coupons and using transaction data till a week before the end of train period 
###        - test lables: test labels are the 310 provided 
###        - test features: compute test features using all train coupons and using transaction data till the end of train period  

### NOTE: the bigger x, the bigger the distance between train data / test data, hence the more the problems. 
###       Ideally, x = 1 should be the best choice, but it could be a good idea trying x = 2,3,4  
###       Maybe, it could be a better idea ensebling models with x = 2,3,4 

### STACKING
###        - (x): train meta-features are the cross-validation predictions of base learner / train labels are cross-validations labels of base learner 
###        - any tranformation applied to features / labels 
###        - type of base learner (e.g. KNN)
###        - any tuning parameter of the base learner. This parameter is contextualized by the previous one (e.g. nrounds works for xgboost but not for nnet)
###
###        ===> <x>_<trans>*_<base_learner>_<tuning_param>*.csv

### NOTE: being MAP@10 the metric, ensembles are .csv files with 10 columns (= top 10 highest probabilty predicted labels)

### PLAN 
###           - pre-check: every week ~310 coupons are activated and that average DISPPERIOD ~6 days   
###        ** getCoupons(x,type=train / xval / pred-train / test): extract from <coupon_list_train> extract coupons activated between (x+1) and x weeks before the end of train period 
###                              coupons activated after x weeks before the end of train period are discarded
###           - var: encode them as 0,1, ... class for xgboost that like 0-based label indeces
###           - var: make factors for caret-based models 
###
###            - pre-condition: categorical variables like CAPSULE_TEXT must be encoded 
###        ** makeFeature(couponList,user_id,exp_smooth): given a list of coupons compute features like 
###                   - CAPSULE_TEXT_1 ~ # coupons bougth by the user in couponList (exp. smoothed) having  CAPSULE_TEXT = CAPSULE_TEXT_1
###                   - CATALOG_PRICE ~ average/sd (exp. smoothed) of coupons in couponList bougth by the user
###                   ... 

### DISPFROM belongs to [2012-06-24 (Sunday) -  2012-06-30 (Saturday)] - (0) test labels DISPFROM
### DISPFROM belongs to [2012-06-17 (Sunday) -  2012-06-23 (Saturday)] - (1) pred-train labels   
### DISPFROM belongs to [2012-06-10 (Sunday) -  2012-06-16 (Saturday)] - (2)
### DISPFROM belongs to [2012-06-3 (Sunday)  -  2012-06-9  (Saturday)] - (3)
### ... 
### DISPFROM belongs to [2011-06-27 (Monday) -  2011-07-2 (Saturday)] - (...) i.e. first sampling week  


##### FEATURE ENGINEEERING 
### WDISPPERIOD time difference in days between the week number DISFROM Saturday and DISFROM  
##    i.e. getWeekBoundary(getWeekNumber("2012-06-27"))$last - as.Date("2012-06-27")

cat(">>> FEATURE ENGINEEERING: computing WDISPPERIOD (as categorical), i.e. time difference in days between Saturday week's DISFROM and DISFROM ... \n") 
getActualDispPeriod = function(x) {
  unlist(Map(function(df,dp) {
    min(dp,as.numeric(getWeekBoundary(week_number = getWeekNumber(day = df))$last - as.Date(df)))
  } , df = x$DISPFROM , dp = x$DISPPERIOD))
}

coupon_list_train.meta$WDISPPERIOD = getActualDispPeriod(coupon_list_train.meta)
coupon_list_test.meta$WDISPPERIOD = getActualDispPeriod(coupon_list_test.meta)
#################
## identify sterile coupons 
coupon_usable_test.meta[is.na(coupon_usable_test.meta)] <- 1
coupon_usable_train.meta[is.na(coupon_usable_train.meta)] <- 1

coupon_usable_train.meta = merge(x = coupon_list_train.meta[,c('COUPON_ID_hash','WDISPPERIOD') , ] , y=coupon_usable_train.meta, by='COUPON_ID_hash' )
coupon_usable_test.meta = merge(x = coupon_list_test.meta[,c('COUPON_ID_hash','WDISPPERIOD') , ] , y=coupon_usable_test.meta, by='COUPON_ID_hash' )

coupon_usable_test.meta$WDISPPERIOD_REAL = NA
lapply(1:nrow(coupon_usable_test.meta) , function(i) {
  wp_i = coupon_usable_test.meta[i, ]$WDISPPERIOD 
  wp_i_real = sum(coupon_usable_test.meta[i, 9]>0) ## SUNDAY 
  if (wp_i>0) {
    wp_i_real = wp_i_real + sum(coupon_usable_test.meta[i, 3:(3+wp_i-1)]>0)  
  }
  coupon_usable_test.meta[i,]$WDISPPERIOD_REAL <<- wp_i_real
})

coupon_usable_train.meta$WDISPPERIOD_REAL = NA
lapply(1:nrow(coupon_usable_train.meta) , function(i) {
  wp_i = coupon_usable_train.meta[i, ]$WDISPPERIOD 
  wp_i_real = sum(coupon_usable_train.meta[i, 9]>0) ## SUNDAY 
  if (wp_i>0) {
    wp_i_real = wp_i_real + sum(coupon_usable_train.meta[i, 3:(3+wp_i-1)]>0)
  }
  coupon_usable_train.meta[i,]$WDISPPERIOD_REAL <<- wp_i_real
})
coupon_list_train.meta = merge(x = coupon_usable_train.meta[,c('COUPON_ID_hash','WDISPPERIOD_REAL') , ] , y=coupon_list_train.meta, by='COUPON_ID_hash' )
coupon_list_test.meta = merge(x = coupon_usable_test.meta[,c('COUPON_ID_hash','WDISPPERIOD_REAL') , ] ,   y=coupon_list_test.meta,  by='COUPON_ID_hash' )
coupon_list_train.meta$WDISPPERIOD = coupon_list_train.meta$WDISPPERIOD_REAL
coupon_list_test.meta$WDISPPERIOD = coupon_list_test.meta$WDISPPERIOD_REAL
coupon_list_train.meta = coupon_list_train.meta[,-grep(pattern = 'WDISPPERIOD_REAL' , colnames(coupon_list_train.meta))]
coupon_list_test.meta = coupon_list_test.meta[,-grep(pattern = 'WDISPPERIOD_REAL' , colnames(coupon_list_test.meta))]

coupon_list_train.meta$WDISPPERIOD_NORM = coupon_list_train.meta$WDISPPERIOD/7
coupon_list_test.meta$WDISPPERIOD_NORM = coupon_list_test.meta$WDISPPERIOD/7

## DISCOUNT_PRICE
coupon_list_train.meta$DISCOUNT_PRICE_LOG = 1/log10(coupon_list_train.meta$DISCOUNT_PRICE)
coupon_list_train.meta$DISCOUNT_PRICE_LOG_NORM = (coupon_list_train.meta$DISCOUNT_PRICE_LOG - min(coupon_list_train.meta$DISCOUNT_PRICE_LOG)) / 
  max(coupon_list_train.meta$DISCOUNT_PRICE_LOG)

coupon_list_test.meta$DISCOUNT_PRICE_LOG = 1/log10(coupon_list_test.meta$DISCOUNT_PRICE)
coupon_list_test.meta$DISCOUNT_PRICE_LOG_NORM = (coupon_list_test.meta$DISCOUNT_PRICE_LOG - min(coupon_list_test.meta$DISCOUNT_PRICE_LOG)) / 
  max(coupon_list_test.meta$DISCOUNT_PRICE_LOG)

## PRICE_RATE
coupon_list_train.meta$PRICE_RATE_NORM = coupon_list_train.meta$PRICE_RATE / 100 
coupon_list_test.meta$PRICE_RATE_NORM = coupon_list_test.meta$PRICE_RATE / 100 

#################

head(coupon_list_train.meta[,c('DISPFROM','DISPEND','DISPPERIOD','WDISPPERIOD')])
head(coupon_list_test.meta[,c('DISPFROM','DISPEND','DISPPERIOD','WDISPPERIOD')])
describe(x = c(coupon_list_train.meta$WDISPPERIOD,coupon_list_test.meta$WDISPPERIOD) )

wp_cat = ff.encodeCategoricalFeature(data.train = coupon_list_train.meta$WDISPPERIOD , 
                                     data.test = coupon_list_test.meta$WDISPPERIOD , 
                                     colname.prefix = 'LIST.WDISPPERIOD')

coupon_data_train = cbind(coupon_data_train ,wp_cat$traindata)
coupon_data_test = cbind(coupon_data_test ,wp_cat$testdata)
rm(wp_cat)

## drop LIST.DISPPERIOD column 
coupon_data_train = coupon_data_train[,-grep(pattern = 'LIST.DISPPERIOD',x = colnames(coupon_data_train)) , drop=F]
coupon_data_test = coupon_data_test[,-grep(pattern = 'LIST.DISPPERIOD',x = colnames(coupon_data_test)) , drop=F]

### LIST.CATALOG_PRICE: making discrete 
cat(">>> FEATURE ENGINEEERING: discretizing LIST.CATALOG_PRICE ... \n")
describe(x = c(coupon_list_train.meta$CATALOG_PRICE,coupon_list_test.meta$CATALOG_PRICE) )
q_pr = quantile(x = c(coupon_list_train.meta$CATALOG_PRICE,coupon_list_test.meta$CATALOG_PRICE) , probs = (100/6*1:5)/100)
print(q_pr)

bidPrice = function(x) {
  unlist(lapply(x$LIST.CATALOG_PRICE,function(x){
    if (x < q_pr[1]) return(1)
    else return(sum(x >= q_pr))
  }))
}

coupon_data_train$LIST.CATALOG_PRICE_BID = bidPrice(coupon_data_train)
coupon_data_test$LIST.CATALOG_PRICE_BID = bidPrice(coupon_data_test)
head(coupon_data_train[,c('LIST.CATALOG_PRICE_BID','LIST.CATALOG_PRICE')])
head(coupon_data_test[,c('LIST.CATALOG_PRICE_BID','LIST.CATALOG_PRICE')])

bp_cat = ff.encodeCategoricalFeature(data.train = coupon_data_train$LIST.CATALOG_PRICE_BID , 
                                     data.test = coupon_data_test$LIST.CATALOG_PRICE_BID , 
                                     colname.prefix = 'LIST.BID_CATALOG_PRICE')

coupon_data_train = cbind(coupon_data_train ,bp_cat$traindata)
coupon_data_test = cbind(coupon_data_test ,bp_cat$testdata)
rm(bp_cat)
rm(q_pr)

## drop LIST.CATALOG_PRICE column 
coupon_data_train = coupon_data_train[,-grep(pattern = 'LIST.CATALOG_PRICE',x = colnames(coupon_data_train)) , drop=F]
coupon_data_test = coupon_data_test[,-grep(pattern = 'LIST.CATALOG_PRICE',x = colnames(coupon_data_test)) , drop=F]

### LIST.PRICE_RATE: making discrete 
cat(">>> FEATURE ENGINEEERING: discretizing LIST.PRICE_RATE ... \n")
describe(x = c(coupon_list_train.meta$PRICE_RATE,coupon_list_test.meta$PRICE_RATE) )
q_ds = quantile(x = c(coupon_list_train.meta$PRICE_RATE,coupon_list_test.meta$PRICE_RATE) , probs = (100/6*1:5)/100)
print(q_ds)
cat(">>> using these quantiles doesn't make sense business side --> Replacing with ... \n")
q_ds = c(49,60,70)
print(q_ds)

bidDisc = function(x) {
  unlist(lapply(x$LIST.PRICE_RATE,function(x){
    if (x < q_ds[1]) return(1)
    else return(sum(x >= q_ds))
  }))
}

coupon_data_train$LIST.PRICE_RATE_BID = bidDisc(coupon_data_train)
coupon_data_test$LIST.PRICE_RATE_BID = bidDisc(coupon_data_test)
head(coupon_data_train[,c('LIST.PRICE_RATE_BID','LIST.PRICE_RATE')])
head(coupon_data_test[,c('LIST.PRICE_RATE_BID','LIST.PRICE_RATE')])

bd_cat = ff.encodeCategoricalFeature(data.train = coupon_data_train$LIST.PRICE_RATE_BID , 
                                     data.test = coupon_data_test$LIST.PRICE_RATE_BID , 
                                     colname.prefix = 'LIST.BID_PRICE_RATE')

coupon_data_train = cbind(coupon_data_train ,bd_cat$traindata)
coupon_data_test = cbind(coupon_data_test ,bd_cat$testdata)
rm(bd_cat)
rm(q_ds)

## drop LIST.PRICE_RATE column 
coupon_data_train = coupon_data_train[,-grep(pattern = 'LIST.PRICE_RATE',x = colnames(coupon_data_train)) , drop=F]
coupon_data_test = coupon_data_test[,-grep(pattern = 'LIST.PRICE_RATE',x = colnames(coupon_data_test)) , drop=F]

## drop LIST.DISCOUNT_PRICE column: 0-information predictor  
coupon_data_train = coupon_data_train[,-grep(pattern = 'LIST.DISCOUNT_PRICE',x = colnames(coupon_data_train)) , drop=F]
coupon_data_test = coupon_data_test[,-grep(pattern = 'LIST.DISCOUNT_PRICE',x = colnames(coupon_data_test)) , drop=F]

##### LABELS VECTORS 
## first trans date: 2011-07-01 (Friday)
## >>> first train week: 2011-07-03 / 2011-07-09 == 51th week 
## >>> first labels week: 50th week == 2011-07-10 / 2011-07-16
## >>> 0 ... 50 (label coupon vectors)
## >>> 0 ... 51 (user coupon vectors)

getLabelVector <- function (week_number , removeCOUPON_ID_hash =F , verbose=T) {
  label_vector = NULL
  
  labels = getLabels(week_number = week_number , verbose =verbose)
  if (week_number >= 1) {
    label_vector = coupon_data_train[coupon_data_train$COUPON_ID_hash %in% labels$coupons,]  
  } else {
    label_vector = coupon_data_test[coupon_data_test$COUPON_ID_hash %in% labels$coupons,]  
  }
  
  if(removeCOUPON_ID_hash) {
    label_vector = label_vector[,-1]
  }
  
  return(label_vector)
}

##labelVect0 = getLabelVector(week_number = 0) ## test label vectors 
##labelVect11 = getLabelVector(week_number = 11) ## train label vectors week n. 11 


##### USER VECTORS 

getDummyCouponVector <- function(r=1) {
  dummy_coupon_vector = coupon_data_train[1:r,-1]
  dummy_coupon_vector[] = 0
  return(dummy_coupon_vector)
}

getTrainCoupon <- function (week_number , verbose=T) {
  stopifnot(week_number>=0 , is.numeric(week_number))
  
  coupons = NULL
  
  w1 = coupon_list_train.meta[coupon_list_train.meta$DISPFROM < getWeekBoundary(week_number)$first ,]
  coupons = w1$COUPON_ID_hash
  if (verbose) cat(">>> train coupons for week ",week_number,":",length(coupons),"\n") 
  
  return(coupons)
}

getUserVector <- function(week_number , exp_smooth = 0, verbose=T, debug=F) {
  stopifnot(week_number<=50)
  coupons = getTrainCoupon(week_number=week_number , verbose=verbose)
  trans = coupon_detail_train.meta[coupon_detail_train.meta$COUPON_ID_hash %in% coupons , ]
  
  coupon_data_train_red = coupon_data_train[coupon_data_train$COUPON_ID_hash %in% coupons , ]
  
  ### computing exponential smoothing
  coupon_meta_red = NULL
  if (exp_smooth != 0) {
    if (verbose) cat(">>> computing exponential smoothing ",exp_smooth,"...\n")
    ## exp_smooth: ascendind (>0) / descending (<0)
    ## exp_smooth = 1 --> exp_smooth2_0.75 = 1/50 * log(0.75); sales_smooth2_0.75 = exp((50-weeks)*exp_smooth2_0.75)
    ## exp_smooth = 2 --> exp_smooth2_0.5 = 1/50 * log(1/2); sales_smooth2_0.5 = exp((50-weeks)*exp_smooth2_0.5)
    ## exp_smooth = 3 --> exp_smooth2_0.25 = 1/50 * log(0.25); sales_smooth2_0.25 = exp((50-weeks)*exp_smooth2_0.25)
    ## exp_smooth = 4 --> exp_smooth2_0.05 = 1/50 * log(0.05); sales_smooth2_0.05 = exp((50-weeks)*exp_smooth2_0.05)
    ## exp_smooth = -1 --> exp_smooth_0.75 = 1/50 * log(0.75); sales_smooth_0.75 = exp((weeks)*exp_smooth_0.75)
    ## exp_smooth = -2 --> exp_smooth_0.5 = 1/50 * log(1/2); sales_smooth_05 = exp((weeks)*exp_smooth_0.5)
    ## exp_smooth = -3 --> exp_smooth_0.25 = 1/50 * log(1/4); sales_smooth_0.25 = exp((weeks)*exp_smooth_0.25)
    ## exp_smooth = -4 --> exp_smooth_0.05 = 1/50 * log(0.05); sales_smooth_0.05 = exp((weeks)*exp_smooth_0.05)
    
    ## compute week diffs 
    coupon_meta_red = coupon_list_train.meta[coupon_list_train.meta$COUPON_ID_hash %in% coupons , ]
    coupon_meta_red$x = getWeekNumber(coupon_meta_red$DISPFROM) - week_number
    
    ## check 
    stopifnot(sum(coupon_meta_red$x<0)==0)
    
    
    ## compute smooth_coeff 
    if (exp_smooth==1)   {
      alpha = 1/50 * log(0.75)
      coupon_meta_red$smooth_coeff = exp((50-coupon_meta_red$x)*alpha)
    } else if (exp_smooth==2)   {
      alpha = 1/50 * log(1/2)
      coupon_meta_red$smooth_coeff = exp((50-coupon_meta_red$x)*alpha)
    } else if (exp_smooth==3)   {
      alpha = 1/50 * log(0.25)
      coupon_meta_red$smooth_coeff = exp((50-coupon_meta_red$x)*alpha)
    } else if (exp_smooth==4)   {
      alpha = 1/50 * log(0.05)
      coupon_meta_red$smooth_coeff = exp((50-coupon_meta_red$x)*alpha)
    } else if (exp_smooth==-1)  {
      alpha = 1/50 * log(0.75)
      coupon_meta_red$smooth_coeff = exp((coupon_meta_red$x)*alpha)
    } else if (exp_smooth==-2)  {
      alpha = 1/50 * log(1/2)
      coupon_meta_red$smooth_coeff = exp((coupon_meta_red$x)*alpha)
    } else if (exp_smooth==-3)  {
      alpha = 1/50 * log(1/4)
      coupon_meta_red$smooth_coeff = exp((coupon_meta_red$x)*alpha)
    } else if (exp_smooth==-4)  {
      alpha = 1/50 * log(0.05)
      coupon_meta_red$smooth_coeff = exp((coupon_meta_red$x)*alpha)
    } else {
      stop(paste0("unknown exp_smooth:",exp_smooth))
    }
    
    ##
    coupon_meta_red = coupon_meta_red[,c('COUPON_ID_hash','smooth_coeff')]
    tmp0 = merge(x=coupon_data_train_red,y=coupon_meta_red,by='COUPON_ID_hash',all = F)
    
    #     ss = ddply(coupon_data_train_red , .(COUPON_ID_hash) , function(x) {
    #       x[2:(length(colnames(coupon_data_train_red))-1)]*rep(x[length(colnames(coupon_data_train_red))],(length(colnames(coupon_data_train_red))-2))
    #     })
    
    sm_mat = matrix(rep(tmp0$smooth_coeff,length(colnames(tmp0))-2), nrow = nrow(tmp0))
    tmp = sm_mat * tmp0[,-c(1,length(colnames(tmp0)))]
    tmp = cbind(COUPON_ID_hash=tmp0$COUPON_ID_hash,tmp)
    
    ## normalize 
    tmp = normalize_coupon_data(coupon_data=tmp) 
    
    if (debug) {
      #       dsize = 10 
      #       cps = sample(coupons,size = dsize)
      #       for (cp in cps) {
      #         tt = coupon_meta_red[coupon_meta_red$COUPON_ID_hash==cp,]$smooth_coeff * coupon_data_train_red[coupon_data_train_red$COUPON_ID_hash==cp,-1]
      #         stopifnot(sum(abs(tt - tmp[tmp$COUPON_ID_hash==cp,-1]))==0)
      #       }
      #       stopifnot(nrow(tmp)==nrow(coupon_data_train_red))
      #       stopifnot(ncol(tmp)==ncol(coupon_data_train_red))
      #       stopifnot(length(tmp$COUPON_ID_hash)==length(coupon_data_train_red$COUPON_ID_hash))
      #       stopifnot(length(unique(tmp$COUPON_ID_hash))==length(unique(coupon_data_train_red$COUPON_ID_hash)))
      #       cat(">>> checked exponential smoothing: [",dsize,"/",dsize,"] OK \n")
    }
    
    ## passing smoothed vectors 
    coupon_data_train_red = tmp
  }
  
  ### rep VISIT 
  coupon_visit
  
  ### rep ITEM_COUNT
  trans_ext = trans[,c('USER_ID_hash','COUPON_ID_hash','SMALL_AREA_NAME','ITEM_COUNT')] 
  delta_rows = sum(trans$ITEM_COUNT) - nrow(trans)
  trans_ext_delta = data.frame(USER_ID_hash=rep(NA,delta_rows), COUPON_ID_hash=rep(NA,delta_rows), 
                               SMALL_AREA_NAME=rep(NA,delta_rows) , ITEM_COUNT=rep(NA,delta_rows))
  nrf = 1 
  aa = lapply( 1:nrow(trans_ext), function(i){
    nt = trans_ext[i,]$ITEM_COUNT
    if (nt>1) {
      lapply (2:nt, function(j) {
        trans_ext_delta[nrf,] <<- trans_ext[i,]
        nrf <<- nrf + 1 
      })
    }
  })
  stopifnot(sum(is.na(trans_ext_delta))==0)
  trans_ext = rbind(trans_ext,trans_ext_delta)

  ### counputing user SMALL_AREA_NAME 
  trans_ext = trans_ext[,c('USER_ID_hash','COUPON_ID_hash','SMALL_AREA_NAME')] 
  l = ff.encodeCategoricalFeature(data.train = trans_ext$SMALL_AREA_NAME,
                                  data.test = trans_ext$SMALL_AREA_NAME , 
                                  colname.prefix = "SMALL_AREA_NAME",
                                  asNumericSequence = T)
  trans_ext = cbind(trans_ext,l$traindata)
  
  ##exponential smoothing on user SMALL_AREA_NAME
  if (exp_smooth != 0) {
    tmp0 = merge(x=trans_ext,y=coupon_meta_red,by='COUPON_ID_hash',all = F)  
    
    sm_mat = matrix(rep(tmp0$smooth_coeff,length(colnames(tmp0))-4), nrow = nrow(tmp0))
    tmp = sm_mat * tmp0[,-c(1:3,length(colnames(tmp0)))]
    tmp = cbind(data.frame(USER_ID_hash=tmp0$USER_ID_hash,
                           COUPON_ID_hash=tmp0$COUPON_ID_hash, 
                           SMALL_AREA_NAME=tmp0$SMALL_AREA_NAME),
                tmp)
    ## 
    trans_ext = tmp
  }
  
  trans_ext_aggr = ddply(trans_ext , .(USER_ID_hash) , function(x) {
    apply(X = x[,4:ncol(x)] ,MARGIN = 2 , sum)  
  }) 
  weights = apply(X = trans_ext_aggr[,-1] , MARGIN = 1 , FUN = sum)
  trans_ext_aggr_norm = trans_ext_aggr[,-1] / weights
  
  # check 
  tt = apply(X = trans_ext_aggr_norm ,MARGIN = 1 , FUN = sum)
  stopifnot(sum(tt)==nrow(trans_ext_aggr_norm)) ## all 1s 
  
  trans_ext_aggr_norm = cbind(USER_ID_hash=trans_ext_aggr$USER_ID_hash , trans_ext_aggr_norm)
  colnames(trans_ext_aggr_norm)[-1] = paste0("AREA.",colnames(trans_ext_aggr_norm)[-1])
  
  #########
  ## >>>>>> remove AREA.SMALL_AREA_NAME_x from coupon_data_train_red
  coupon_data_train_red = coupon_data_train_red[,-grep(pattern = "AREA.SMALL_AREA_NAME_" , 
                                                       x = colnames(coupon_data_train_red))]
  
  ### TODO VISIT 
  trans = trans_ext[,c('USER_ID_hash','COUPON_ID_hash')] 
  user_vect_ext_trans = merge(x = trans , y = coupon_data_train_red , by='COUPON_ID_hash' , all.x = T , all.y = F)
  user_vect_trans = ddply( user_vect_ext_trans , .(USER_ID_hash) , function(x) {
    return(apply(x[,-c(1,2)],MARGIN = 2 ,FUN = mean))
  })
  
  ## >>>>>> add AREA.SMALL_AREA_NAME_x to user_vect_trans
  user_vect_trans = merge(x = trans_ext_aggr_norm , y = user_vect_trans , by='USER_ID_hash' ,all = T)
  stopifnot(sum(is.na(sum(is.na(user_vect_trans))))==0)
  
  ###
  if (debug && exp_smooth==0) {
    #     uids = unique(trans$USER_ID_hash)
    #     cfr = 0
    #     for (uid in uids) {
    #       cid = trans[trans$USER_ID_hash == uid,]$COUPON_ID_hash
    #       if (length(cid) == length(unique(cid))) {
    #         cfr = cfr + 1 
    #         aa = as.numeric(user_vect_trans[user_vect_trans$USER_ID_hash == uid , -1])-as.numeric(apply(coupon_data_train[coupon_data_train$COUPON_ID_hash %in% cid,-1] , MARGIN = 2 , FUN = mean))
    #         if(sum(abs(aa))!=0) stop(paste0("error for uid:",uid))
    #       }
    #     }
    #     cat(">>> checked aggregating: [",cfr,"/",length(uids),"] OK\n")
  }
  
  ## merging users 0-purchases 
  users0 = setdiff(user_data$USER_ID_hash , user_vect_trans$USER_ID_hash)
  user_vect_trans0 = cbind(USER_ID_hash=users0,getDummyCouponVector(r = length(users0)))
  
  user_vect = rbind(user_vect_trans,user_vect_trans0)
  
  
  ## checks 
  stopifnot(sum(abs(user_vect[user_vect$USER_ID_hash %in% users0 , -1])) == 0)
  stopifnot(sum(abs(user_vect[!user_vect$USER_ID_hash %in% users0 , -1])) != 0)
  stopifnot(sum(is.na(user_vect))==0)
  
  return(user_vect)
}

#check 
cat(">>> checking data ...\n")
check_coupon_data(coupon_data=coupon_data_train)
check_coupon_data(coupon_data=coupon_data_test)

if (debug) {
  cat(">>> debug mode activated ... checking congruence between user vector with and without exp smoothing ... \n")
  uv0 = getUserVector(week_number = 50,debug = T)
  uv2 = getUserVector(week_number = 50,exp_smooth = 2,debug = T)
  uv_2 = getUserVector(week_number = 50,exp_smooth = -2,debug = T)  
  for(ii in sample(x = 1:1000,size = 10)) {
    isCouponVectorWellFormed(vect = uv0[ii,])
  }
  stopifnot(sum(is.na(uv0))==0)
  lv0 = getLabelVector(week_number = 0)
  stopifnot(sum(is.na(lv0))==0)
  uv0 = getUserVector(week_number = 0,debug = T)
  for(ii in sample(x = 1:1000,size = 10)) {
    isCouponVectorWellFormed(vect = uv0[ii,])
  }
  cat(">>> debug: OK ...\n")
}



