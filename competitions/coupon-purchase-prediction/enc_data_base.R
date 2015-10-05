library(fastfurious)
library(data.table)
library(plyr)

### FUNCS
wn = function (day.from = "2011-07-01",day) {
  return(1 + floor(as.numeric(as.Date(day)-as.Date(day.from))/7))
}

jp2num = function (levels, value) {
  stopifnot(length(value)==1)
  idx = which(levels == value)
  if (length(idx)==0) stop(paste0('unknown value:',value))
  if (length(idx)>1) stop(paste0('more than 1 match for value:',value))
  return (idx)
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'process' , sub_path = 'data_process')

### DATA
users = as.data.frame( fread(paste(ff.getPath("data") , "user_list.csv" , sep=''))) 

coupon_list_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_list_train.csv" , sep='')))
coupon_list_test = as.data.frame( fread(paste(ff.getPath("data") , "coupon_list_test.csv" , sep='')))

coupon_area_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_area_train.csv" , sep='')))
coupon_area_test = as.data.frame( fread(paste(ff.getPath("data") , "coupon_area_test.csv" , sep='')))

coupon_detail_train = as.data.frame( fread(paste(ff.getPath("data") , "coupon_detail_train.csv" , sep='')))

### USERS 
users$SEX_ID = ifelse(users$SEX_ID == 'f',1,0)
users = users[,-4] # cut WITHDRAW_DATE --> 21935 NAs and using MAP@10 as metric is not important predicting NOT sales   
users$REG_DATE = as.Date(users$REG_DATE)

### COUPON_LISTS 

## > 36% NAs values 
drop_columns_2 = paste0("USABLE_DATE_MON|USABLE_DATE_TUE|USABLE_DATE_WED|USABLE_DATE_THU|USABLE_DATE_FRI|USABLE_DATE_SAT|USABLE_DATE_SUN|")
drop_columns_2 = paste0(drop_columns_2,"USABLE_DATE_HOLIDAY|USABLE_DATE_BEFORE_HOLIDAY")

# save a copy for meta purpouses 
coupon_usable_train.meta = coupon_list_train[,grep(pattern = paste0("COUPON_ID_hash|",drop_columns_2) , x = colnames(coupon_list_train))]
coupon_usable_test.meta = coupon_list_test[,grep(pattern = paste0("COUPON_ID_hash|",drop_columns_2) , x = colnames(coupon_list_test))]

# drop columns  
drop_columns = paste0("VALIDFROM|VALIDEND|VALIDPERIOD|",drop_columns_2)
coupon_list_train = coupon_list_train[,-grep(pattern = drop_columns , x = colnames(coupon_list_train))]
coupon_list_test = coupon_list_test[,-grep(pattern = drop_columns , x = colnames(coupon_list_test))]


### COUPON_AREA 

### COUPON_DETAIL 

### LEVELS
PREF_NAME.levels = unique(c(users$PREF_NAME,
                          coupon_list_train$ken_name,coupon_list_test$ken_name, 
                          coupon_area_train$PREF_NAME,coupon_area_test$PREF_NAME))

SMALL_AREA_NAME.levels = unique(c(coupon_list_train$small_area_name,coupon_list_test$small_area_name, 
                                  coupon_detail_train$SMALL_AREA_NAME, 
                                  coupon_area_train$SMALL_AREA_NAME,coupon_area_test$SMALL_AREA_NAME))

LARGE_AREA_NAME.levels = unique(c(coupon_list_train$large_area_name,coupon_list_test$large_area_name))

CAPSULE_TEXT.levels = unique(c(coupon_list_train$CAPSULE_TEXT,coupon_list_test$CAPSULE_TEXT))

GENRE_NAME.levels = unique(c(coupon_list_train$GENRE_NAME,coupon_list_test$GENRE_NAME))

### REPLACE JP W/ NUMBERS 
cat(">>> replacing JP w/ numbers ... \n")

## PREF_NAME
users$PREF_NAME = unlist(lapply(users$PREF_NAME,function(x) jp2num(value = x,levels=PREF_NAME.levels)))

coupon_list_train$PREF_NAME = unlist(lapply(coupon_list_train$ken_name,function(x) jp2num(value = x,levels=PREF_NAME.levels)))
coupon_list_test$PREF_NAME = unlist(lapply(coupon_list_test$ken_name,function(x) jp2num(value = x,levels=PREF_NAME.levels)))
coupon_list_train = coupon_list_train[, -grep(pattern = 'ken_name' , x = colnames(coupon_list_train))]
coupon_list_test = coupon_list_test[, -grep(pattern = 'ken_name' , x = colnames(coupon_list_test))]

coupon_area_train$PREF_NAME = unlist(lapply(coupon_area_train$PREF_NAME,function(x) jp2num(value = x,levels=PREF_NAME.levels)))
coupon_area_test$PREF_NAME = unlist(lapply(coupon_area_test$PREF_NAME,function(x) jp2num(value = x,levels=PREF_NAME.levels)))

## SMALL_AREA_NAME
coupon_list_train$SMALL_AREA_NAME = unlist(lapply(coupon_list_train$small_area_name,
                                                  function(x) jp2num(value = x,levels=SMALL_AREA_NAME.levels)))
coupon_list_test$SMALL_AREA_NAME = unlist(lapply(coupon_list_test$small_area_name,
                                                  function(x) jp2num(value = x,levels=SMALL_AREA_NAME.levels)))
coupon_list_train = coupon_list_train[, -grep(pattern = 'small_area_name' , x = colnames(coupon_list_train))]
coupon_list_test = coupon_list_test[, -grep(pattern = 'small_area_name' , x = colnames(coupon_list_test))]

coupon_detail_train$SMALL_AREA_NAME = unlist(lapply(coupon_detail_train$SMALL_AREA_NAME,
                                                  function(x) jp2num(value = x,levels=SMALL_AREA_NAME.levels)))

coupon_area_train$SMALL_AREA_NAME = unlist(lapply(coupon_area_train$SMALL_AREA_NAME,
                                                  function(x) jp2num(value = x,levels=SMALL_AREA_NAME.levels)))
coupon_area_test$SMALL_AREA_NAME = unlist(lapply(coupon_area_test$SMALL_AREA_NAME,
                                                  function(x) jp2num(value = x,levels=SMALL_AREA_NAME.levels)))

## LARGE_AREA_NAME
coupon_list_train$LARGE_AREA_NAME = unlist(lapply(coupon_list_train$large_area_name,
                                                 function(x) jp2num(value = x,levels=LARGE_AREA_NAME.levels)))
coupon_list_test$LARGE_AREA_NAME = unlist(lapply(coupon_list_test$large_area_name,
                                                  function(x) jp2num(value = x,levels=LARGE_AREA_NAME.levels)))
coupon_list_train = coupon_list_train[, -grep(pattern = 'large_area_name' , x = colnames(coupon_list_train))]
coupon_list_test = coupon_list_test[, -grep(pattern = 'large_area_name' , x = colnames(coupon_list_test))]

## CAPSULE_TEXT
coupon_list_train$CAPSULE_TEXT = unlist(lapply(coupon_list_train$CAPSULE_TEXT,
                                                  function(x) jp2num(value = x,levels=CAPSULE_TEXT.levels)))
coupon_list_test$CAPSULE_TEXT = unlist(lapply(coupon_list_test$CAPSULE_TEXT,
                                               function(x) jp2num(value = x,levels=CAPSULE_TEXT.levels)))

## GENRE_NAME
coupon_list_train$GENRE_NAME = unlist(lapply(coupon_list_train$GENRE_NAME,
                                               function(x) jp2num(value = x,levels=GENRE_NAME.levels)))
coupon_list_test$GENRE_NAME = unlist(lapply(coupon_list_test$GENRE_NAME,
                                             function(x) jp2num(value = x,levels=GENRE_NAME.levels)))


### ENCODING & AGGREGATING 
cat(">>> encoding & aggregating features ... \n")

## users 
colnames(users) = paste0('USER.',colnames(users))
users_enc = ff.makeFeatureSet(data.train = users[,-grep(pattern = 'USER_ID_hash' , x = colnames(users))] , 
                              data.test = users[,-grep(pattern = 'USER_ID_hash' , x = colnames(users))] , 
                              meta = c('D','N','N','C') , 
                              scaleNumericFeatures = F , parallelize = F)$traindata
users = cbind(USER_ID_hash = users$USER.USER_ID_hash , users_enc)
rm(users_enc)

## coupon_area_train / coupon_area_test 
# create a copy for meta-pourposes 
#coupon_area_train.meta = coupon_area_train
#coupon_area_test.meta = coupon_area_test

## add pref_name e small_area_name di LIST ad AREA 
coupon_area_train = rbind(coupon_area_train , coupon_list_train[,c('SMALL_AREA_NAME','PREF_NAME','COUPON_ID_hash')])
coupon_area_test = rbind(coupon_area_test , coupon_list_test[,c('SMALL_AREA_NAME','PREF_NAME','COUPON_ID_hash')])

# create a copy for meta-pourposes 
coupon_area_train_ext.meta = coupon_area_train
coupon_area_test_ext.meta = coupon_area_test

colnames(coupon_area_train) = paste0('AREA.',colnames(coupon_area_train))
colnames(coupon_area_test) = paste0('AREA.',colnames(coupon_area_test))
af = ff.makeFeatureSet(data.train = coupon_area_train[,-3], 
                       data.test = coupon_area_test[,-3], 
                       meta = c('C','C'),
                       scaleNumericFeatures = F , parallelize = F)

coupon_area_train = cbind(COUPON_ID_hash = coupon_area_train$AREA.COUPON_ID_hash , af$traindata)
coupon_area_test = cbind(COUPON_ID_hash = coupon_area_test$AREA.COUPON_ID_hash , af$testdata)
rm(af)

coupon_area_train = ddply(coupon_area_train , .(COUPON_ID_hash) , function(x) {
  x = x[,-grep(pattern = 'COUPON_ID_hash' , x = colnames(x))]
  unlist(lapply(x, function(y){
    ret = sum(y)
    if (ret > 1 ) ret = 1 
    return(ret)
  }))
})

coupon_area_test = ddply(coupon_area_test , .(COUPON_ID_hash) , function(x) {
  x = x[,-grep(pattern = 'COUPON_ID_hash' , x = colnames(x))]
  unlist(lapply(x, function(y){
    ret = sum(y)
    if (ret > 1 ) ret = 1 
    return(ret)
  }))
})

## coupon_list_train / coupon_list_test 
coupon_list_train$DISPFROM = as.Date(coupon_list_train$DISPFROM)
coupon_list_train$DISPEND = as.Date(coupon_list_train$DISPEND)
coupon_list_test$DISPFROM = as.Date(coupon_list_test$DISPFROM)
coupon_list_test$DISPEND = as.Date(coupon_list_test$DISPEND)

# create a copy for meta-pourposes 
coupon_list_train.meta = coupon_list_train 
coupon_list_test.meta = coupon_list_test

# add prefix to columns 
colnames(coupon_list_train) = paste0('LIST.',colnames(coupon_list_train))
colnames(coupon_list_test) = paste0('LIST.',colnames(coupon_list_test))

drop_cols = 'LIST.COUPON_ID_hash|LIST.DISPFROM|LIST.DISPEND|LIST.SMALL_AREA_NAME|LIST.PREF_NAME'
cp = ff.makeFeatureSet(data.train = coupon_list_train[,-grep(pattern = drop_cols , x = colnames(coupon_list_train))] , 
                                      data.test = coupon_list_test[,-grep(pattern = drop_cols , x = colnames(coupon_list_test))], 
                                      meta = c('C','C','N','N','N','N','C') , 
                                      scaleNumericFeatures = F , parallelize = F)
coupon_list_train = cbind(COUPON_ID_hash = coupon_list_train.meta$COUPON_ID_hash , cp$traindata)
coupon_list_test = cbind(COUPON_ID_hash = coupon_list_test.meta$COUPON_ID_hash , cp$testdata)
rm(cp)

## coupon_detail_train
coupon_detail_train$I_DATE = as.Date(coupon_detail_train$I_DATE)

# create a copy for meta-pourposes 
coupon_detail_train.meta = coupon_detail_train 

# add prefix to columns 
colnames(coupon_detail_train) = paste0('TRANS.',colnames(coupon_detail_train))
coupon_detail_train_enc = ff.makeFeatureSet(data.train = coupon_detail_train[,-grep(pattern = 'TRANS.PURCHASEID_hash|TRANS.USER_ID_hash|TRANS.COUPON_ID_hash' , 
                                                          x = colnames(coupon_detail_train))],
                  data.test = coupon_detail_train[,-grep(pattern = 'TRANS.PURCHASEID_hash|TRANS.USER_ID_hash|TRANS.COUPON_ID_hash' , 
                                                          x = colnames(coupon_detail_train))],
                  meta = c('N','D','C'),
                  scaleNumericFeatures = F , parallelize = F)$traindata
coupon_detail_train_enc = cbind(USER_ID_hash = coupon_detail_train$TRANS.USER_ID_hash,coupon_detail_train_enc)
coupon_detail_train_enc = cbind(COUPON_ID_hash = coupon_detail_train$TRANS.COUPON_ID_hash,coupon_detail_train_enc)
coupon_detail_train_enc = cbind(PURCHASEID_hash = coupon_detail_train$TRANS.PURCHASEID_hash,coupon_detail_train_enc)
coupon_detail_train = coupon_detail_train_enc
rm(coupon_detail_train_enc)
                  
### MERGING FINAL 3 DATA CHUNKS 
cat(">>> merging data in main data chunks ... \n")

## user_data
user_data = users 
rm(users)

## coupon_data 
coupon_data_train = merge(x = coupon_list_train , y = coupon_area_train , by = 'COUPON_ID_hash' , all.x = T , all.y = F)
coupon_data_train[is.na(coupon_data_train)] = 0 

coupon_data_test = merge(x = coupon_list_test , y = coupon_area_test , by = 'COUPON_ID_hash' , all.x = T , all.y = F)
#coupon_data_test[is.na(coupon_data_train)] = 0 

rm(list = c('coupon_list_train','coupon_area_train','coupon_list_test','coupon_area_test'))

## trans_data 
trans_data = coupon_detail_train
rm(coupon_detail_train)

#### DEBUG 
if (debug) {
  cat(">>> debug mode activated ... checking congruence between LIST and AREA ... \n")
  
  ## train SMALL_AREA_NAME 
  coupon_area_train = ddply(coupon_area_train_ext.meta , .(COUPON_ID_hash) , function(x) {
    x = x[,-grep(pattern = 'COUPON_ID_hash|PREF_NAME' , x = colnames(x))]
    t(x)
  })
  
  colnames(coupon_area_train)[-1] = paste0("SA",colnames(coupon_area_train)[-1])
  m = merge(x = coupon_list_train.meta , y = coupon_area_train , by='COUPON_ID_hash' , all = T )
  
  yes = rep(F,nrow(m))
  for (i in 1:nrow(m)) {
    yes[i] = m[i,]$SMALL_AREA_NAME %in% m[i,paste0('SA',1:55)]
  }
  stopifnot(sum(!yes)==0) ## 0
  
  ##### train PREF_NAME 
  coupon_area_train = ddply(coupon_area_train_ext.meta, .(COUPON_ID_hash) , function(x) {
    x = x[,-grep(pattern = 'COUPON_ID_hash|SMALL_AREA_NAME' , x = colnames(x))]
    t(x)
  })
  
  colnames(coupon_area_train)[-1] = paste0("PA",colnames(coupon_area_train)[-1])
  m = merge(x = coupon_list_train.meta , y = coupon_area_train , by='COUPON_ID_hash' , all = T )
  
  yes = rep(F,nrow(m))
  for (i in 1:nrow(m)) {
    yes[i] = m[i,]$PREF_NAME %in% m[i,paste0('PA',1:55)]
  }
  stopifnot(sum(!yes)==0) ## 0
  
  ##### test SMALL_AREA_NAME 
  coupon_area_test = ddply(coupon_area_test_ext.meta , .(COUPON_ID_hash) , function(x) {
    x = x[,-grep(pattern = 'COUPON_ID_hash|PREF_NAME' , x = colnames(x))]
    t(x)
  })
  
  colnames(coupon_area_test)[-1] = paste0("SA",colnames(coupon_area_test)[-1])
  m = merge(x = coupon_list_test.meta , y = coupon_area_test , by='COUPON_ID_hash' , all = T )
  
  yes = rep(F,nrow(m))
  for (i in 1:nrow(m)) {
    yes[i] = m[i,]$SMALL_AREA_NAME %in% m[i,paste0('SA',1:30)]
  }
  stopifnot(sum(!yes)==0) ## 0
  
  ##### test PREF_NAME 
  coupon_area_test = ddply(coupon_area_test_ext.meta , .(COUPON_ID_hash) , function(x) {
    x = x[,-grep(pattern = 'COUPON_ID_hash|SMALL_AREA_NAME' , x = colnames(x))]
    t(x)
  })
  
  colnames(coupon_area_test)[-1] = paste0("PA",colnames(coupon_area_test)[-1])
  m = merge(x = coupon_list_test.meta , y = coupon_area_test , by='COUPON_ID_hash' , all = T )
  
  yes = rep(F,nrow(m))
  for (i in 1:nrow(m)) {
    yes[i] = m[i,]$PREF_NAME %in% m[i,paste0('PA',1:30)]
  }
  stopifnot(sum(!yes)==0) ## 0
  cat(">>> debug: OK ...\n")
}

