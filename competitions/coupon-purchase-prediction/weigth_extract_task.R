library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 

vect_prod = function (train_vect, label_vect, idx) {
  prob_mat = as.data.frame(as.matrix(train_vect[,idx]) %*% as.matrix(t(label_vect[,idx])))
  colnames(prob_mat) = label_vect$COUPON_ID_hash
  stopifnot(max(prob_mat)<=1+10^-9, min(prob_mat)>=0) 
  stopifnot(sum(is.na(prob_mat))==0) 
  return(prob_mat)
}


weight_wn = function(week_number , 
                            smooth_coeff,
                            verbose=T) {
  remove_PREF_NAME = T 
  
  ## train_<WN>_<SMOOTH>.csv
  train_vect = as.data.frame( fread( paste(ff.getPath("elab_train"),"train_",week_number,"_",smooth_coeff,".csv",sep='') , stringsAsFactors = F))
  
  ## labels vect 
  label_vect = getLabelVector(week_number=week_number , removeCOUPON_ID_hash =F , verbose=T)
  labels = getLabels(week_number = week_number,verbose = verbose)
  
  ## order columns train_vect and label_vect
  train_vect = train_vect[, c(colnames(train_vect)[1] , sort(colnames(train_vect)[-1]))] 
  label_vect = label_vect[, c(colnames(label_vect)[1] , sort(colnames(label_vect)[-1]))] 
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  if (remove_PREF_NAME) {
    if (verbose) cat(">>> removing AREA.PREF_NAME_x ...\n")
    #AREA.PREF_NAME_x  
    train_vect = train_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(label_vect))]
  }
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  ## compute components 
  idx.LIST.CAPSULE_TEXT = grep(pattern = 'LIST.CAPSULE_TEXT_',x = colnames(train_vect))
  idx.LIST.GENRE_NAME = grep(pattern = 'LIST.GENRE_NAME_',x = colnames(train_vect))
  idx.LIST.LARGE_AREA_NAME = grep(pattern = 'LIST.LARGE_AREA_NAME_',x = colnames(train_vect))
  idx.AREA.SMALL_AREA_NAME = grep(pattern = 'AREA.SMALL_AREA_NAME_',x = colnames(train_vect))
  #idx.AREA.PREF_NAME = grep(pattern = 'AREA.PREF_NAME_',x = colnames(train_vect))
  idx.LIST.BID_CATALOG_PRICE = grep(pattern = 'LIST.BID_CATALOG_PRICE_',x = colnames(train_vect))
  idx.LIST.BID_PRICE_RATE = grep(pattern = 'LIST.BID_PRICE_RATE_',x = colnames(train_vect))
  idx.LIST.WDISPPERIOD = grep(pattern = 'LIST.WDISPPERIOD_',x = colnames(train_vect))
  
  ## 
  prob_mat.LIST.CAPSULE_TEXT = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.CAPSULE_TEXT)[2] = 'LIST.CAPSULE_TEXT'
  
  prob_mat.LIST.GENRE_NAME = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.GENRE_NAME)[2] = 'LIST.GENRE_NAME'
  
  prob_mat.LIST.LARGE_AREA_NAME = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.LARGE_AREA_NAME)[2] = 'LIST.LARGE_AREA_NAME'
  
  prob_mat.AREA.SMALL_AREA_NAME = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.AREA.SMALL_AREA_NAME)[2] = 'AREA.SMALL_AREA_NAME'
  
  prob_mat.LIST.BID_CATALOG_PRICE = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.BID_CATALOG_PRICE)[2] = 'LIST.BID_CATALOG_PRICE'
  
  prob_mat.LIST.BID_PRICE_RATE = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.BID_PRICE_RATE)[2] = 'LIST.BID_PRICE_RATE'
  
  prob_mat.LIST.WDISPPERIOD = data.frame(USER_ID_hash=train_vect$USER_ID_hash , rep(0,nrow(train_vect)))
  colnames(prob_mat.LIST.WDISPPERIOD)[2] = 'LIST.WDISPPERIOD'
  
  lapply(seq_along(train_vect$USER_ID_hash), function(i){
    uid = train_vect[i,]$USER_ID_hash
    t_v = train_vect[train_vect$USER_ID_hash==uid,]
    lb = labels$labels
    l_v = label_vect[label_vect$COUPON_ID_hash %in% unlist(strsplit(x = lb[lb$USER_ID_hash == uid , 'PURCHASED_COUPONS'],split = ' ')),]
    
    if (nrow(l_v)>0) {
      prob_mat.LIST.CAPSULE_TEXT[prob_mat.LIST.CAPSULE_TEXT$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.CAPSULE_TEXT)
      prob_mat.LIST.GENRE_NAME[prob_mat.LIST.GENRE_NAME$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.GENRE_NAME)
      prob_mat.LIST.LARGE_AREA_NAME[prob_mat.LIST.LARGE_AREA_NAME$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.LARGE_AREA_NAME)
      prob_mat.AREA.SMALL_AREA_NAME[prob_mat.AREA.SMALL_AREA_NAME$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.AREA.SMALL_AREA_NAME)
      prob_mat.LIST.BID_CATALOG_PRICE[prob_mat.LIST.BID_CATALOG_PRICE$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.BID_CATALOG_PRICE)
      prob_mat.LIST.BID_PRICE_RATE[prob_mat.LIST.BID_PRICE_RATE$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.BID_PRICE_RATE)
      prob_mat.LIST.WDISPPERIOD[prob_mat.LIST.WDISPPERIOD$USER_ID_hash==uid,2] <<- vect_prod(train_vect = t_v , label_vect = l_v , idx = idx.LIST.WDISPPERIOD)
    }
  })
  
  ret = merge(prob_mat.LIST.CAPSULE_TEXT,prob_mat.LIST.GENRE_NAME)
  ret = merge(ret,prob_mat.LIST.LARGE_AREA_NAME)
  ret = merge(ret,prob_mat.AREA.SMALL_AREA_NAME)
  ret = merge(ret,prob_mat.LIST.BID_CATALOG_PRICE)
  ret = merge(ret,prob_mat.LIST.BID_PRICE_RATE)
  ret = merge(ret,prob_mat.LIST.WDISPPERIOD)

  
  return(ret)
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')

### GLOBAL CONFIG 
debug = F

### DATA 
source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)
ff.bindPath(type = 'elab_pred' , sub_path = 'dataset/coupon-purchase-prediction/elab/pred' , createDir = T)
ff.bindPath(type = 'elab_meta' , sub_path = 'dataset/coupon-purchase-prediction/elab/meta' , createDir = T)


### PROCESSING 
wn.first = 10 
wn.last = 30 

cat(">>> weight extract:....\n")
for (wn in wn.first:wn.last) {
  for (sm in seq(from = -4,to = 4,by = 1)) {
    cat(">>> weigthing week number",wn," // smooth_coeff",sm,"...\n")
    wg = weight_wn(week_number=wn , smooth_coeff =sm,verbose=T)
    fn = paste(ff.getPath("elab_meta"),"w_",wn,"_",sm,".csv",sep='')
    write.csv(wg,
              quote=FALSE, 
              file=fn ,
              row.names=FALSE)
  }
}




