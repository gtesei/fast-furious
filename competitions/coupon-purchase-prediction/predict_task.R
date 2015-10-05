library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 
adjust_prediction = function(week_number,smooth_coeff,pred) {
  
  ## adjust 
  ord_coup = unlist(strsplit(x = pred[pred$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',]$PURCHASED_COUPONS , split = ' '))
  
  ord_coup_mat = ddply(pred, .(USER_ID_hash) , function(x) {
    unlist(strsplit(x = x$PURCHASED_COUPONS , split = ' '))
  })
  colnames(ord_coup_mat)[2:11] = paste0("coup_",1:10)
  stopifnot(sum(ord_coup_mat[ord_coup_mat$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',2:11] == ord_coup)==10)
  
  ##
  for (uid in coupon_visit_test$USER_ID_hash) {
    coup_old = as.vector(ord_coup_mat[ord_coup_mat$USER_ID_hash==uid,2:11])
    coup_new = coupon_visit_test[coupon_visit_test$USER_ID_hash==uid,]$COUPON_ID_hash
    coup_new = unique(coup_new)
    coup_diff = setdiff(coup_old,coup_new)
    coup_new = c(coup_new,coup_diff[1:(10-length(coup_new))])
    stopifnot(length(coup_new)==10)
    stopifnot(length(coup_new)==length(unique(coup_new)))
    ord_coup_mat[ord_coup_mat$USER_ID_hash==uid,2:11] = coup_new
  }
  
  ##
  ord_coup_sub = ddply( ord_coup_mat , .(USER_ID_hash) , function(x) {
    PURCHASED_COUPONS = paste(x[-1] , collapse = ' ')
    return(setNames(object = PURCHASED_COUPONS,nm = 'PURCHASED_COUPONS'))
  })
  
  stopifnot(sum(unlist(strsplit(x = ord_coup_sub[ord_coup_sub$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',2], split = ' ')) 
                == ord_coup_mat[ord_coup_mat$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',2:11])==10)
  
  return(ord_coup_sub)
}
predict_wn_smooth = function(week_number , smooth_coeff, 
                             remove_WDISPPERIOD = F , 
                             remove_PREF_NAME = T , 
                             remove_LARGE_AREA_NAME = F , 
                             verbose=T,debug=F) {
  
  ## train_<WN>_<SMOOTH>.csv
  train_vect = as.data.frame( fread( paste(ff.getPath("elab_train"),"train_",week_number,"_",smooth_coeff,".csv",sep='') , stringsAsFactors = F))
  
  ## labels vect 
  label_vect = getLabelVector(week_number=week_number , removeCOUPON_ID_hash =F , verbose=T)
  
  ## order columns train_vect and label_vect
  train_vect = train_vect[, c(colnames(train_vect)[1] , sort(colnames(train_vect)[-1]))] 
  label_vect = label_vect[, c(colnames(label_vect)[1] , sort(colnames(label_vect)[-1]))] 
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  if (remove_WDISPPERIOD) {
    if (verbose) cat(">>> removing LIST.WDISPPERIOD_x ...\n")
    #LIST.WDISPPERIOD_x  
    train_vect = train_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(label_vect))]
  }
  
  if (remove_PREF_NAME) {
    if (verbose) cat(">>> removing AREA.PREF_NAME_x ...\n")
    #AREA.PREF_NAME_x  
    train_vect = train_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(label_vect))]
  }
  
  if (remove_LARGE_AREA_NAME) {
    if (verbose) cat(">>> removing LIST.LARGE_AREA_NAME ...\n")
    #LIST.LARGE_AREA_NAME_x  
    train_vect = train_vect[,-grep(pattern = "LIST.LARGE_AREA_NAME_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "LIST.LARGE_AREA_NAME_" , x = colnames(label_vect))]
  }
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  
  ## computing probs matrix 
  prob_mat = as.data.frame(as.matrix(train_vect[,-1]) %*% as.matrix(t(label_vect[,-1])))
  colnames(prob_mat) = label_vect$COUPON_ID_hash
  prob_mat = cbind(USER_ID_hash=train_vect$USER_ID_hash,prob_mat,stringsAsFactors=F)
  
  ## computing max probs coupons for each user 
  stopifnot(length(prob_mat$USER_ID_hash)==length(unique(prob_mat$USER_ID_hash)))
  pred = ddply(prob_mat, .(USER_ID_hash) , function(x) {
    user_id = x[1]
    len_coupons = length(x)-1
    prob_user_i_t = t(x[,-1])
    prob_user_i_t = prob_user_i_t[order(prob_user_i_t,decreasing = T),]
    prob_user_i_t_10 = prob_user_i_t[1:10]
    pred_user_i = names(prob_user_i_t_10)
    return(pred_user_i)
  })
  colnames(pred)[2:11] = paste0("pur_coup_",1:10)
  
  if(debug) {
    idxs = sample(prob_mat$USER_ID_hash[1:10000],12) ## taglio ai primi 10.000 per non icappare in utenti che non hanno fatto acquisti 
    for (idx in idxs) {
      prob_user_i = prob_mat[prob_mat$USER_ID_hash==idx,-1]
      prob_user_i_t = t(prob_user_i)
      colnames(prob_user_i_t) = 'prob'
      prob_user_i_t = prob_user_i_t[order(prob_user_i_t,decreasing = T),]
      prob_user_i_t_10 = prob_user_i_t[1:10]
      pred_user_i = names(prob_user_i_t_10)
      stopifnot(sum(pred[pred$USER_ID_hash==idx,2:11] != pred_user_i)==0)
    }
    cat(">>> debug [12/12]: OK\n")
  }
  
  stopifnot(sum(is.na(pred))==0)
  
  ## sub format
  pred_sub =  ddply( pred , .(USER_ID_hash) , function(x) {
    PURCHASED_COUPONS = paste(x[-1] , collapse = ' ')
    return(setNames(object = PURCHASED_COUPONS,nm = 'PURCHASED_COUPONS'))
  })
  
  stopifnot(sum(is.na(pred_sub))==0)
  
  return(pred_sub)
}

vect_prod = function (train_vect, label_vect, idx) {
  prob_mat = as.data.frame(as.matrix(train_vect[,idx]) %*% as.matrix(t(label_vect[,idx])))
  colnames(prob_mat) = label_vect$COUPON_ID_hash
  stopifnot(max(prob_mat)<=1+10^-9, min(prob_mat)>=0) 
  stopifnot(sum(is.na(prob_mat))==0) 
  return(prob_mat)
}

predict_equal = function(week_number , 
                         smooth_coeff,
                         verbose=T) {
  remove_WDISPPERIOD = T 
  remove_PREF_NAME = T 
  
  ## train_<WN>_<SMOOTH>.csv
  train_vect = as.data.frame( fread( paste(ff.getPath("elab_train"),"train_",week_number,"_",smooth_coeff,".csv",sep='') , stringsAsFactors = F))
  
  ## labels vect 
  label_vect = getLabelVector(week_number=week_number , removeCOUPON_ID_hash =F , verbose=T)
  
  if (remove_WDISPPERIOD) {
    if (verbose) cat(">>> removing LIST.WDISPPERIOD_x ...\n")
    #LIST.WDISPPERIOD_x  
    train_vect = train_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(label_vect))]
  }
  
  if (remove_PREF_NAME) {
    if (verbose) cat(">>> removing AREA.PREF_NAME_x ...\n")
    #AREA.PREF_NAME_x  
    train_vect = train_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(label_vect))]
  }
  
  ## order columns train_vect and label_vect
  train_vect = train_vect[, c(colnames(train_vect)[1] , sort(colnames(train_vect)[-1]))] 
  label_vect = label_vect[, c(colnames(label_vect)[1] , sort(colnames(label_vect)[-1]))] 
  
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
  
  ## LIST.CAPSULE_TEXT
  prob_mat.LIST.CAPSULE_TEXT = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.CAPSULE_TEXT)
  
  ## LIST.GENRE_NAME
  prob_mat.LIST.GENRE_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.GENRE_NAME)
  
  ## LIST.LARGE_AREA_NAME
  prob_mat.LIST.LARGE_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.LARGE_AREA_NAME)
  
  ## AREA.SMALL_AREA_NAME
  prob_mat.AREA.SMALL_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.AREA.SMALL_AREA_NAME)
  
  ## LIST.BID_CATALOG_PRICE
  prob_mat.LIST.BID_CATALOG_PRICE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_CATALOG_PRICE)
  
  ## LIST.BID_PRICE_RATE
  prob_mat.LIST.BID_PRICE_RATE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_PRICE_RATE)
  
  ## the Hypothesis
  lin_term = prob_mat.LIST.GENRE_NAME 
  lin_term = lin_term + prob_mat.LIST.CAPSULE_TEXT 
  #lin_term = lin_term + prob_mat.LIST.BID_CATALOG_PRICE 
  #lin_term = lin_term + prob_mat.LIST.BID_PRICE_RATE 
  lin_term = lin_term + prob_mat.LIST.LARGE_AREA_NAME + prob_mat.AREA.SMALL_AREA_NAME
  prob_mat = lin_term
  
  ## add wsperiod norm 
  c_dp = NULL
  if (week_number==0) {
    c_dp = coupon_list_test.meta[,c('COUPON_ID_hash','WDISPPERIOD_NORM')]
  } else {
    c_dp = coupon_list_train.meta[coupon_list_train.meta$COUPON_ID_hash %in% colnames(prob_mat) ,c('COUPON_ID_hash','WDISPPERIOD_NORM')]
  } 
  for (cn in colnames(prob_mat)) {
    # WDISPPERIOD_NORM
    prob_mat[,cn] = prob_mat[,cn] + c_dp[c_dp$COUPON_ID_hash==cn,]$WDISPPERIOD_NORM
    
    # DISCOUNT_PRICE_LOG_NORM
    prob_mat[,cn] = prob_mat[,cn] + c_dp[c_dp$COUPON_ID_hash==cn,]$DISCOUNT_PRICE_LOG_NORM
    
    # PRICE_RATE_NORM
    prob_mat[,cn] = prob_mat[,cn] + c_dp[c_dp$COUPON_ID_hash==cn,]$PRICE_RATE_NORM
  }
  
  ## attach USER_ID_hash 
  prob_mat = cbind(USER_ID_hash=train_vect$USER_ID_hash,prob_mat,stringsAsFactors=F)
  
  ## computing max probs coupons for each user 
  stopifnot(length(prob_mat$USER_ID_hash)==length(unique(prob_mat$USER_ID_hash)))
  pred = ddply(prob_mat, .(USER_ID_hash) , function(x) {
    user_id = x[1]
    len_coupons = length(x)-1
    prob_user_i_t = t(x[,-1])
    prob_user_i_t = prob_user_i_t[order(prob_user_i_t,decreasing = T),]
    prob_user_i_t_10 = prob_user_i_t[1:10]
    pred_user_i = names(prob_user_i_t_10)
    return(pred_user_i)
  })
  colnames(pred)[2:11] = paste0("pur_coup_",1:10)
  
  stopifnot(sum(is.na(pred))==0)
  
  ## sub format
  pred_sub =  ddply( pred , .(USER_ID_hash) , function(x) {
    PURCHASED_COUPONS = paste(x[-1] , collapse = ' ')
    return(setNames(object = PURCHASED_COUPONS,nm = 'PURCHASED_COUPONS'))
  })
  
  stopifnot(sum(is.na(pred_sub))==0)
  
  return(pred_sub)
}

predict_weight = function(week_number , 
                          smooth_coeff,
                          verbose=T) {
  remove_WDISPPERIOD = T 
  remove_PREF_NAME = T 
  
  ## weigths
  #weight_fn = paste("w_",(week_number+1),"_",smooth_coeff,".csv",sep="")
  weight_fn = paste("avg_",smooth_coeff,".csv",sep="")
  weigths = as.data.frame( fread( paste(ff.getPath("elab_meta"),weight_fn,sep='') , stringsAsFactors = F))
  #weigths[weigths==0] <- 1
  aa = unlist(lapply(1:nrow(weigths) , function(i) {
    #     if (sum(weigths[i,2:ncol(weigths)])==0) {
    #       weigths[i,2:ncol(weigths)] <<-1
    #     }
    #     weigths[i,2:ncol(weigths)] <<- 1/weigths[i,2:ncol(weigths)]
    weigths[i,2:ncol(weigths)] <<- 1 - weigths[i,2:ncol(weigths)]
  }))
  stopifnot( sum(weigths[,2:ncol(weigths)]<0) == 0 )
  colnames(weigths)[2:ncol(weigths)] = paste0("W.",colnames(weigths)[2:ncol(weigths)] )
  
  ## train_<WN>_<SMOOTH>.csv
  train_vect = as.data.frame( fread( paste(ff.getPath("elab_train"),"train_",week_number,"_",smooth_coeff,".csv",sep='') , stringsAsFactors = F))
  
  ## labels vect 
  label_vect = getLabelVector(week_number=week_number , removeCOUPON_ID_hash =F , verbose=T)
  
  if (remove_WDISPPERIOD) {
    if (verbose) cat(">>> removing LIST.WDISPPERIOD_x ...\n")
    #LIST.WDISPPERIOD_x  
    train_vect = train_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(label_vect))]
  }
  
  if (remove_PREF_NAME) {
    if (verbose) cat(">>> removing AREA.PREF_NAME_x ...\n")
    #AREA.PREF_NAME_x  
    train_vect = train_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "AREA.PREF_NAME_" , x = colnames(label_vect))]
  }
  
  ## order columns train_vect and label_vect
  train_vect = train_vect[, c(colnames(train_vect)[1] , sort(colnames(train_vect)[-1]))] 
  label_vect = label_vect[, c(colnames(label_vect)[1] , sort(colnames(label_vect)[-1]))] 
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  ## weighting: same uid  
  train_vect = merge(x = train_vect , y = weigths , by='USER_ID_hash')
  
  ## compute components 
  idx.LIST.CAPSULE_TEXT = grep(pattern = 'LIST.CAPSULE_TEXT_',x = colnames(train_vect))
  idx.LIST.GENRE_NAME = grep(pattern = 'LIST.GENRE_NAME_',x = colnames(train_vect))
  idx.LIST.LARGE_AREA_NAME = grep(pattern = 'LIST.LARGE_AREA_NAME_',x = colnames(train_vect))
  idx.AREA.SMALL_AREA_NAME = grep(pattern = 'AREA.SMALL_AREA_NAME_',x = colnames(train_vect))
  #idx.AREA.PREF_NAME = grep(pattern = 'AREA.PREF_NAME_',x = colnames(train_vect))
  idx.LIST.BID_CATALOG_PRICE = grep(pattern = 'LIST.BID_CATALOG_PRICE_',x = colnames(train_vect))
  idx.LIST.BID_PRICE_RATE = grep(pattern = 'LIST.BID_PRICE_RATE_',x = colnames(train_vect))
  
  ## LIST.CAPSULE_TEXT
  prob_mat.LIST.CAPSULE_TEXT = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.CAPSULE_TEXT)
  
  ## LIST.GENRE_NAME
  prob_mat.LIST.GENRE_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.GENRE_NAME)
  
  ## LIST.LARGE_AREA_NAME
  prob_mat.LIST.LARGE_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.LARGE_AREA_NAME)
  
  ## AREA.SMALL_AREA_NAME
  prob_mat.AREA.SMALL_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.AREA.SMALL_AREA_NAME)
  
  ## LIST.BID_CATALOG_PRICE
  prob_mat.LIST.BID_CATALOG_PRICE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_CATALOG_PRICE)
  
  ## LIST.BID_PRICE_RATE
  prob_mat.LIST.BID_PRICE_RATE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_PRICE_RATE)
  
  ## weighting
  prob_mat.LIST.CAPSULE_TEXT = prob_mat.LIST.CAPSULE_TEXT * 
    as.data.frame(matrix(rep(train_vect$W.LIST.CAPSULE_TEXT,ncol(prob_mat.LIST.CAPSULE_TEXT)),ncol=ncol(prob_mat.LIST.CAPSULE_TEXT)))
  
  prob_mat.LIST.GENRE_NAME = prob_mat.LIST.GENRE_NAME * 
    as.data.frame(matrix(rep(train_vect$W.LIST.GENRE_NAME,ncol(prob_mat.LIST.GENRE_NAME)),ncol=ncol(prob_mat.LIST.GENRE_NAME)))
  
  prob_mat.LIST.LARGE_AREA_NAME = prob_mat.LIST.LARGE_AREA_NAME * 
    as.data.frame(matrix(rep(train_vect$W.LIST.LARGE_AREA_NAME,ncol(prob_mat.LIST.LARGE_AREA_NAME)),ncol=ncol(prob_mat.LIST.LARGE_AREA_NAME)))
  
  prob_mat.AREA.SMALL_AREA_NAME = prob_mat.AREA.SMALL_AREA_NAME * 
    as.data.frame(matrix(rep(train_vect$W.AREA.SMALL_AREA_NAME,ncol(prob_mat.AREA.SMALL_AREA_NAME)),ncol=ncol(prob_mat.AREA.SMALL_AREA_NAME)))
  
  prob_mat.LIST.BID_CATALOG_PRICE = prob_mat.LIST.BID_CATALOG_PRICE * 
    as.data.frame(matrix(rep(train_vect$W.LIST.BID_CATALOG_PRICE,ncol(prob_mat.LIST.BID_CATALOG_PRICE)),ncol=ncol(prob_mat.LIST.BID_CATALOG_PRICE)))
  
  prob_mat.LIST.BID_PRICE_RATE = prob_mat.LIST.BID_PRICE_RATE * 
    as.data.frame(matrix(rep(train_vect$W.LIST.BID_PRICE_RATE,ncol(prob_mat.LIST.BID_PRICE_RATE)),ncol=ncol(prob_mat.LIST.BID_PRICE_RATE)))
  
  
  ## the Hypothesis
  lin_term = prob_mat.LIST.CAPSULE_TEXT + prob_mat.LIST.GENRE_NAME  
  lin_term = lin_term + prob_mat.LIST.BID_CATALOG_PRICE 
  lin_term = lin_term + prob_mat.LIST.BID_PRICE_RATE 
  lin_term = lin_term + prob_mat.LIST.LARGE_AREA_NAME + prob_mat.AREA.SMALL_AREA_NAME
  prob_mat = lin_term
  
  ## add wsperiod norm 
  c_dp = NULL
  if (week_number==0) {
    c_dp = coupon_list_test.meta[,c('COUPON_ID_hash','WDISPPERIOD_NORM')]
  } else {
    c_dp = coupon_list_train.meta[coupon_list_train.meta$COUPON_ID_hash %in% colnames(prob_mat) ,c('COUPON_ID_hash','WDISPPERIOD_NORM')]
  } 
  for (cn in colnames(prob_mat)) {
    prob_mat[,cn] = prob_mat[,cn] + c_dp[c_dp$COUPON_ID_hash==cn,]$WDISPPERIOD_NORM
  }
  
  ## attach USER_ID_hash 
  prob_mat = cbind(USER_ID_hash=train_vect$USER_ID_hash,prob_mat,stringsAsFactors=F)
  
  ## computing max probs coupons for each user 
  stopifnot(length(prob_mat$USER_ID_hash)==length(unique(prob_mat$USER_ID_hash)))
  pred = ddply(prob_mat, .(USER_ID_hash) , function(x) {
    user_id = x[1]
    len_coupons = length(x)-1
    prob_user_i_t = t(x[,-1])
    prob_user_i_t = prob_user_i_t[order(prob_user_i_t,decreasing = T),]
    prob_user_i_t_10 = prob_user_i_t[1:10]
    pred_user_i = names(prob_user_i_t_10)
    return(pred_user_i)
  })
  colnames(pred)[2:11] = paste0("pur_coup_",1:10)
  
  stopifnot(sum(is.na(pred))==0)
  
  ## sub format
  pred_sub =  ddply( pred , .(USER_ID_hash) , function(x) {
    PURCHASED_COUPONS = paste(x[-1] , collapse = ' ')
    return(setNames(object = PURCHASED_COUPONS,nm = 'PURCHASED_COUPONS'))
  })
  
  stopifnot(sum(is.na(pred_sub))==0)
  
  return(pred_sub)
}
predict_wn_model = function(week_number , 
                            smooth_coeff, 
                            linear_coeff=c(0.2,0.6,1,0.6) ,
                            verbose=T) {
  remove_WDISPPERIOD = T 
  remove_PREF_NAME = T 
  
  ## train_<WN>_<SMOOTH>.csv
  train_vect = as.data.frame( fread( paste(ff.getPath("elab_train"),"train_",week_number,"_",smooth_coeff,".csv",sep='') , stringsAsFactors = F))
  
  ## labels vect 
  label_vect = getLabelVector(week_number=week_number , removeCOUPON_ID_hash =F , verbose=T)
  
  ## order columns train_vect and label_vect
  train_vect = train_vect[, c(colnames(train_vect)[1] , sort(colnames(train_vect)[-1]))] 
  label_vect = label_vect[, c(colnames(label_vect)[1] , sort(colnames(label_vect)[-1]))] 
  
  ## check 
  stopifnot(sum(colnames(train_vect)[-1] != colnames(label_vect)[-1])==0)  
  
  if (remove_WDISPPERIOD) {
    if (verbose) cat(">>> removing LIST.WDISPPERIOD_x ...\n")
    #LIST.WDISPPERIOD_x  
    train_vect = train_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(train_vect))]
    label_vect = label_vect[,-grep(pattern = "LIST.WDISPPERIOD_" , x = colnames(label_vect))]
  }
  
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
  
  ## LIST.CAPSULE_TEXT
  prob_mat.LIST.CAPSULE_TEXT = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.CAPSULE_TEXT)
  
  ## LIST.GENRE_NAME
  prob_mat.LIST.GENRE_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.GENRE_NAME)
  
  ## LIST.LARGE_AREA_NAME
  prob_mat.LIST.LARGE_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.LARGE_AREA_NAME)
  
  ## AREA.SMALL_AREA_NAME
  prob_mat.AREA.SMALL_AREA_NAME = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.AREA.SMALL_AREA_NAME)
  
  ## LIST.BID_CATALOG_PRICE
  prob_mat.LIST.BID_CATALOG_PRICE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_CATALOG_PRICE)
  
  ## LIST.BID_PRICE_RATE
  prob_mat.LIST.BID_PRICE_RATE = vect_prod(train_vect = train_vect , label_vect = label_vect , idx = idx.LIST.BID_PRICE_RATE)
  
  ## the Hypothesis
  #lin_term = linear_coeff[2] * prob_mat.LIST.CAPSULE_TEXT + (1-linear_coeff[2]) * prob_mat.LIST.GENRE_NAME  
  lin_term = linear_coeff[2] * prob_mat.LIST.CAPSULE_TEXT + linear_coeff[2] * prob_mat.LIST.GENRE_NAME  
  lin_term = lin_term + linear_coeff[3] * prob_mat.LIST.BID_CATALOG_PRICE 
  lin_term = lin_term + linear_coeff[4] * prob_mat.LIST.BID_PRICE_RATE 
  prob_mat = linear_coeff[1] * prob_mat.LIST.LARGE_AREA_NAME + (1-linear_coeff[1]) * prob_mat.AREA.SMALL_AREA_NAME
  prob_mat = prob_mat * lin_term
  
  ## attach USER_ID_hash 
  prob_mat = cbind(USER_ID_hash=train_vect$USER_ID_hash,prob_mat,stringsAsFactors=F)
  
  ## computing max probs coupons for each user 
  stopifnot(length(prob_mat$USER_ID_hash)==length(unique(prob_mat$USER_ID_hash)))
  pred = ddply(prob_mat, .(USER_ID_hash) , function(x) {
    user_id = x[1]
    len_coupons = length(x)-1
    prob_user_i_t = t(x[,-1])
    prob_user_i_t = prob_user_i_t[order(prob_user_i_t,decreasing = T),]
    prob_user_i_t_10 = prob_user_i_t[1:10]
    pred_user_i = names(prob_user_i_t_10)
    return(pred_user_i)
  })
  colnames(pred)[2:11] = paste0("pur_coup_",1:10)
  
  stopifnot(sum(is.na(pred))==0)
  
  ## sub format
  pred_sub =  ddply( pred , .(USER_ID_hash) , function(x) {
    PURCHASED_COUPONS = paste(x[-1] , collapse = ' ')
    return(setNames(object = PURCHASED_COUPONS,nm = 'PURCHASED_COUPONS'))
  })
  
  stopifnot(sum(is.na(pred_sub))==0)
  
  return(pred_sub)
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')
ff.bindPath(type = 'data' , sub_path = 'dataset/coupon-purchase-prediction/data')

### GLOBAL CONFIG 
debug = F

### DATA & COUPON VECTORS 
coupon_visit = as.data.frame( fread(paste(ff.getPath("data") , "coupon_visit_train.csv" , sep='')))
source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)
ff.bindPath(type = 'elab_pred' , sub_path = 'dataset/coupon-purchase-prediction/elab/pred' , createDir = T)
ff.bindPath(type = 'elab_meta' , sub_path = 'dataset/coupon-purchase-prediction/elab/meta' , createDir = T)

### BOOSTING VIEWED COUPONS 
coupon_visit$I_DATE = as.Date(coupon_visit$I_DATE)
colnames(coupon_visit)[5] = 'COUPON_ID_hash'
coupon_visit_test = merge(x=coupon_visit,y=coupon_list_test.meta[,c('COUPON_ID_hash','DISPFROM')] , by='COUPON_ID_hash' , all=F)
coupon_visit_test = coupon_visit_test[,c('USER_ID_hash','COUPON_ID_hash')]
coupon_visit_test_uniq = ddply(coupon_visit_test,.(USER_ID_hash,COUPON_ID_hash))
stopifnot(nrow(coupon_visit_test_uniq)==nrow(coupon_visit_test))
coupon_visit_test[coupon_visit_test$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',]

### PROCESSING 
#linear_coeff = linear_coeff=c(0.2,0.6,1,0.6)
#linear_coeff = linear_coeff=c(0.2,1,1,1)
for (wn in 0:10) {
  for (sm in seq(from = -4,to = 4,by = 1)) {
    cat(">>> predicting week number",wn," // smooth_coeff",sm,"...\n")
    #pred = predict_wn_model(week_number = wn , smooth_coeff = sm, linear_coeff = linear_coeff )
    pred = predict_equal(week_number = wn , smooth_coeff = sm)
    #pred = predict_weight(week_number = wn , smooth_coeff = sm) 
    
    if (wn == 0) {
      cat(">>> adjusting ... ")
      pred_adj = adjust_prediction(week_number=wn,smooth_coeff=sm,pred=pred)
      fn_adj = paste(ff.getPath("elab_pred"),
                     "pred_equal__VIEW_",
                     wn,"_",sm,".csv",sep='')
      write.csv(pred_adj,
                quote=FALSE, 
                file=fn_adj ,
                row.names=FALSE)
    }
    
    ## write on disk 
    fn = paste(ff.getPath("elab_pred"),"pred_equal_",wn,"_",sm,".csv",sep='')
    write.csv(pred,
              quote=FALSE, 
              file=fn ,
              row.names=FALSE)
    cat(">>> pred: created",fn,"\n")
  }
}

# for (wn in 0:10) {
#   for (sm in seq(from = -4,to = 4,by = 1)) {
#     cat(">>> predicting week number",wn," // smooth_coeff",sm,"...\n")
#     pred_sub_wn_sm = predict_wn_smooth(week_number=wn, smooth_coeff=sm, verbose=T,debug=T)
#     stopifnot(sum(is.na(pred_sub_wn_sm))==0)
#     
#     ## write on disk 
#     fn = paste(ff.getPath("elab_pred"),"pred_NOPA_NOLA_",wn,"_",sm,".csv",sep='')
#     write.csv(pred_sub_wn_sm,
#               quote=FALSE, 
#               file=fn ,
#               row.names=FALSE)
#     cat(">>> pred: created",fn,"\n")
#   }
# }




