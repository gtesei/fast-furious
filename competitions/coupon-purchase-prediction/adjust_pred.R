library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 

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

### BOOSTING VIEWED COUPONS 
coupon_visit$I_DATE = as.Date(coupon_visit$I_DATE)
colnames(coupon_visit)[5] = 'COUPON_ID_hash'
coupon_visit_test = merge(x=coupon_visit,y=coupon_list_test.meta[,c('COUPON_ID_hash','DISPFROM')] , by='COUPON_ID_hash' , all=F)
coupon_visit_test = coupon_visit_test[,c('USER_ID_hash','COUPON_ID_hash')]
coupon_visit_test_uniq = ddply(coupon_visit_test,.(USER_ID_hash,COUPON_ID_hash))
stopifnot(nrow(coupon_visit_test_uniq)==nrow(coupon_visit_test))

coupon_visit_test[coupon_visit_test$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',]


## pred
adjust_prediction = function(week_number,smooth_coeff) {
  #week_number = 0 
  #smooth_coeff = -4 
  #fn = paste(ff.getPath("elab_pred"),"pred_NOWP_",week_number,"_",smooth_coeff,".csv",sep='')
  #fn = paste(ff.getPath("elab_pred"),"pred_NOWP_NOPA_NOLA_",week_number,"_",smooth_coeff,".csv",sep='')
  fn = paste(ff.getPath("elab_pred"),"pred_NOPA_NOLA_",week_number,"_",smooth_coeff,".csv",sep='')
  pred = as.data.frame( fread( fn , stringsAsFactors = F))
  
  ## adjust 
  ord_coup = unlist(strsplit(x = pred[pred$USER_ID_hash=='7b775ad4772cf5ec9cbe9add131e63e5',]$PURCHASED_COUPONS , split = ' '))
  
  ord_coup_mat = ddply(pred, .(USER_ID_hash) , function(x) {
    aa = unlist(strsplit(x = x$PURCHASED_COUPONS , split = ' '))
    aa[1:10]
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
  
  ##
  #fn = paste(ff.getPath("elab_pred"),"pred_NOWP_VIEW_",week_number,"_",smooth_coeff,".csv",sep='')
  #fn = paste(ff.getPath("elab_pred"),"pred_NOWP_NOPA_NOLA_VIEW_",week_number,"_",smooth_coeff,".csv",sep='')
  #fn = paste(ff.getPath("elab_pred"),"ppp2.csv",sep='')
  fn = paste(ff.getPath("elab_pred"),"pred_NOPA_NOLA_VIEW_",week_number,"_",smooth_coeff,".csv",sep='')
  
  write.csv(ord_coup_sub,
            quote=FALSE, 
            file=fn ,
            row.names=FALSE)
  
}


### PROCESSING 
wn = 0
for (sm in seq(from = -4,to = 4,by = 1)) {
  cat(">>> adjusting week number",wn," // smooth_coeff",sm,"...\n")
  adjust_prediction(week_number=wn,smooth_coeff=sm)
}









