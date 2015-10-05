library(fastfurious)
library(data.table)
library(plyr)
library(Hmisc)

### FUNCS 
do_week_number = function(week_number , smooth_coeffs = seq(from = -4,to = 4,by = 1) , verbose=T , debug = F) {
  
  if (verbose) cat(">>> do_wn: processing week_number:",week_number, " // smooth_coeffs:",smooth_coeffs,"...\n")
  coupon_labels = getLabels(week_number=week_number , verbose=verbose)
  
  ## coupon_<WN>.csv
  coupon_wn = data.frame(COUPON_ID_hash = coupon_labels$coupons) 
  write.csv(coupon_wn,
            quote=FALSE, 
            file=paste(ff.getPath("elab_labels"),"coupon_",week_number,".csv",sep='') ,
            row.names=FALSE)
  
  if (week_number>0) { ## week 0 labels if the goal of the competition ;-)  
    ## coupon_labels_<WN>.csv
    write.csv(coupon_labels$labels,
              quote=FALSE, 
              file=paste(ff.getPath("elab_labels"),"coupon_labels_",week_number,".csv",sep='') ,
              row.names=FALSE)
  }
  
  if (verbose) cat(">>> do_wn: created:",paste("coupon_labels_",week_number,".csv",sep=''), "and",
                   paste("coupon_",week_number,".csv",sep='')," under ",ff.getPath("elab_labels"),"\n")
  
  ## train_<WN>_<SMOOTH>.csv
  for (sm in smooth_coeffs) {
    fn = paste(ff.getPath("elab_train"),"train_",week_number,"_",sm,".csv",sep='')
    if (verbose) cat(">>> do_wn: creating:",fn," ... \n")
    
    user_vect = getUserVector(week_number=week_number , exp_smooth = sm, verbose=verbose, debug=debug)
    write.csv(user_vect,
              quote=FALSE, 
              file=fn ,
              row.names=FALSE)
  }
  if (verbose) cat(">>> do_wn: processed week_number:",week_number, " // smooth_coeffs:",smooth_coeffs,"...\n")
}

### FAST-FURIOUS 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'code' , sub_path = 'competitions/coupon-purchase-prediction')

### GLOBAL CONFIG 
debug = F

### DATA & COUPON VECTORS 
source(paste0(ff.getPath('code'),'make_coupon_vector.R'))
ff.bindPath(type = 'elab' , sub_path = 'dataset/coupon-purchase-prediction/elab' , createDir = T)
ff.bindPath(type = 'elab_train' , sub_path = 'dataset/coupon-purchase-prediction/elab/train' , createDir = T)
ff.bindPath(type = 'elab_labels' , sub_path = 'dataset/coupon-purchase-prediction/elab/labels' , createDir = T)

### PROCESSING 
wn.first = 0
wn.last = 10 

for (wn in wn.first:wn.last) {
  for (sm in seq(from = -4,to = 4,by = 1)) {
    do_week_number(week_number = wn , smooth_coeffs = sm, verbose = T , debug = F)
  }
}




