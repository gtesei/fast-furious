library(fastfurious)
library(data.table)
library(plyr)


### DATA 
ff.setBasePath(path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious/')
ff.bindPath(type = 'elab_meta' , sub_path = 'dataset/coupon-purchase-prediction/elab/meta' , createDir = T)

### PROC
w_in = 28
w_end = 1 
smooth_coeff = seq(from = -4,to = 4,by = 1)
for (sm in smooth_coeff) {
  cat(">>> processing smooth_coeff ",sm,"from",w_in,"to",w_end,"...\n") 
  matList = list()
  for (i in w_end:w_in) {
    fn = paste(ff.getPath("elab_meta"),"w_",i,"_",sm,".csv",sep='')
    cat("> loading ",fn,"...\n") 
    matList[[i]] =  as.data.frame( fread( fn , stringsAsFactors = F))
  }
  outMat = data.frame(USER_ID_hash=matList[[1]]$USER_ID_hash , LIST.CAPSULE_TEXT = 0 , LIST.GENRE_NAME = 0 , LIST.LARGE_AREA_NAME = 0, 
                      AREA.SMALL_AREA_NAME=0, LIST.BID_CATALOG_PRICE=0, LIST.BID_PRICE_RATE=0,LIST.WDISPPERIOD=0)
  for (i in 1:nrow(outMat)) {
    uid = outMat[i,]$USER_ID_hash
    alist = list() 
    nn = 1
    for (j in seq_along(matList)) {
      if (sum(matList[[j]][matList[[j]]$USER_ID_hash==uid,2:8]) >0) {
        alist[[nn]] <- j 
        nn <- nn + 1 
      }
    }
    if (length(alist)>0) {
      tt = data.frame(LIST.CAPSULE_TEXT = rep(0,length(alist)) , LIST.GENRE_NAME = rep(0,length(alist)) , 
                      LIST.LARGE_AREA_NAME = rep(0,length(alist)), AREA.SMALL_AREA_NAME=rep(0,length(alist)), 
                      LIST.BID_CATALOG_PRICE=rep(0,length(alist)), 
                      LIST.BID_PRICE_RATE=rep(0,length(alist)),LIST.WDISPPERIOD=rep(0,length(alist)))
      for (j in seq_along(alist)) {
        tt[j,] <- matList[[alist[[j]]]][matList[[alist[[j]]]]$USER_ID_hash==uid,2:8]
      }
      outMat[i,2:8] <- apply(X = tt,MARGIN = 2, FUN = mean)
    } else {
      outMat[i,2:8] <- rep(1,8)
    }
  }
  ## write on disk 
  fn = paste(ff.getPath("elab_meta"),"avg_",sm,".csv",sep='')
  write.csv(outMat,
            quote=FALSE, 
            file=fn ,
            row.names=FALSE)
  gc()
}



