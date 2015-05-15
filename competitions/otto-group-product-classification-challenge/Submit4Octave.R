library(data.table)

getBasePath = function (type = "data") {
  ret = ""
  base.path1 = ""
  base.path2 = ""
  
  if(type == "data") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/otto-group-product-classification-challenge/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/otto-group-product-classification-challenge"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/otto-group-product-classification-challenge/"
  } else if (type == "process") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/data_process"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/data_process/"
  } else {
    stop("unrecognized type.")
  }
  
  if (file.exists(base.path1))  {
    ret = paste0(base.path1,"/")
  } else {
    ret = base.path2
  }
  
  ret
}

##
sub.file = "pred_NN.csv"
##submit4Octave = function (sub.file = "sub_xgb.csv" ) {  
  cat (">>> processing ",sub.file,"... \n")
  
  base.fn = substr(x = sub.file, start = 1, stop = (nchar(sub.file)-4) ) 
  
  
  sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                                "sampleSubmission.csv" , sep='')))
  
  sub = as.data.frame( fread(paste(getBasePath("data") , 
                                   sub.file , sep='')))

  if (F) { 
  # #### adjiusting --- 
  sub.max.idx = apply(sub,1,function(x) which.max(x))
  
  Class_j_mod = matrix(rep(0,nrow(sub)*9),nrow=nrow(sub),ncol = 9)
  
  for (j in 1:nrow(sub)) {
    for (i in 1:9) {
      norm = sum(sub[j, ]) - sub[j,sub.max.idx[j]]
      Class_j_mod[j,i] = ifelse(i == sub.max.idx[j] , sub[j,i],   (1 - sub[j,sub.max.idx[j]]) * sub[j,i] / norm  )
    }
    ## some check 
    if ( abs( sum(Class_j_mod[j,]) - 1 ) > 0.001 ) stop("sum != 1")
    if ( sum(Class_j_mod[j,] < 0) > 1) stop("some element <0") 
  }
  
  sub.2 = data.frame ( id = sampleSubmission$id , 
                       Class_1 = Class_j_mod[ , 1] , 
                       Class_2 = Class_j_mod[ , 2] , 
                       Class_3 = Class_j_mod[ , 3] , 
                       Class_4 = Class_j_mod[ , 4] , 
                       Class_5 = Class_j_mod[ , 5] , 
                       Class_6 = Class_j_mod[ , 6] , 
                       Class_7 = Class_j_mod[ , 7] , 
                       Class_8 = Class_j_mod[ , 8] , 
                       Class_9 = Class_j_mod[ , 9] 
  )
  
  sub.2[,2:10] = format(sub.2[,2:10] , digits=2,scientific=F) # shrink the size of submission
  
  ### storing on disk 
  write.csv(sub.2,quote=FALSE, 
            file=paste(getBasePath("data"),base.fn,"_adjiusted.csv",sep='') ,
            row.names=FALSE)
  
  cat("<<<<< submission correctly stored on disk >>>>>\n") 
  
  
  } else {
    sub$id = sampleSubmission$id 
    colnames(sub)[1:9] = colnames(sampleSubmission)[2:10]
    
    write.csv(sub,quote=FALSE, 
              file=paste(getBasePath("data"),base.fn,"_adjiusted.csv",sep='') ,
              row.names=FALSE)
    
    cat("<<<<< submission correctly stored on disk >>>>>\n")
  }
  
  
  
##  sub.2
##}


##sub = submit4Octave  (sub.file = "pred_NN_reduced.csv" )



