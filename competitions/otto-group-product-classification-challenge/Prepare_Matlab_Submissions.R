library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)

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

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#######
verbose = T

sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

pred.matlab = as.data.frame( fread(paste(getBasePath("data") , 
                                      "pred_nn_matlab_2.csv" , sep='')))


#pred.matlab = t(as.matrix(pred.matlab))
pred.matlab = as.matrix(pred.matlab)

sub = data.frame( id = sampleSubmission$id , 
                  Class_1 = pred.matlab[ , 1] , 
                  Class_2 = pred.matlab[ , 2] , 
                  Class_3 = pred.matlab[ , 3] , 
                  Class_4 = pred.matlab[ , 4] , 
                  Class_5 = pred.matlab[ , 5] , 
                  Class_6 = pred.matlab[ , 6] , 
                  Class_7 = pred.matlab[ , 7] , 
                  Class_8 = pred.matlab[ , 8] , 
                  Class_9 = pred.matlab[ , 9] 
  ) 

##### adjiusting 
sub.max.idx = apply(sub[,-1],1,function(x) which.max(x))

Class_j_mod = matrix(rep(0,nrow(sub)*9),nrow=nrow(sub),ncol = 9)

for (j in 1:length(sub$id)) 
  for (i in 1:9) {
    rec = sum(sub[j, 2:10]) - sub[j,(sub.max.idx[j]+1)]
    Class_j_mod[j,i] = ifelse(i == sub.max.idx[j] , sub[j,(i+1)],   (1 - sub[j,(sub.max.idx[j]+1)]) * sub[j,(i+1)] / rec )
  }

sub = data.frame ( id = sub$id , 
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

### storing on disk 
write.csv(sub,quote=FALSE, 
          file=paste(getBasePath("data"),"pred_nn_matlab_2_ready.csv",sep='') ,
          row.names=FALSE)

cat("<<<<< submission correctly stored on disk >>>>>\n") 
