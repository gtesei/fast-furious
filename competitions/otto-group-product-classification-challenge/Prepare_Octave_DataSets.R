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
#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

#train = as.data.frame( fread(paste(getBasePath("data") , 
#                                      "encoded_train.csv" , sep=''))) 

#y = as.data.frame( fread(paste(getBasePath("data") , 
#                                   "encoded_y.csv" , sep=''))) 

#y = as.integer(y$y)-1

#test = as.data.frame( fread(paste(getBasePath("data") , 
#                                      "encoded_test.csv" , sep='')))

train = as.data.frame( fread(paste(getBasePath("data") , 
                                   "train.csv" , sep=''))) 

test = as.data.frame( fread(paste(getBasePath("data") , 
                                  "test.csv" , sep='')))


########
verbose = T

#########
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = as.numeric(gsub('Class_','',y))

train = train[,-ncol(train)]

########## store on disk for octave 
cat(">>> storing on disk for octave ... \n")
write.table(data.frame(y = y),
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_y.csv") ,
            row.names=FALSE, 
            col.names=FALSE)

write.table(train,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_train.csv") ,
            row.names=FALSE,
            col.names=FALSE)

write.table(test,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_test.csv") ,
            row.names=FALSE,
            col.names=FALSE)