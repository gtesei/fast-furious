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
source(paste0( getBasePath("process") , "/Transform_Lib.R"))

#########
require(xgboost)
require(methods)

#train = read.csv('data/train.csv',header=TRUE,stringsAsFactors = F)
#test = read.csv('data/test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

train = train[,-ncol(train)]
####

trans.scal <- preProcess(rbind(train,test),
                        method = c("center", "scale") )

trans.scal

train = predict(trans.scal,train)
test = predict(trans.scal,test)
######
x = rbind(train,test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "eta" = 0.05,  ## suggested in ESLII
              "gamma" = 0.5,  
              "max_depth" = 10, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 8)

cat(">>Train the model ... \n")
# Train the model
nround = 400
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

## feature importance 
importance <- xgb.importance(colnames(train), model = bst)
head(importance)

xgb.plot.importance(importance_matrix = importance)

write.table(importance,
            "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/otto-group-product-classification-challenge/xgboost_feat_imp.csv",
            sep=",",
            row.names=FALSE,
            quote=FALSE)

## saving a dataset with reduced set of features 
n.reduced = 40 
train.reduced = train[, importance$Feature[1:n.reduced]]
test.reduced = test[, importance$Feature[1:n.reduced]]

cat(">>> storing on disk for octave ... \n")
write.table(train.reduced,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_train_reduced_xg.csv") ,
            row.names=FALSE,
            col.names=FALSE)

write.table(test.reduced,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"oct_test_reduced_xg.csv") ,
            row.names=FALSE,
            col.names=FALSE)

cat(">>> storing on disk for R ... \n")
write.table(train.reduced,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"train_reduced_xg.csv") ,
            row.names=FALSE)

write.table(test.reduced,
            quote=FALSE, 
            file=paste0(getBasePath("data"),"test_reduced_xg.csv") ,
            row.names=FALSE)



