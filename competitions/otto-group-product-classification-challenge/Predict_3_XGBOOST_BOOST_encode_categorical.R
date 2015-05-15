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

grid = as.data.frame( fread(paste(getBasePath("data") , 
                                  "xgboost_grid_cat_vs_numeric.csv" , sep='')))

ftoenc = as.numeric( (grid[grid$is.categorical & ! is.na(grid$is.categorical) ,]$feature) )

########
verbose = T
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

#########
require(xgboost)
require(methods)

train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

train = train[,-ncol(train)]

## encodinng 
cat(">> found ",length(ftoenc)," features to encode as categorical .. \n")
toRemove = NULL
for (i in ftoenc) {
  cat(">> enacoding feat",i," ..\n")
  
  l = encodeCategoricalFeature (train[,i] , test[,i] , colname.prefix = paste0("feat_",i) , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  toRemove = c(toRemove,i)
}

train = train[ , -toRemove]
test = test[ , -toRemove]

## feature scaling 
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
              "max_depth" = 25, 
              "subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10, 
              
              "min_child_weight" = 1 , 
              "colsample_bytree" = 0.5, 
              "max_delta_step" = 1
              )

cat(">>Params:\n")
print(param)

# Run Cross Valication
cat(">>Cross validation ... \n")

inCV = T
early.stop = cv.nround = 2000

while (inCV) {
  
  cat(">> cv.nround: ",cv.nround,"\n") 
  bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                  nfold = 5, nrounds=cv.nround)
  print(bst.cv)
  early.stop = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
  cat(">> early.stop: ",early.stop," [test.mlogloss.mean:",bst.cv[early.stop,]$test.mlogloss.mean,"]\n") 
  if (early.stop < cv.nround) {
    inCV = F
    cat(">> stopping [early.stop < cv.nround=",cv.nround,"] ... \n") 
  } else {
    cat(">> redo-cv [early.stop == cv.nround=",cv.nround,"] with 2 * cv.nround ... \n") 
    cv.nround = cv.nround * 2 
  }
  gc()
}

cat(">>Train the model ... \n")
# Train the model
nround = early.stop
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
#write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)
write.csv(pred,file=paste(getBasePath("data") , 
                          "sub_xgb_boost_3gen_categorical.csv" , sep=''), quote=FALSE,row.names=FALSE)

