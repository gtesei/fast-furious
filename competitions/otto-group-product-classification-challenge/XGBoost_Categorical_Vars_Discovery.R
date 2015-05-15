library(caret)
library(Hmisc)
library(data.table)
library(kernlab)
library(subselect)
library(plyr)
library(binhf)
library(fBasics)
require(xgboost)
require(methods)

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

source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))
###
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

train = train[,-ncol(train)]
trans.scal <- preProcess(rbind(train,test),
                         method = c("center", "scale") )
trans.scal
train.proc = predict(trans.scal,train)
test.proc = predict(trans.scal,test)
######## base parameters 
verbose = T

# Set parameters
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "eta" = 0.3,  ## default 0.3 
              "gamma" = 0, # default 0   
              "max_depth" = 6, # defualt 0  
              #"subsample" = 0.5 , ## suggested in ESLII
              "nthread" = 10
              
              #"min_child_weight" = 1 , 
              #"colsample_bytree" = 0.5, 
              #"max_delta_step" = 5  
)

cat(">>>>> parameters:\n") 
print(param)
cv.nround = 300

######## grid 
grid = data.frame(
      feature = 1:ncol(train),
      values = NA, 
      has.fractions = NA, 
      
      xgboost.test.perf = NA, 
      xgboost.test.sd = NA, 
      xgboost.early.stop = NA, 
      
      xgboost.test.perf.ref = NA, 
      xgboost.test.sd.ref = NA, 
      xgboost.early.stop.ref = NA, 
      
      couldbe.categorical = NA, 
      is.categorical = NA
  )

for (j in 1:ncol(train)) {
  ##cat(">> processing feat",j," ...\n")
  val = unique(train[,j])
  grid[j,]$values = list(val)
  grid[j,]$has.fractions = sum(! val%%1 == 0)
}

######## Reference performance  
x = rbind(train.proc,test.proc)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Run Cross Valication
cat(">>Cross validation ... \n")

cat(">> cv.nround: ",cv.nround,"\n") 
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

xgboost.test.perf.ref = min(bst.cv$test.mlogloss.mean)
idx = which(bst.cv$test.mlogloss.mean == min(bst.cv$test.mlogloss.mean) )
xgboost.test.sd.ref = bst.cv[idx,]$test.mlogloss.std
xgboost.early.stop.ref = idx
cat(">> xgboost.test.perf.ref: ",xgboost.test.perf.ref," \n") 
cat(">> xgboost.test.sd.ref: ",xgboost.test.sd.ref," \n") 
cat(">> xgboost.early.stop.ref: ",xgboost.early.stop.ref," \n") 

#### processing featurtes 
for (j in 1:ncol(train)) {
  cat(">> processing feat",j," ...\n")
  
  ## encode 
  l = encodeCategoricalFeature (train[,j] , test[,j] , colname.prefix = paste0("feat_",j) , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train.j = cbind(train.proc,tr)
  test.j = cbind(test.proc,ts)
  
  train.j = train.j[ , -j]
  test.j = test.j[ , -j]
  
  ## prepare data for xgboost 
  x = rbind(train.j,test.j)
  x = as.matrix(x)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  trind = 1:length(y)
  teind = (nrow(train.j)+1):nrow(x)
  
  bst.cv.j = xgb.cv(param=param, data = x[trind,], label = y, 
                  nfold = 3, nrounds=cv.nround)
  
  ## update grid 
  cat(">>> updating grid ... \n") 
  
  grid[j,]$xgboost.test.perf = min(bst.cv.j$test.mlogloss.mean)
  idx.j = which(bst.cv.j$test.mlogloss.mean == min(bst.cv.j$test.mlogloss.mean) ) 
  grid[j,]$xgboost.test.sd = bst.cv.j[idx.j,]$test.mlogloss.std
  grid[j,]$xgboost.early.stop = idx.j
  
  grid[j,]$xgboost.test.perf.ref = xgboost.test.perf.ref 
  grid[j,]$xgboost.test.sd.ref = xgboost.test.sd.ref 
  grid[j,]$xgboost.early.stop.ref = xgboost.early.stop.ref 
  
  grid[j,]$couldbe.categorical = ifelse( grid[j,]$xgboost.test.perf < grid[j,]$xgboost.test.perf.ref , T, F) 
  
  if (grid[j,]$couldbe.categorical) {
    delta.j = xgboost.test.perf.ref - grid[j,]$xgboost.test.perf
    grid[j,]$is.categorical = ifelse( delta.j > grid[j,]$xgboost.test.sd , T , F) 
  } else {
    grid[j,]$is.categorical = F
  }
  
  print(grid[j,-2])
  gc()
  
  ### storing on disk 
  cat(">> storing on disk ... \n")
  write.csv(grid[,-2],
            file = paste( getBasePath("data") , "xgboost_grid_cat_vs_numeric.csv" , sep='' ), 
            quote = FALSE,
            row.names = FALSE)
}

