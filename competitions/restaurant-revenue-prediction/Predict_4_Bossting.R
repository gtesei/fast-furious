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
    base.path1 = "C:/docs/ff/gitHub/fast-furious/dataset/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/dataset/restaurant-revenue-prediction/"
  } else if (type == "code") {
    base.path1 = "C:/docs/ff/gitHub/fast-furious/competitions/restaurant-revenue-prediction"
    base.path2 = "/Users/gino/kaggle/fast-furious/gitHub/fast-furious/competitions/restaurant-revenue-prediction/"
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

buildData.basic = function(train.raw , test.raw) {
  ## remove id 
  train = train.raw[ , -1] 
  test = test.raw[ , -1] 
  
  ## 2014 should be the target year ... so use open date to misure the number of years between open date and the target year 
  train$years.to.target = 2014 - year(as.Date( train.raw[,2] , "%m/%d/%Y"))
  test$years.to.target = 2014 - year(as.Date( test.raw[,2] , "%m/%d/%Y"))
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## City Group 
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "city.group" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## Type
  l = encodeCategoricalFeature (train[,1] , test[,1] , colname.prefix = "type" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -1]
  test = test[ , -1]
  
  ## extracting y 
  y = train[,38]
  train = train[,-38]
  
  list(train,y,test)
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "boot", number = 200)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]

### first we compute the response both in training set (6-folds-like procedure) and test set 
### w/ BaggedTree_Reg removing only zero variance predictors
#### feature selection <<<<<<<<<<<<<<
l = featureSelect (train,test,
                   removeOnlyZeroVariacePredictors=T,
                   removePredictorsMakingIllConditionedSquareMatrix = F, 
                   removeHighCorrelatedPredictors = F, 
                   featureScaling = F)
traindata = l[[1]]
testdata = l[[2]]

### test set 
if (verbose) cat("Training on test data and making prediction w/ bagged trees .. \n")
pred.1.test = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            "BaggedTree_Reg" , 
                            controlObject, 
                            best.tuning = T)

pred.1.test = ifelse(pred.1.test >= 1150 , pred.1.test , 1150) 

### train set 
k = 6
folds = kfolds(k,dim(traindata)[1])

pred.1.train = rep(NA,dim(traindata)[1])

for(j in 1:k) {  
  if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
  traindata.train <- traindata[ folds != j,]
  traindata.y.train = y[folds != j]
  
  traindata.xval <- traindata[folds == j,]
  traindata.y.xval = y[folds == j]
  
  ###
  pred.1.train.fold = reg.trainAndPredict( traindata.y.train , 
                                        traindata.train , 
                                        traindata.xval , 
                                        "BaggedTree_Reg" , 
                                        controlObject, 
                                        best.tuning = T)
  
  pred.1.train.fold = ifelse(pred.1.train.fold >= 1150 , pred.1.train.fold , 1150) 
  
  pred.1.train[folds == j] = pred.1.train.fold
} ### end of k-fold 

## check 
if (sum(is.na(pred.1.train)) > 0)
  stop("something wrong (NAs) in tip.train")

### we compute the residual, i.e. the difference between the observed value and such prediction in train set
cat(">>> computing residuals in train set ... \n")
res = (y - pred.1.train) 

# we fit an Enet_Reg applying caret’s nearZeroVar function using the residuals as response and predict on test set 
cat("predicting Enet_Reg applying caret’s nearZeroVar function using the residuals as response ... \n")
#### feature selection <<<<<<<<<<<<<<
l = featureSelect (train,test)
traindata = l[[1]]
testdata = l[[2]]

if (verbose) cat("Training on test data and making prediction w/ bagged trees .. \n")
pred.res = reg.trainAndPredict( res , 
                            traindata , 
                            testdata , 
                            "Enet_Reg" , 
                            controlObject, 
                            best.tuning = T)

### update the predicted value adding to the previous prediction (BaggedTree_Reg) to the predicted value of residuals
pred = pred.1.test + pred.res

pred = ifelse(pred >= 1150 , pred , 1150)

### storing on disk 
write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_boost_1.csv",sep='') ,
          row.names=FALSE)