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

######## first build the "tip" features, i.e. Enet_Reg applying  caret’s nearZeroVar function both in test set and training set 
#### feature selection <<<<<<<<<<<<<<
l = featureSelect (train,test)
traindata = l[[1]]
testdata = l[[2]]

#### test set 
cat(">> building tip feature on test set ... \n")
tip.test = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            "Enet_Reg" , 
                            controlObject, 
                            best.tuning = T)

tip.test = ifelse(tip.test >= 1150 , tip.test , 1150)

test$tip = tip.test

#### training set 
k = 6
folds = kfolds(k,dim(traindata)[1])

tip.train = rep(NA,dim(traindata)[1])

for(j in 1:k) {  
  if (verbose) cat("--k-fold:: ",j, "/",k , "\n")
  traindata.train <- traindata[ folds != j,]
  traindata.y.train = y[folds != j]
  
  traindata.xval <- traindata[folds == j,]
  traindata.y.xval = y[folds == j]
  
  ###
  tip.train.fold = reg.trainAndPredict( traindata.y.train , 
                           traindata.train , 
                           traindata.xval , 
                           "Enet_Reg" , 
                           controlObject, 
                           best.tuning = T)
  
  tip.train.fold = ifelse(tip.train.fold >= 1150 , tip.train.fold , 1150)
  
  tip.train[folds == j] = tip.train.fold
} ### end of k-fold 

## check 
if (sum(is.na(tip.train)) > 0)
  stop("something wrong (NAs) in tip.train")

train$tip = tip.train

######## then predict w/ BaggedTree_Reg removing only zero variance predictors
cat("predicting w/ BaggedTree_Reg removing only zero variance predictors ... \n")
#### feature selection <<<<<<<<<<<<<<
l = featureSelect (train,test,
                   removeOnlyZeroVariacePredictors=T,
                   removePredictorsMakingIllConditionedSquareMatrix = T, 
                   removeHighCorrelatedPredictors = T, 
                   featureScaling = T)
traindata = l[[1]]
testdata = l[[2]]

if (verbose) cat("Training on test data and making prediction w/ bagged trees .. \n")
pred = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            "BaggedTree_Reg" , 
                            controlObject, 
                            best.tuning = T)

pred = ifelse(pred >= 1150 , pred , 1150) 

### storing on disk 
write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_tip_1.csv",sep='') ,
          row.names=FALSE)
