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
  
  ## P29
  l = encodeCategoricalFeature (train[,29] , test[,29] , colname.prefix = "P29" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -29]
  test = test[ , -29]
  
  ## P35
  l = encodeCategoricalFeature (train[,35] , test[,35] , colname.prefix = "P35" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -35]
  test = test[ , -35]
  
  ## P25
  l = encodeCategoricalFeature (train[,25] , test[,25] , colname.prefix = "P25" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -25]
  test = test[ , -25]
  
  ## P36
  l = encodeCategoricalFeature (train[,36] , test[,36] , colname.prefix = "P36" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  train = train[ , -36]
  test = test[ , -36]
  
#     ## P31
#     l = encodeCategoricalFeature (train[,31] , test[,31] , colname.prefix = "P31" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -31]
#     test = test[ , -31]
#   
#     # P27
#     l = encodeCategoricalFeature (train[,27] , test[,27] , colname.prefix = "P27" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -27]
#     test = test[ , -27]
#   
#     ## P17
#     l = encodeCategoricalFeature (train[,17] , test[,17] , colname.prefix = "P17" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -17]
#     test = test[ , -17]
#     
#     ## P5
#     l = encodeCategoricalFeature (train[,5] , test[,5] , colname.prefix = "P5" , asNumeric=F)
#     tr = l[[1]]
#     ts = l[[2]]
#     
#     train = cbind(train,tr)
#     test = cbind(test,ts)
#     
#     train = train[ , -5]
#     test = test[ , -5]
  
  ## 
  list(train,y,test)
}

predict.train.k.folds = function (traindata , 
                                  y , 
                                  model.label = "RandomForest_Reg", 
                                  controlObject , 
                                  k = 6 ) {
  ### train set 
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
                                             model.label , 
                                             controlObject, 
                                             best.tuning = T)
    
    pred.1.train.fold = ifelse(pred.1.train.fold >= 1150 , pred.1.train.fold , 1150) 
    
    pred.1.train[folds == j] = pred.1.train.fold
  } ### end of k-fold 
  
  ## check 
  if (sum(is.na(pred.1.train)) > 0)
    stop("something wrong (NAs) in tip.train")
  
  pred.1.train
}

#######
verbose = T
source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

#controlObject <- trainControl(method = "boot", number = 200)
controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)

cat(">>> controlObject == repeatedcv 10 30 \n")
#cat(">>> controlObject == boot 200 \n")

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


##### MODELS DATA FRAME
model.boosting = data.frame(
  lev = c(1,2) , 
  model = as.character(c("BaggedTree_Reg","Enet_Reg")) , 
  removeOnlyZeroVariacePredictors = c(T,F) , 
  performVarianceAnalysisOnTrainSetOnly = c(T,T) , 
  correlationRhreshold = c(NA,NA) 
  )

print(model.boosting)

##### LEVEL 1  
level = 1 

##
l = featureSelect (train,test,y=y,
                   removeOnlyZeroVariacePredictors = model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors,
                   performVarianceAnalysisOnTrainSetOnly = model.boosting[model.boosting$lev==level,]$performVarianceAnalysisOnTrainSetOnly,
                   correlationRhreshold = model.boosting[model.boosting$lev==level,]$correlationRhreshold
                     )
traindata = l[[1]]
testdata = l[[2]]

## estimating res 
cat(">>> estimating residuals on training set of the first level .. \n")
pred.1.train = predict.train.k.folds (traindata , 
                                      y , 
                                      model.label = model.boosting[model.boosting$lev==level,]$model ,
                                      controlObject , 
                                      k = 6 )

res.1 = (y - pred.1.train) 
cat(">>> residuals in train set - mean =",mean(res.1)," sd =",sd(res.1)," ... \n")

##### LEVEL 2 
level = 2

l = featureSelect (train,test,y=y,
                   removeOnlyZeroVariacePredictors = model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors,
                   performVarianceAnalysisOnTrainSetOnly = model.boosting[model.boosting$lev==level,]$performVarianceAnalysisOnTrainSetOnly,
                   correlationRhreshold = model.boosting[model.boosting$lev==level,]$correlationRhreshold
)
traindata = l[[1]]
testdata = l[[2]]

### finding te best model fitting residuals .... 
cat(">> finding te best model fitting residuals .... \n")
RegModels = c("Average" , "Mode",  
              "LinearReg", "RobustLinearReg", 
              "PLS_Reg" , "Ridge_Reg" , "Enet_Reg" , 
              "KNN_Reg", 
              "SVM_Reg", 
              "BaggedTree_Reg"
              , "RandomForest_Reg"
              , "Cubist_Reg" 
              #, "NNet"
) 

cat("resampling = repeated cv 10 30",
    " - removeOnlyZeroVariacePredictors=",model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors,
    " - correlationRhreshold =",model.boosting[model.boosting$lev==level,]$correlationRhreshold,
    "\n")

l = trainAndPredict.kfold.reg (k = 6,traindata,
                               res.1,
                               RegModels,
                               controlObject)
model.winner = l[[1]]
.grid = l[[2]]
perf.kfold = l[[3]]

### results 
cat("****** RMSE - each model/fold ****** \n")
print(perf.kfold)
cat("\n****** RMSE - mean ****** \n")
print(.grid)
cat("\n>>>>>>>>>>>> The winner is ... ",model.winner,"\n")






