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
  
  ###
  toRemove = NULL
  
  ## P35
  l = encodeCategoricalFeature (train[,35] , test[,35] , colname.prefix = "P35" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  toRemove = c(toRemove,35)
  #   train = train[ , -35]
  #   test = test[ , -35]
  
  ## P25
  l = encodeCategoricalFeature (train[,25] , test[,25] , colname.prefix = "P25" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  toRemove = c(toRemove,25)
  #   train = train[ , -25]
  #   test = test[ , -25]
  
  ## P36
  l = encodeCategoricalFeature (train[,36] , test[,36] , colname.prefix = "P36" , asNumeric=F)
  tr = l[[1]]
  ts = l[[2]]
  
  train = cbind(train,tr)
  test = cbind(test,ts)
  
  toRemove = c(toRemove,36)
  #   train = train[ , -36]
  #   test = test[ , -36]
  
  ## REMOVE
  train = train[ , -toRemove]
  test = test[ , -toRemove]
  
  ## high revenue combinations 
  train$hrc = ifelse(train$P1 == 5 & train$P8 == 3 , 1 , 0) 
  test$hrc = ifelse(test$P1 == 5 & test$P8 == 3 , 1 , 0) 
  
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

controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)

cat(">>> resampling:: repeatedcv 10 30 \n")

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

cat("****** Available regression models ******\n") 
print(RegModels)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

## init levels (you can load the initial grid from file)
model.boosting = expand.grid(
  lev = 1 ,  
  model = "BaggedTree_Reg" , 
  #model = "Average" , 
  need.finding = F , 
  removeOnlyZeroVariacePredictors = T , 
  performVarianceAnalysisOnTrainSetOnly = T , 
  correlationRhreshold = NA, 
  res.train.mean = NA , 
  res.train.sd = NA , 
  RMSE.train.kfold = NA , 
  RMSE.xval.winner = NA 
)

### predicting 
level = 1
cat("******** PROCESSING LEVEL ",level,"********\n")

## unroll parameters  
models = as.character(model.boosting[model.boosting$lev==level,]$model)
need.finding = as.logical(model.boosting[model.boosting$lev==level,]$need.finding)
removeOnlyZeroVariacePredictors = as.logical(model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors) 
performVarianceAnalysisOnTrainSetOnly = as.logical(model.boosting[model.boosting$lev==level,]$performVarianceAnalysisOnTrainSetOnly) 
correlationRhreshold = as.numeric(model.boosting[model.boosting$lev==level,]$correlationRhreshold) 
res.train.mean = as.numeric(model.boosting[model.boosting$lev==level,]$res.train.mean) 
res.train.sd = as.numeric(model.boosting[model.boosting$lev==level,]$res.train.sd) 
RMSE.train.kfold = as.numeric(model.boosting[model.boosting$lev==level,]$RMSE.train.kfold) 
RMSE.xval.winner = as.numeric(model.boosting[model.boosting$lev==level,]$RMSE.xval.winner) 

variants = length(models)

####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]

## process data 
l = featureSelect (train,test,y=y,
                   removeOnlyZeroVariacePredictors = model.boosting[model.boosting$lev==level,]$removeOnlyZeroVariacePredictors,
                   performVarianceAnalysisOnTrainSetOnly = model.boosting[model.boosting$lev==level,]$performVarianceAnalysisOnTrainSetOnly,
                   correlationRhreshold = model.boosting[model.boosting$lev==level,]$correlationRhreshold
)
traindata = l[[1]]
testdata = l[[2]]

## pred.1 
pred.1 = 
  reg.trainAndPredict( y , 
                       traindata , 
                       testdata , 
                       model.boosting[model.boosting$lev==level,]$model , 
                       controlObject, 
                       best.tuning = T)

## estimating res 
cat(">>> estimating residuals on training set of the first level .. \n")
pred.1.train = predict.train.k.folds (traindata , 
                                      y , 
                                      model.label = model.boosting[model.boosting$lev==level,]$model ,
                                      controlObject , 
                                      k = 6 )

res.1 = (y - pred.1.train) 
cat(">>> residuals in train set - mean =",mean(res.1)," sd =",sd(res.1)," ... \n")

#### updating grid 
model.boosting[model.boosting$lev==level,]$res.train.mean = mean(res.1)
model.boosting[model.boosting$lev==level,]$res.train.sd = sd(res.1)
model.boosting[model.boosting$lev==level,]$RMSE.train.kfold = RMSE(pred = pred.1.train, obs = y)
model.boosting[model.boosting$lev==level,]$RMSE.xval.winner = NA

#### storing 
cat(">>> Storing on disk ... \n")

write.csv(model.boosting,
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting0_grid.csv",sep='') ,
          row.names=FALSE)

write.csv(data.frame(Id = test.raw$Id , Pred_1 = pred.1 ),
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting0_pred.test.csv",sep='') ,
          row.names=FALSE)

write.csv(data.frame(Id = train.raw$Id , res.1 = res.1),
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting0_res.csv",sep='') ,
          row.names=FALSE)

write.csv(data.frame(Id = test.raw$Id , Prediction = pred.1 ),
          quote=FALSE, 
          file=paste(getBasePath("data"),"model_boosting0_____pred.csv",sep='') ,
          row.names=FALSE)


