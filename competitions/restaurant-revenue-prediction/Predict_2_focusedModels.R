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

RegModels = c("Average" , "Mode",  
              "LinearReg",  
              "PLS_Reg" , 
              "BaggedTree_Reg" , 
              "RandomForest_Reg") 

controlObject <- trainControl(method = "boot", number = 200)

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

cat("\nRegression models:\n")
print(RegModels)

####### basic feature processing 
l = buildData.basic(train.raw , test.raw)
train = l[[1]]
y = l[[2]]
test = l[[3]]

####### feature selection <<<<<<<<<<<<<<
l = featureSelect (train,test,
                   removeOnlyZeroVariacePredictors=T,
                   removePredictorsMakingIllConditionedSquareMatrix = F, 
                   removeHighCorrelatedPredictors = F, 
                   featureScaling = F)
traindata = l[[1]]
testdata = l[[2]]

### k-fold 
l = trainAndPredict.kfold.reg (k = 6,traindata,y,RegModels,controlObject)
model.winner = l[[1]]
.grid = l[[2]]
perf.kfold = l[[3]]

### results 
if (verbose) {
  cat("****** RMSE - each model/fold ****** \n")
  print(perf.kfold)
  cat("\n****** RMSE - mean ****** \n")
  print(.grid)
  cat("\n>>>>>>>>>>>> The winner is ... ",model.winner,"\n")
}

### making prediction - bagged tree  
if (verbose) cat("Training on test data and making prediction w/ bagged trees .. \n")
pred = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            "BaggedTree_Reg" , 
                            controlObject, 
                            best.tuning = T)

pred = ifelse(pred >= 1150 , pred , 1150) ## TODO better 

### storing on disk 
write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_bagged_tree.csv",sep='') ,
          row.names=FALSE)

### making prediction - random forest 
if (verbose) cat("Training on test data and making prediction w/ random forest  .. \n")
pred = reg.trainAndPredict( y , 
                            traindata , 
                            testdata , 
                            "RandomForest_Reg" , 
                            controlObject, 
                            best.tuning = T)

pred = ifelse(pred >= 1150 , pred , 1150) ## TODO better 

### storing on disk 
write.csv(data.frame(Id = test.raw$Id , Prediction = pred),
          quote=FALSE, 
          file=paste(getBasePath("data"),"mySub_random_forest.csv",sep='') ,
          row.names=FALSE)
