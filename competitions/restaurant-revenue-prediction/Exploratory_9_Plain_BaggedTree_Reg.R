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

buildData.basic = function(train.raw , test.raw , features_to_encode , remove_enc) {
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
  
  ## encoding 
  toRemove = NULL
  
  for (i in features_to_encode) {
    
    ft = paste0("P",i)
    
    l = encodeCategoricalFeature (train[,i] , test[,i] , colname.prefix = ft , asNumeric=F)
    tr = l[[1]]
    ts = l[[2]]
    
    train = cbind(train,tr)
    test = cbind(test,ts)
    
    toRemove = c(toRemove,i)
  }
  
  ## high revenue combinations 
  train$hrc = ifelse(train$P1 == 5 & train$P8 == 3 , 1 , 0) 
  test$hrc = ifelse(test$P1 == 5 & test$P8 == 3 , 1 , 0) 
  
  ## removing 
  if (remove_enc) {
    train = train[ , -toRemove]
    test = test[ , -toRemove]
  }
  
  list(train,y,test)
}

####### 
verbose = T

source(paste0( getBasePath("process") , "/FeatureSelection_Lib.R"))
source(paste0( getBasePath("process") , "/Resample_Lib.R"))
source(paste0( getBasePath("process") , "/Regression_Lib.R"))
source(paste0( getBasePath("process") , "/FeatureEncode_Lib.R"))

controlObject <- trainControl(method = "repeatedcv", repeats = 10, number = 30)
#controlObject <- trainControl(method = "boot", number = 3000)

#cat(">>> resampling:: repeatedcv 10 30 \n")
cat(">>> resampling:: boot 3000 \n")

#######
sampleSubmission = as.data.frame( fread(paste(getBasePath("data") , 
                                              "sampleSubmission.csv" , sep='')))

train.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                       "train.csv" , sep=''))) 

test.raw = as.data.frame( fread(paste(getBasePath("data") , 
                                      "test.csv" , sep='')))

cat(">> training observations:",dim(train.raw)[1]," \n")

####### basic feature processing 
l = buildData.basic(train.raw , test.raw , c(35,25,36) , T)
train = l[[1]]
y = l[[2]]
test = l[[3]]

## process data 
l = featureSelect (train,test,y=y,
                   removeOnlyZeroVariacePredictors = T,
                   performVarianceAnalysisOnTrainSetOnly = T,
                   correlationRhreshold = NA , 
                   removePredictorsMakingIllConditionedSquareMatrix = F, 
                   removeHighCorrelatedPredictors = F, 
                   featureScaling = T
)
traindata = l[[1]]
testdata = l[[2]]

## pred.1 
# pred = 
#   reg.trainAndPredict( y , 
#                        traindata , 
#                        testdata , 
#                        "BaggedTree_Reg" , 
#                        controlObject, 
#                        best.tuning = T)


# fit <- train(y = y, x = traindata ,
#              method = "treebag",
#              trControl = controlObject)

fit <- train(y = y, x = traindata ,
             ethod = "qrf", trControl = controlObject, 
             ntree = 500)

pred = as.numeric( predict(fit , testdata )  )

featImp = varImp(fit , scale = F)

cat(">>> Storing on disk ... \n")

write.csv(data.frame(Id = test.raw$Id , Prediction = pred ),
          quote=FALSE, 
          file=paste(getBasePath("data"),"pred_qrf.csv",sep='') ,
          row.names=FALSE)



